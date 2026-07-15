"""Tests for racing/evolution.py — the genetic algorithm.

The GA never looks inside a genome; it only recombines and perturbs rows of a
(P, G) matrix guided by fitness rank. That makes its contracts very testable:
elitism must preserve the best rows bit-exact, selection must be greedy in the
large-k limit, and crossover/mutation must be pure recombination/perturbation
with exactly the configured statistics. Every RNG is seeded, so each test is
fully deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from racing.config import EvoConfig
from racing.evolution import next_generation, sigma_at, tournament_select


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def small_cfg(**overrides) -> EvoConfig:
    """A tiny EvoConfig for hand-built populations.

    next_generation reads elite / tournament_k / crossover_rate /
    mutation_prob / heavy_tail_prob / heavy_tail_scale from the config
    (population size comes from the genomes array itself).
    """
    base = dict(
        population=16, elite=2, tournament_k=3,
        crossover_rate=0.5, mutation_prob=0.10,
        sigma_init=0.20, sigma_decay=0.995, sigma_floor=0.02,
        heavy_tail_prob=0.10, heavy_tail_scale=5.0,
    )
    base.update(overrides)
    return EvoConfig(**base)


def random_population(pop: int, n_genes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((pop, n_genes)).astype(np.float32)


# ---------------------------------------------------------------------------
# sigma_at
# ---------------------------------------------------------------------------

def test_sigma_at_generation_zero_is_sigma_init():
    """decay**0 == 1, so the schedule must start exactly at sigma_init.

    If it didn't, the very first generation would explore at the wrong scale
    and the config knob would silently lie.
    """
    cfg = small_cfg(sigma_init=0.37, sigma_decay=0.99, sigma_floor=0.001)
    assert sigma_at(0, cfg) == pytest.approx(0.37)


def test_sigma_at_decays_monotonically():
    """Coarse-to-fine search requires sigma to never increase over time.

    A non-monotone schedule would re-inject coarse noise late in training and
    destroy fine-tuned genomes.
    """
    cfg = small_cfg(sigma_init=0.20, sigma_decay=0.995, sigma_floor=0.02)
    values = [sigma_at(g, cfg) for g in range(0, 2000, 25)]
    diffs = np.diff(values)
    assert np.all(diffs <= 0.0)


def test_sigma_at_floors_for_huge_generation():
    """The floor keeps mutation alive forever: sigma must never decay to ~0.

    Without the floor, late-generation populations would stop moving entirely
    and evolution would silently stall.
    """
    cfg = small_cfg(sigma_init=0.20, sigma_decay=0.995, sigma_floor=0.02)
    assert sigma_at(10_000_000, cfg) == cfg.sigma_floor
    # And the floor is a max(), so it can never be undercut at any generation.
    for g in (0, 1, 100, 5000):
        assert sigma_at(g, cfg) >= cfg.sigma_floor


# ---------------------------------------------------------------------------
# tournament_select
# ---------------------------------------------------------------------------

def test_tournament_select_shape_and_dtype():
    """Callers use the result to fancy-index the genome matrix, so it must be
    a 1-D integer index array of exactly n_picks entries in [0, pop)."""
    rng = np.random.default_rng(0)
    fitness = np.array([0.1, 0.9, 0.5, 0.3])
    picks = tournament_select(fitness, k=3, n_picks=25, rng=rng)
    assert picks.shape == (25,)
    assert np.issubdtype(picks.dtype, np.integer)
    assert picks.min() >= 0 and picks.max() < fitness.shape[0]


def test_tournament_select_huge_k_always_picks_argmax():
    """As k grows, selection becomes greedy: the global best should win every
    tournament. Candidates are drawn WITH replacement, so k == pop does not
    guarantee the argmax is even a candidate — with k = 50 * pop the chance of
    missing it in any pick is ~(1 - 1/pop)^(50*pop) ~ 2e-22, and the seeded RNG
    makes the outcome deterministic."""
    rng = np.random.default_rng(7)
    fitness = np.array([0.2, 1.5, -0.3, 0.9, 0.0, 0.7, 1.1, 0.4])
    pop = fitness.shape[0]
    picks = tournament_select(fitness, k=50 * pop, n_picks=100, rng=rng)
    np.testing.assert_array_equal(picks, np.full(100, int(fitness.argmax())))


def test_tournament_select_beats_random_baseline_on_average():
    """The whole point of selection: winners must be fitter than uniform
    random picks on average, otherwise there is no selection pressure and
    the GA degenerates into a random walk."""
    rng = np.random.default_rng(123)
    fitness = np.random.default_rng(5).random(64)
    picks = tournament_select(fitness, k=3, n_picks=2000, rng=rng)
    baseline = np.random.default_rng(99).integers(0, 64, size=2000)
    assert fitness[picks].mean() > fitness[baseline].mean()
    # k=3 winners should also clearly beat the population mean.
    assert fitness[picks].mean() > fitness.mean() + 0.05


def test_tournament_select_seeded_determinism():
    """Reproducibility contract: same fitness + same seed => same parents.
    Whole-run replays (a core feature of this project) depend on it."""
    fitness = np.random.default_rng(11).random(32)
    a = tournament_select(fitness, k=3, n_picks=50, rng=np.random.default_rng(42))
    b = tournament_select(fitness, k=3, n_picks=50, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# next_generation: structure
# ---------------------------------------------------------------------------

def test_next_generation_preserves_shape_and_dtype():
    """The population matrix is the GA's only state; its (P, G) shape and
    float32 dtype must be invariants or downstream vectorized sim code breaks."""
    genomes = random_population(16, 10, seed=1)
    fitness = np.random.default_rng(2).random(16)
    cfg = small_cfg(elite=2)
    out = next_generation(genomes, fitness, cfg, np.random.default_rng(3), sigma=0.1)
    assert out.shape == genomes.shape
    assert out.dtype == np.float32


def test_next_generation_elite_rows_are_fittest_genomes_bit_exact():
    """Elitism guarantee: the top `elite` genomes survive verbatim (sorted by
    fitness descending), so best-fitness can never regress on a fixed task."""
    genomes = random_population(12, 8, seed=4)
    # Distinct fitness values so the descending order is unambiguous.
    fitness = np.array([3., 11., 7., 1., 9., 5., 2., 10., 6., 8., 0., 4.])
    cfg = small_cfg(elite=3)
    out = next_generation(genomes, fitness, cfg, np.random.default_rng(5), sigma=0.1)
    expected = genomes[np.argsort(-fitness)[:3]]
    np.testing.assert_array_equal(out[:3], expected)


def test_next_generation_deterministic_same_seed_different_across_seeds():
    """(Config, seed) fully determines a run: identical seeds must give
    bit-identical offspring, and different seeds must actually diverge
    (proof the RNG is really driving the operators)."""
    genomes = random_population(16, 12, seed=6)
    fitness = np.random.default_rng(7).random(16)
    cfg = small_cfg(elite=2, mutation_prob=0.5)
    a = next_generation(genomes, fitness, cfg, np.random.default_rng(0), sigma=0.2)
    b = next_generation(genomes, fitness, cfg, np.random.default_rng(0), sigma=0.2)
    c = next_generation(genomes, fitness, cfg, np.random.default_rng(1), sigma=0.2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# next_generation: operator semantics
# ---------------------------------------------------------------------------

def test_no_crossover_no_mutation_children_are_exact_copies():
    """With both variation operators off, breeding is pure cloning: every
    child row must equal some existing genome bit-exact. Any drift here would
    mean the operators leak even when disabled (e.g. adding 0.0-noise that
    changes dtype or rounding)."""
    genomes = random_population(10, 15, seed=8)
    fitness = np.random.default_rng(9).random(10)
    cfg = small_cfg(elite=2, crossover_rate=0.0, mutation_prob=0.0,
                    heavy_tail_prob=0.0)
    out = next_generation(genomes, fitness, cfg, np.random.default_rng(10), sigma=0.3)
    for row in out:
        assert np.any(np.all(row == genomes, axis=1)), \
            "child is not an exact copy of any parent genome"


def test_full_crossover_mixes_genes_from_both_parents_only():
    """Uniform crossover must be pure recombination: with parents made of all
    0s and all 1s, every child gene must be exactly 0 or 1 (no averaging, no
    noise), and at least one child must contain BOTH values — proof genes were
    really taken from two different parents."""
    n_genes = 20
    pop = 32
    # Interleave 0-genomes and 1-genomes and interleave fitness ranks, so
    # tournaments routinely pick one parent of each kind.
    genomes = np.zeros((pop, n_genes), dtype=np.float32)
    genomes[1::2] = 1.0
    fitness = np.arange(pop, dtype=np.float64)
    cfg = small_cfg(elite=0, crossover_rate=1.0, mutation_prob=0.0,
                    heavy_tail_prob=0.0)
    out = next_generation(genomes, fitness, cfg, np.random.default_rng(12), sigma=0.5)
    assert np.all(np.isin(out, [0.0, 1.0]))
    mixed_rows = np.any(out == 0.0, axis=1) & np.any(out == 1.0, axis=1)
    assert mixed_rows.any(), "no child mixed genes from both parent types"


def test_mutation_touches_approximately_mutation_prob_fraction():
    """The per-gene mutation rate is the main search knob; the realized
    fraction of perturbed genes must match cfg.mutation_prob. Starting from
    all-zero genomes, any nonzero gene == a mutated gene (P[N(0,s)=0] is 0)."""
    pop, n_genes = 64, 178  # real genome length G=178
    genomes = np.zeros((pop, n_genes), dtype=np.float32)
    fitness = np.random.default_rng(13).random(pop)
    cfg = small_cfg(elite=0, crossover_rate=0.0, mutation_prob=0.10,
                    heavy_tail_prob=0.0)
    out = next_generation(genomes, fitness, cfg, np.random.default_rng(14), sigma=0.5)
    frac = np.mean(out != 0.0)
    # ~11.4k Bernoulli(0.1) trials: std err ~ 0.003, so +/-0.02 is generous.
    assert frac == pytest.approx(0.10, abs=0.02)


def test_heavy_tail_scales_perturbation_std():
    """The heavy-tail kick must actually be bigger: with heavy_tail_prob=1
    every child mutates at heavy_tail_scale * sigma, so the perturbation std
    should be ~scale times the std of the heavy_tail_prob=0 run. This is the
    'escape local optima' mechanism — if the scale silently didn't apply, the
    knob would do nothing."""
    pop, n_genes = 64, 178
    genomes = np.zeros((pop, n_genes), dtype=np.float32)
    fitness = np.random.default_rng(15).random(pop)
    kwargs = dict(elite=0, crossover_rate=0.0, mutation_prob=1.0,
                  heavy_tail_scale=5.0)
    cfg_plain = small_cfg(heavy_tail_prob=0.0, **kwargs)
    cfg_heavy = small_cfg(heavy_tail_prob=1.0, **kwargs)
    # Same seed => identical RNG streams (identical call shapes), so the two
    # runs differ only by the per-child sigma scaling.
    out_plain = next_generation(genomes, fitness, cfg_plain,
                                np.random.default_rng(16), sigma=0.1)
    out_heavy = next_generation(genomes, fitness, cfg_heavy,
                                np.random.default_rng(16), sigma=0.1)
    ratio = out_heavy.std() / out_plain.std()
    # Statistically the ratio concentrates hard around 5; +/-20% is generous.
    assert ratio == pytest.approx(cfg_heavy.heavy_tail_scale, rel=0.20)
    # Sanity: the plain run's perturbations are at the base sigma scale.
    assert out_plain.std() == pytest.approx(0.1, rel=0.15)


# ---------------------------------------------------------------------------
# self-adaptive sigma (strategy self-adaptation)
# ---------------------------------------------------------------------------

import dataclasses

from racing.config import EvoConfig
from racing.evolution import next_generation_self_adaptive

SA_CFG = dataclasses.replace(EvoConfig(), population=16, elite=3,
                             self_adaptive_sigma=True)


def _sa_setup(seed=0):
    rng = np.random.default_rng(seed)
    genomes = rng.normal(0, 1, (16, 20)).astype(np.float32)
    sigmas = rng.uniform(0.05, 0.3, 16).astype(np.float32)
    fitness = rng.normal(0, 1, 16).astype(np.float32)
    return genomes, sigmas, fitness


def test_sa_elites_keep_genome_and_sigma():
    """Elitism must preserve BOTH the genome and its strategy parameter —
    losing a good sigma is losing part of what selection learned."""
    genomes, sigmas, fitness = _sa_setup()
    new_g, new_s = next_generation_self_adaptive(
        genomes, sigmas, fitness, SA_CFG, np.random.default_rng(1))
    order = np.argsort(-fitness)
    np.testing.assert_array_equal(new_g[:3], genomes[order[:3]])
    np.testing.assert_array_equal(new_s[:3], sigmas[order[:3]])


def test_sa_shapes_clip_and_determinism():
    """Child sigmas must stay inside [sigma_min, sigma_max] (a runaway sigma
    destroys its lineage before selection can react), shapes must be
    preserved, and the operator must be deterministic under a seed."""
    genomes, sigmas, fitness = _sa_setup()
    g1, s1 = next_generation_self_adaptive(
        genomes, sigmas, fitness, SA_CFG, np.random.default_rng(2))
    g2, s2 = next_generation_self_adaptive(
        genomes, sigmas, fitness, SA_CFG, np.random.default_rng(2))
    assert g1.shape == genomes.shape and s1.shape == sigmas.shape
    assert s1.dtype == np.float32
    assert np.all(s1 >= SA_CFG.sigma_min) and np.all(s1 <= SA_CFG.sigma_max)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(s1, s2)


def test_sa_sigma_evolves_lognormally_around_parent():
    """With tau > 0, child sigmas should scatter multiplicatively around the
    inherited value — the mechanism selection exploits to tune step size."""
    genomes, sigmas, fitness = _sa_setup()
    sigmas[:] = 0.1  # every parent identical -> children's spread is pure tau
    cfg = dataclasses.replace(SA_CFG, sigma_tau=0.3, crossover_rate=0.0,
                              mutation_prob=0.0)
    _, new_s = next_generation_self_adaptive(
        genomes, sigmas, fitness, cfg, np.random.default_rng(3))
    kids = new_s[cfg.elite:]
    assert kids.std() > 0.01          # they scatter...
    assert 0.07 < np.median(kids) < 0.14  # ...around the inherited 0.1


# ---------------------------------------------------------------------------
# island model
# ---------------------------------------------------------------------------

from racing.evolution import island_diversity, next_generation_islands

ISL_CFG = dataclasses.replace(EvoConfig(), population=16, elite=4, islands=4,
                              migrants=1)


def test_islands_breed_independently_and_preserve_shape():
    """Each island's elite must come from ITS OWN members (separated gene
    pools are the whole point), and shapes must be preserved."""
    rng = np.random.default_rng(0)
    genomes = rng.normal(0, 1, (16, 10)).astype(np.float32)
    fitness = rng.normal(0, 1, 16).astype(np.float32)
    new = next_generation_islands(genomes, fitness, ISL_CFG,
                                  np.random.default_rng(1), 0.1, migrate=False)
    assert new.shape == genomes.shape
    size = 4
    for i in range(4):
        island_fit = fitness[i * size:(i + 1) * size]
        best = genomes[i * size + int(island_fit.argmax())]
        np.testing.assert_array_equal(new[i * size], best)  # per-island elite


def test_islands_migration_copies_ring_neighbors_best():
    """On migration generations the source island's best genome must appear
    in the NEXT island's tail — gene flow around the ring, nowhere else."""
    rng = np.random.default_rng(2)
    genomes = rng.normal(0, 1, (16, 10)).astype(np.float32)
    fitness = np.arange(16, dtype=np.float32)  # island 3 holds the global best
    new = next_generation_islands(genomes, fitness, ISL_CFG,
                                  np.random.default_rng(3), 0.1, migrate=True)
    size = 4
    for i in range(4):
        src_best = genomes[i * size + 3]  # fitness ascending within island
        dst = (i + 1) % 4
        np.testing.assert_array_equal(new[dst * size + size - 1], src_best)


def test_island_diversity_reports_two_scales():
    """Sanity: with islands seeded at different offsets, across-island
    distance must exceed within-island distance."""
    rng = np.random.default_rng(4)
    blocks = [rng.normal(loc, 0.01, (4, 10)) for loc in (0, 5, 10, 15)]
    genomes = np.vstack(blocks).astype(np.float32)
    w, a = island_diversity(genomes, 4, np.random.default_rng(5))
    assert a > w * 3
