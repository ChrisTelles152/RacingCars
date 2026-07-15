"""The genetic algorithm: how one generation of brains becomes the next.

This is the entire "learning algorithm" of the project, and it fits in a few
dozen lines because it never looks inside the genomes — it just recombines
and perturbs rows of a (P, G) matrix, using fitness to decide whose rows
get copied. The steps, and why each exists:

- **Elitism**: the top `elite` genomes are copied verbatim. The best solution
  found so far can never be lost to an unlucky mutation, so (on a fixed task)
  best-fitness is monotone non-decreasing.
- **Tournament selection (k=3)**: each parent is the best of k randomly drawn
  genomes. Only *rank* matters, not fitness magnitude — no normalization
  needed — and weak genomes still occasionally win, preserving diversity.
  k is the selection-pressure knob: bigger k, greedier evolution.
- **Uniform crossover**: a child takes each gene from parent A or B by coin
  flip. Uniform (rather than single-point) because a flattened weight vector
  has no meaningful gene ordering to exploit. NOTE: crossover of NN weights
  is genuinely contested (two good parents can encode the same behavior with
  permuted hidden neurons, making their blend garbage — the "competing
  conventions" problem). It's a config knob; try crossover_rate=0.0 and
  compare learning curves.
- **Gaussian mutation**: each gene gets +N(0, sigma) with probability
  mutation_prob. This is the primary search operator. Sigma decays over
  generations — coarse exploration early, fine tuning late — and a small
  fraction of children mutate at 5x sigma ("heavy tail") as a cheap escape
  hatch from local optima.
"""

from __future__ import annotations

import numpy as np

from .config import EvoConfig


def sigma_at(generation: int, cfg: EvoConfig) -> float:
    """Mutation strength schedule: exponential decay to a floor."""
    return max(cfg.sigma_floor, cfg.sigma_init * cfg.sigma_decay**generation)


def tournament_select(fitness: np.ndarray, k: int, n_picks: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Indices of n_picks parents, each the fittest of k random candidates."""
    pop = fitness.shape[0]
    cand = rng.integers(0, pop, size=(n_picks, k))
    return cand[np.arange(n_picks), fitness[cand].argmax(axis=1)]


def next_generation_self_adaptive(
        genomes: np.ndarray, sigmas: np.ndarray, fitness: np.ndarray,
        cfg: EvoConfig, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Breed with EVOLVED per-genome mutation strength (strategy self-adaptation).

    Each genome carries its own sigma. A child inherits parent A's sigma,
    first perturbs it log-normally (sigma * exp(tau * N(0,1)), clipped), then
    mutates its genes with that OWN sigma. Selection never scores sigma
    directly — but genomes whose step size fits the local landscape produce
    fitter children, so good sigmas hitchhike. That's meta-learning in a
    dozen lines: watch the population's sigma anneal itself, and re-inflate
    after curriculum promotions when exploration pays again.

    No heavy-tail kicks here: sigma DIVERSITY across the population plays
    that role (some lineages always run hot).
    """
    pop, n_genes = genomes.shape
    order = np.argsort(-fitness)
    elite = genomes[order[:cfg.elite]].copy()
    elite_sigma = sigmas[order[:cfg.elite]].copy()

    n_children = pop - cfg.elite
    pa_idx = tournament_select(fitness, cfg.tournament_k, n_children, rng)
    pb_idx = tournament_select(fitness, cfg.tournament_k, n_children, rng)

    is_crossed = rng.random(n_children) < cfg.crossover_rate
    from_b = rng.random((n_children, n_genes)) < 0.5
    children = np.where(is_crossed[:, None] & from_b,
                        genomes[pb_idx], genomes[pa_idx])

    child_sigma = sigmas[pa_idx] * np.exp(
        cfg.sigma_tau * rng.standard_normal(n_children)).astype(np.float32)
    np.clip(child_sigma, cfg.sigma_min, cfg.sigma_max, out=child_sigma)

    mutate = rng.random((n_children, n_genes)) < cfg.mutation_prob
    noise = rng.standard_normal((n_children, n_genes)).astype(np.float32)
    children = children + mutate * noise * child_sigma[:, None]

    return (np.vstack([elite, children]).astype(np.float32),
            np.concatenate([elite_sigma, child_sigma]).astype(np.float32))


def next_generation(genomes: np.ndarray, fitness: np.ndarray,
                    cfg: EvoConfig, rng: np.random.Generator,
                    sigma: float) -> np.ndarray:
    """Breed a full new (P, G) population from fitness scores. Fully vectorized."""
    pop, n_genes = genomes.shape
    order = np.argsort(-fitness)
    elite = genomes[order[:cfg.elite]].copy()

    n_children = pop - cfg.elite
    pa = genomes[tournament_select(fitness, cfg.tournament_k, n_children, rng)]
    pb = genomes[tournament_select(fitness, cfg.tournament_k, n_children, rng)]

    # Uniform crossover for a random subset of children; the rest clone parent A.
    is_crossed = rng.random(n_children) < cfg.crossover_rate
    from_b = rng.random((n_children, n_genes)) < 0.5
    children = np.where(is_crossed[:, None] & from_b, pb, pa)

    # Per-gene Gaussian mutation, with occasional heavy-tail "kicks".
    per_child_sigma = np.full(n_children, sigma, dtype=np.float32)
    heavy = rng.random(n_children) < cfg.heavy_tail_prob
    per_child_sigma[heavy] *= cfg.heavy_tail_scale
    mutate = rng.random((n_children, n_genes)) < cfg.mutation_prob
    noise = rng.standard_normal((n_children, n_genes)).astype(np.float32)
    children = children + mutate * noise * per_child_sigma[:, None]

    return np.vstack([elite, children]).astype(np.float32)
