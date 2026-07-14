"""Tests for racing/brain.py — the flat-genome MLP.

The genome layout is the contract between the GA (which only sees a (P, G)
float matrix) and the network (which interprets slices of that matrix as
weights). If unpack/forward/init disagree about the layout, evolution would
silently mutate garbage, so these tests pin the layout, the math, and the
initialization statistics.
"""

from __future__ import annotations

import numpy as np

from racing.brain import BrainSpec, forward, init_population, make_spec, unpack
from racing.config import BrainConfig, SensorConfig


# ---------------------------------------------------------------------------
# genome_size / BrainSpec arithmetic
# ---------------------------------------------------------------------------

def test_genome_size_default_config_is_242():
    """The default (11 rays + speed) -> 16 hidden -> 2 net must be exactly 242
    floats: every saved genome, checkpoint, and population matrix depends on
    this number staying fixed. (It was 178 with the original 7-ray fan —
    old checkpoints still load because they embed their own config.)"""
    spec = make_spec(BrainConfig(), SensorConfig())
    assert spec.n_in == 12  # 11 ray angles + 1 speed input
    assert spec.hidden == 16
    assert spec.n_out == 2
    # 12*16 (w1) + 16 (b1) + 16*2 (w2) + 2 (b2)
    assert spec.genome_size == 242


def test_genome_size_toy_spec_math():
    """genome_size must equal w1 + b1 + w2 + b2 element counts — the slice
    boundaries in unpack() are derived from this same arithmetic."""
    spec = BrainSpec(n_in=3, hidden=4)
    assert spec.genome_size == 3 * 4 + 4 + 4 * 2 + 2  # == 26


# ---------------------------------------------------------------------------
# unpack: layout + zero-copy views
# ---------------------------------------------------------------------------

def test_unpack_views_share_memory_and_land_in_right_slices():
    """unpack() must return *views*, not copies: init_population and any
    layer-wise tooling write through these views expecting the flat genome to
    change. Also pins the flat layout order [w1 | b1 | w2 | b2]."""
    spec = BrainSpec(n_in=3, hidden=4)  # boundaries: a=12, b=16, c=24, G=26
    p = 2
    genomes = np.zeros((p, spec.genome_size), dtype=np.float32)
    w1, b1, w2, b2 = unpack(genomes, spec)

    assert w1.shape == (p, 3, 4)
    assert b1.shape == (p, 4)
    assert w2.shape == (p, 4, 2)
    assert b2.shape == (p, 2)

    # Write single sentinel values through the views; each must appear at the
    # exact flat offset implied by C-order (row-major) reshape.
    w1[0, 1, 2] = 5.0   # flat index 1*4 + 2 = 6
    b1[0, 3] = 7.0      # flat index 12 + 3 = 15
    w2[0, 2, 1] = 9.0   # flat index 16 + 2*2 + 1 = 21
    b2[0, 1] = 11.0     # flat index 24 + 1 = 25
    assert genomes[0, 6] == 5.0
    assert genomes[0, 15] == 7.0
    assert genomes[0, 21] == 9.0
    assert genomes[0, 25] == 11.0
    # Row 1 untouched — views are per-row aligned, no cross-talk.
    np.testing.assert_array_equal(genomes[1], np.zeros(spec.genome_size, np.float32))


def test_unpack_full_round_trip():
    """Filling every view completely must reconstruct the whole flat genome:
    the four slices are contiguous, ordered, and cover all G entries."""
    spec = BrainSpec(n_in=3, hidden=4)
    p = 2
    genomes = np.zeros((p, spec.genome_size), dtype=np.float32)
    w1, b1, w2, b2 = unpack(genomes, spec)

    rng = np.random.default_rng(0)
    w1[:] = rng.normal(size=w1.shape)
    b1[:] = rng.normal(size=b1.shape)
    w2[:] = rng.normal(size=w2.shape)
    b2[:] = rng.normal(size=b2.shape)

    flat = np.concatenate(
        [w1.reshape(p, -1), b1, w2.reshape(p, -1), b2], axis=1
    )
    np.testing.assert_array_equal(genomes, flat)


# ---------------------------------------------------------------------------
# forward: math correctness
# ---------------------------------------------------------------------------

def test_forward_matches_hand_computed_tiny_network():
    """The batched einsum must compute the ordinary tanh(obs @ W1 + b1) chain
    per individual. A hand-checked 2-2-2 net catches transposed weight axes
    or a wrong contraction subscript, which random-data tests would miss."""
    spec = BrainSpec(n_in=2, hidden=2)
    genomes = np.zeros((1, spec.genome_size), dtype=np.float32)
    w1, b1, w2, b2 = unpack(genomes, spec)

    # w1[input, hidden], w2[hidden, output] — chosen asymmetric so a transpose
    # would change the answer.
    W1 = np.array([[0.5, -0.25], [0.1, 0.7]], dtype=np.float32)
    B1 = np.array([0.05, -0.1], dtype=np.float32)
    W2 = np.array([[0.3, -0.6], [0.8, 0.2]], dtype=np.float32)
    B2 = np.array([-0.2, 0.4], dtype=np.float32)
    w1[0] = W1
    b1[0] = B1
    w2[0] = W2
    b2[0] = B2

    obs = np.array([[0.6, -0.3]], dtype=np.float32)
    out = forward(genomes, obs, spec)

    h_expected = np.tanh(obs[0] @ W1 + B1)          # (2,)
    out_expected = np.tanh(h_expected @ W2 + B2)    # (2,)
    np.testing.assert_allclose(out[0], out_expected, rtol=1e-6, atol=1e-7)


def test_forward_output_shape_range_and_float32():
    """Controls must be shape (P, 2) and strictly inside (-1, 1): downstream
    car kinematics assume tanh-bounded steer/throttle with no saturation at
    exactly +/-1. float32 inputs are the production dtype and must work."""
    spec = BrainSpec(n_in=4, hidden=5)
    rng = np.random.default_rng(7)
    p = 32
    genomes = rng.normal(0.0, 0.5, (p, spec.genome_size)).astype(np.float32)
    obs = rng.uniform(-1.0, 1.0, (p, spec.n_in)).astype(np.float32)

    out = forward(genomes, obs, spec)
    assert out.shape == (p, 2)
    assert np.all(np.isfinite(out))
    assert np.all(out > -1.0)
    assert np.all(out < 1.0)


def test_forward_batched_equals_per_individual():
    """Each row of the batched forward must equal running that individual
    alone: the einsum batches over P but must never mix weights or
    observations between population members."""
    spec = BrainSpec(n_in=3, hidden=4)
    rng = np.random.default_rng(123)
    p = 6
    genomes = rng.normal(0.0, 0.8, (p, spec.genome_size)).astype(np.float32)
    obs = rng.uniform(-1.0, 1.0, (p, spec.n_in)).astype(np.float32)

    batched = forward(genomes, obs, spec)
    for i in range(p):
        solo = forward(genomes[i:i + 1], obs[i:i + 1], spec)
        np.testing.assert_allclose(batched[i], solo[0], rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# init_population: determinism + statistics
# ---------------------------------------------------------------------------

def test_init_population_seeded_determinism_and_shape():
    """(Config, seed) must fully determine generation 0 — reproducibility is
    the whole point of the seeded-RNG design. Same seed -> identical genomes,
    different seed -> different genomes."""
    spec = BrainSpec(n_in=3, hidden=4)
    a = init_population(10, spec, np.random.default_rng(42))
    b = init_population(10, spec, np.random.default_rng(42))
    c = init_population(10, spec, np.random.default_rng(43))

    assert a.shape == (10, spec.genome_size)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_init_population_biases_zero_and_weight_scale():
    """Xavier-style init: biases exactly 0, weight std ~ scale/sqrt(fan_in).
    If the std were off by a large factor, tanh units would start saturated
    and generation 0 would lose its behavioral diversity."""
    spec = make_spec(BrainConfig(), SensorConfig())  # n_in=8, hidden=16
    p = 256
    genomes = init_population(p, spec, np.random.default_rng(0), scale=1.0)
    w1, b1, w2, b2 = unpack(genomes, spec)

    np.testing.assert_array_equal(b1, np.zeros_like(b1))
    np.testing.assert_array_equal(b2, np.zeros_like(b2))

    expected_w1 = 1.0 / np.sqrt(spec.n_in)     # fan_in of layer 1
    expected_w2 = 1.0 / np.sqrt(spec.hidden)   # fan_in of layer 2
    std_w1 = float(np.std(w1))
    std_w2 = float(np.std(w2))
    # With 256*8*16 = 32768 (resp. 8192) samples the empirical std is very
    # close to the target; a 3x band is a loose sanity check on the scale.
    assert expected_w1 / 3 < std_w1 < expected_w1 * 3
    assert expected_w2 / 3 < std_w2 < expected_w2 * 3
    # Means centered near zero relative to the std.
    assert abs(float(np.mean(w1))) < expected_w1 / 10
    assert abs(float(np.mean(w2))) < expected_w2 / 10


def test_init_population_scale_parameter():
    """The scale argument multiplies the weight std — this is EvoConfig's
    init_scale knob; it must actually change initialization strength."""
    spec = BrainSpec(n_in=3, hidden=4)
    small = init_population(200, spec, np.random.default_rng(1), scale=0.5)
    large = init_population(200, spec, np.random.default_rng(1), scale=2.0)
    w1_small, _, _, _ = unpack(small, spec)
    w1_large, _, _, _ = unpack(large, spec)
    ratio = float(np.std(w1_large)) / float(np.std(w1_small))
    assert 3.5 < ratio < 4.5  # same seed, 4x the scale
