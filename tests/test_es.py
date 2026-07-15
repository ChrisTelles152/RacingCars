"""Tests for racing/es.py — the ES machinery on closed-form objectives.

ES's claim is "gradient ascent without gradients": on a smooth toy objective
where the true optimum is known, a few iterations must move theta toward it.
No simulator involved — these test the OPTIMIZER, not the driving.
"""

from __future__ import annotations

import numpy as np

from racing.es import centered_ranks, es_step


def test_centered_ranks_properties():
    """Rank shaping must be scale/outlier-invariant (only order matters),
    zero-centered, and bounded in [-0.5, 0.5]."""
    x = np.array([3.0, -1.0, 100.0, 0.5])
    r = centered_ranks(x)
    np.testing.assert_allclose(sorted(r), [-0.5, -1/6, 1/6, 0.5], atol=1e-6)
    assert abs(r.sum()) < 1e-6
    # a monstrous outlier changes nothing (vs raw-fitness weighting)
    x2 = np.array([3.0, -1.0, 1e9, 0.5])
    np.testing.assert_array_equal(centered_ranks(x2), r)


def test_es_climbs_a_quadratic():
    """On f(x) = -||x - target||^2 the estimated gradient must carry theta
    toward the target — the core 'ES estimates gradients by sampling' claim."""
    target = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)

    def eval_batch(cand):
        return -np.sum((cand - target) ** 2, axis=1)

    theta = np.zeros(4, dtype=np.float32)
    rng = np.random.default_rng(0)
    d0 = np.linalg.norm(theta - target)
    for _ in range(150):
        theta, _, _ = es_step(theta, sigma=0.1, alpha=0.3, n_pairs=16,
                              eval_batch=eval_batch, rng=rng)
    assert np.linalg.norm(theta - target) < 0.25 * d0


def test_es_step_deterministic_under_seed():
    """Same seed, same step — fine-tuning runs must be reproducible."""
    def eval_batch(cand):
        return -np.sum(cand ** 2, axis=1)
    t1, _, _ = es_step(np.ones(6, np.float32), 0.05, 0.1, 8, eval_batch,
                       np.random.default_rng(4))
    t2, _, _ = es_step(np.ones(6, np.float32), 0.05, 0.1, 8, eval_batch,
                       np.random.default_rng(4))
    np.testing.assert_array_equal(t1, t2)
