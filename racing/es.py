"""OpenAI-style Evolution Strategies: gradients without backpropagation.

A deliberately different optimizer family from the GA that trained the
champion, sharing only the fitness function:

- The **GA** maintains a *population* of solutions and moves by selecting
  among them (rank-based survival, crossover, mutation).
- **ES** maintains ONE solution `theta` plus a Gaussian search distribution
  around it, and moves theta along an *estimated gradient* of expected
  fitness:  ∇_theta E[f(theta + sigma*eps)] ≈ 1/(n*sigma) Σ f_i eps_i.
  It is gradient ASCENT where the gradient comes from sampling, not from
  backprop — which is why it works on a non-differentiable simulator.

Two classic variance-reduction tricks make the estimate usable:

- **Mirrored (antithetic) sampling**: evaluate theta + sigma*eps AND
  theta - sigma*eps; their fitness *difference* isolates the effect of the
  direction eps and cancels the common baseline.
- **Rank shaping**: replace raw fitness with centered ranks in [-0.5, 0.5].
  Scale-invariant and outlier-robust — the same reason the GA uses
  tournament (rank) selection rather than fitness-proportional roulette.
"""

from __future__ import annotations

import numpy as np


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Map values to their ranks, centered on zero: [-0.5, 0.5]."""
    ranks = np.empty(x.size, dtype=np.float32)
    ranks[np.argsort(x)] = np.arange(x.size, dtype=np.float32)
    return ranks / max(1, x.size - 1) - 0.5


def es_step(theta: np.ndarray, sigma: float, alpha: float, n_pairs: int,
            eval_batch, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """One ES iteration: sample, evaluate, estimate the gradient, step.

    eval_batch: (2*n_pairs, G) genome matrix -> (2*n_pairs,) fitness.
    Returns (new theta, best candidate fitness, mean candidate fitness).
    """
    eps = rng.standard_normal((n_pairs, theta.size)).astype(np.float32)
    candidates = np.vstack([theta + sigma * eps, theta - sigma * eps])
    fitness = np.asarray(eval_batch(candidates), dtype=np.float32)

    shaped = centered_ranks(fitness)
    # Mirrored estimator: the (f+ - f-) difference per direction.
    signal = shaped[:n_pairs] - shaped[n_pairs:]
    grad = (signal[:, None] * eps).sum(axis=0) / (2.0 * n_pairs * sigma)
    return (theta + alpha * grad).astype(np.float32), \
        float(fitness.max()), float(fitness.mean())
