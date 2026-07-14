"""Ray-cast distance sensors for the whole population, via grid marching.

Each car "sees" with R rays fanned out relative to its heading. A ray reports
the distance to the first wall it meets, normalized to [0, 1] (1.0 = clear).
These R numbers (+ speed) are the network's ENTIRE view of the world — no
coordinates, no map. That poverty is deliberate: a brain that only ever sees
local wall distances has nothing track-specific to memorize, which is what
forces the evolved policy to generalize to unseen tracks.

Implementation: instead of intersecting rays with wall segments analytically,
we march S sample points along each ray and look them up in the track's
occupancy grid (True = wall). The first occupied sample is the hit. One
fancy-index gather answers all P*R rays at once, and the cost is independent
of how complicated the track shape is. Measured ~10x faster than exact
segment intersection at this scale; the distance is quantized to
ray_length / S (~7 px), far below what a tanh controller can resolve anyway.
"""

from __future__ import annotations

import numpy as np

from .config import SensorConfig


def ray_geometry(cfg: SensorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the per-ray relative angles (R,) and sample distances (S,)."""
    rel_angles = np.deg2rad(np.asarray(cfg.ray_angles_deg, dtype=np.float32))
    sample_ts = np.linspace(cfg.ray_length / cfg.n_samples, cfg.ray_length,
                            cfg.n_samples, dtype=np.float32)
    return rel_angles, sample_ts


def sense(
    pos: np.ndarray,       # (P, 2) float32
    heading: np.ndarray,   # (P,)   float32
    occ: np.ndarray,       # (G, G) bool occupancy grid, True = wall
    cfg: SensorConfig,
    rel_angles: np.ndarray,
    sample_ts: np.ndarray,
) -> np.ndarray:
    """Normalized ray distances (P, R) in [0, 1] for every car at once."""
    g = occ.shape[0]
    angles = heading[:, None] + rel_angles[None, :]              # (P, R)
    dx = np.cos(angles)[:, :, None] * sample_ts                  # (P, R, S)
    dy = np.sin(angles)[:, :, None] * sample_ts
    xs = pos[:, 0, None, None] + dx
    ys = pos[:, 1, None, None] + dy
    # Samples leaving the grid clamp to the border, which is wall — so rays
    # pointing off-world correctly read "wall at the edge".
    xi = np.clip(xs, 0, g - 1).astype(np.int32)
    yi = np.clip(ys, 0, g - 1).astype(np.int32)
    hits = occ[yi, xi]                                           # (P, R, S)
    first = hits.argmax(axis=2)          # index of first wall sample (0 if none)
    any_hit = hits.any(axis=2)
    dist = np.where(any_hit, sample_ts[first], cfg.ray_length)
    return (dist / cfg.ray_length).astype(np.float32)
