"""Ray-cast distance sensors for the whole population, via grid marching.

Each car "sees" with R rays fanned out relative to its heading. A ray reports
the distance to the first wall it meets, normalized by that ray's own range
to [0, 1] (1.0 = clear). These R numbers (+ speed) are the network's ENTIRE
view of the world — no coordinates, no map. That poverty is deliberate: a
brain that only ever sees local wall distances has nothing track-specific to
memorize, which is what forces the evolved policy to generalize.

Rays have PER-RAY ranges: long and dense toward the front, short to the
sides. The front matters because of a stopping-distance argument — braking
from top speed to tight-corner speed consumes ~125 px, so corner geometry
must be legible well beyond that or the policy physically cannot react in
time. The tightness of an upcoming corner is encoded in the small-angle rays
(±4–22°), which is why the fan is densest there.

Implementation: instead of intersecting rays with wall segments analytically,
we march S sample points along each ray and look them up in the track's
occupancy grid (True = wall). The first occupied sample is the hit. One
fancy-index gather answers all P*R rays at once, and the cost is independent
of how complicated the track shape is. Distances are quantized to
ray_range / S (~4–8 px) — far below what a tanh controller can resolve.
"""

from __future__ import annotations

import numpy as np

from .config import SensorConfig


def ray_geometry(cfg: SensorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-ray relative angles (R,), sample distances (R, S), and
    ranges (R,). Each ray marches S samples spread over its own range."""
    rel_angles = np.deg2rad(np.asarray(cfg.ray_angles_deg, dtype=np.float32))
    if cfg.ray_lengths is not None:
        lengths = np.asarray(cfg.ray_lengths, dtype=np.float32)
        if lengths.shape != rel_angles.shape:
            raise ValueError(f"ray_lengths has {lengths.shape[0]} entries for "
                             f"{rel_angles.shape[0]} ray angles")
    else:
        lengths = np.full(rel_angles.shape, cfg.ray_length, dtype=np.float32)
    steps = np.arange(1, cfg.n_samples + 1, dtype=np.float32) / cfg.n_samples
    sample_ts = lengths[:, None] * steps[None, :]          # (R, S)
    return rel_angles, sample_ts, lengths


def sense(
    pos: np.ndarray,        # (P, 2) float32
    heading: np.ndarray,    # (P,)   float32
    occ: np.ndarray,        # (G, G) bool occupancy grid, True = wall
    cfg: SensorConfig,
    rel_angles: np.ndarray,  # (R,)   from ray_geometry
    sample_ts: np.ndarray,   # (R, S) from ray_geometry
    lengths: np.ndarray,     # (R,)   from ray_geometry
) -> np.ndarray:
    """Normalized ray distances (P, R) in [0, 1] for every car at once."""
    g = occ.shape[0]
    angles = heading[:, None] + rel_angles[None, :]              # (P, R)
    dx = np.cos(angles)[:, :, None] * sample_ts[None, :, :]      # (P, R, S)
    dy = np.sin(angles)[:, :, None] * sample_ts[None, :, :]
    xs = pos[:, 0, None, None] + dx
    ys = pos[:, 1, None, None] + dy
    # Samples leaving the grid clamp to the border, which is wall — so rays
    # pointing off-world correctly read "wall at the edge".
    xi = np.clip(xs, 0, g - 1).astype(np.int32)
    yi = np.clip(ys, 0, g - 1).astype(np.int32)
    hits = occ[yi, xi]                                           # (P, R, S)
    first = hits.argmax(axis=2)          # index of first wall sample (0 if none)
    any_hit = hits.any(axis=2)
    r_idx = np.arange(rel_angles.shape[0])
    dist = np.where(any_hit, sample_ts[r_idx[None, :], first], lengths[None, :])
    return (dist / lengths[None, :]).astype(np.float32)
