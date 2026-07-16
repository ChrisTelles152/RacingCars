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


def radar_nearest(pos: np.ndarray, heading: np.ndarray,
                  obstacle_pos: np.ndarray | None,
                  cfg: SensorConfig,
                  occ: np.ndarray | None = None) -> np.ndarray:
    """(P, 3) egocentric polar readout of the nearest VISIBLE frontal obstacle.

    Channels: [distance / radar_range, sin(bearing), cos(bearing)], bearing
    measured from the car's heading. "Nothing in range" reads [1, 0, 0] —
    max distance, zero direction vector.

    This is the deliberate contrast with ray sensing: a ray only reports an
    obstacle if one happens to align with it (angular aliasing — the failure
    mode that kept cone crashes at 64-97%), while radar reports the nearest
    threat's true bearing regardless of alignment, continuously through the
    whole approach. Two masks keep the signal honest:
    - frontal arc only — what's behind the car cannot be hit by driving on;
    - line of sight against `occ` — without it the radar reports cones on
      OTHER folds of the track through solid walls (measured: ~62% of
      reports were through-wall phantoms), training the exact blanket
      caution the channel exists to avoid. The sight line is sampled
      strictly SHORT of the cone so its own stamped wall cells never count
      as occlusion; another cone sitting on the line does occlude (it is
      the nearer threat and reports instead).
    """
    p = pos.shape[0]
    out = np.zeros((p, 3), dtype=np.float32)
    out[:, 0] = 1.0
    if obstacle_pos is None or len(obstacle_pos) == 0:
        return out
    diff = obstacle_pos[None, :, :] - pos[:, None, :]           # (P, N, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                     # (P, N)
    dist_safe = np.maximum(dist, 1e-6)
    bearing = np.arctan2(diff[..., 1], diff[..., 0]) - heading[:, None]
    bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi         # wrap [-pi, pi]
    valid = (dist < cfg.radar_range) & (np.abs(bearing) < np.radians(100.0))

    if occ is not None:
        g = occ.shape[0]
        n_los = 16
        # Sample the sight line up to ~10 px short of the cone (cone radius
        # + margin), so the cone's own occupancy never self-occludes.
        seg = np.maximum(dist - 10.0, 1.0)                      # (P, N)
        unit = diff / dist_safe[..., None]                      # (P, N, 2)
        fr = np.arange(1, n_los + 1, dtype=np.float32) / (n_los + 1)
        sample = (pos[:, None, None, :]
                  + unit[:, :, None, :] * (seg[..., None] * fr)[..., None])
        xi = np.clip(sample[..., 0], 0, g - 1).astype(np.int32)
        yi = np.clip(sample[..., 1], 0, g - 1).astype(np.int32)
        blocked = occ[yi, xi].any(axis=2)                       # (P, N)
        valid &= ~blocked

    dist_masked = np.where(valid, dist, np.inf)
    nearest = dist_masked.argmin(axis=1)                        # (P,)
    rows = np.arange(p)
    has = np.isfinite(dist_masked[rows, nearest])
    d = (dist[rows, nearest] / cfg.radar_range).astype(np.float32)
    th = bearing[rows, nearest]
    out[has, 0] = d[has]
    out[has, 1] = np.sin(th[has]).astype(np.float32)
    out[has, 2] = np.cos(th[has]).astype(np.float32)
    return out


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
