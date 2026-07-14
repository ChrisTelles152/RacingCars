"""Procedural race track generation.

A track is a closed loop built in four steps:

1. **Random control points** — N points around a circle at jittered angles and
   radii. Difficulty raises N and the jitter (wigglier tracks) and narrows the
   corridor.
2. **Closed Catmull-Rom spline** — a smooth curve that passes *through* every
   control point (unlike Bezier, which only approaches them). Each segment is
   a cubic polynomial blended from 4 neighboring control points.
3. **Arc-length resampling** — the raw spline samples bunch up in tight
   corners, so we re-space them to M points exactly `length / M` apart. These
   equally spaced points double as progress checkpoints: "how far along the
   track am I" becomes an array index.
4. **Rasterization** — the corridor is stamped into two boolean occupancy
   grids (True = wall). Sensors and collision checks then become plain array
   lookups instead of geometry math, which is what makes the whole population
   cheap to simulate at once.

Tracks are deterministic: the same (seed, difficulty, config) always produces
the same track. Invalid candidates (corners too tight for the corridor width,
sections pinching together, geometry off the grid) are rejected and re-rolled
from a derived seed — "rejection sampling".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackConfig


@dataclass
class Track:
    seed: int
    difficulty: float
    half_width: float
    centerline: np.ndarray   # (M, 2) float32, equally spaced along the loop
    tangents: np.ndarray     # (M, 2) float32, unit vectors along the track
    normals: np.ndarray      # (M, 2) float32, unit vectors across the track
    cum_s: np.ndarray        # (M,) float32, arc length at each checkpoint
    total_length: float
    occ_sensor: np.ndarray   # (G, G) bool, True = wall as the car sees it
    occ_coll: np.ndarray     # (G, G) bool, True = crash if car *center* enters
    start_pos: np.ndarray    # (2,) float32
    start_heading: float

    @property
    def n_checkpoints(self) -> int:
        return self.centerline.shape[0]


def _lerp(easy: float, hard: float, d: float) -> float:
    return easy + (hard - easy) * d


def _knob(easy: float, hard: float, extreme: float, d: float, d_max: float) -> float:
    """Difficulty knob: easy->hard over [0, 1], hard->extreme over (1, d_max].

    The second segment exists for curriculum overshoot — training slightly
    past d=1.0 so that d=1.0 evaluation happens inside the training
    distribution rather than at its edge.
    """
    if d <= 1.0:
        return _lerp(easy, hard, d)
    if d_max <= 1.0:
        return hard
    return _lerp(hard, extreme, min(d - 1.0, d_max - 1.0) / (d_max - 1.0))


def _catmull_rom_closed(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Sample a closed Catmull-Rom spline through `points` ((N, 2) array).

    Segment i runs from points[i] to points[i+1] and is shaped by its two
    outer neighbors as well. The standard basis (tension 0.5):

        C(t) = 0.5 * ( 2*p1 + (-p0 + p2) t + (2p0 - 5p1 + 4p2 - p3) t^2
                       + (-p0 + 3p1 - 3p2 + p3) t^3 ),  t in [0, 1)

    Wrapping the neighbor indices makes the loop closed and C1-continuous.
    Returns (N * samples_per_segment, 2).
    """
    p0 = np.roll(points, 1, axis=0)[:, None, :]
    p1 = points[:, None, :]
    p2 = np.roll(points, -1, axis=0)[:, None, :]
    p3 = np.roll(points, -2, axis=0)[:, None, :]
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[None, :, None]
    dense = 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
    )
    return dense.reshape(-1, 2)


def _resample_arclength(dense: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-space a closed polyline to m points equally spaced by arc length.

    Returns (points (m, 2), cum_s (m,), total_length).
    """
    closed = np.vstack([dense, dense[:1]])  # include the wrap-around segment
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    targets = np.linspace(0.0, total, m, endpoint=False)
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.stack([x, y], axis=1), targets, total


def _tangents_normals(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents by central difference; normals = tangents rotated +90°."""
    diff = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangents = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return tangents, normals


def _min_curvature_radius(tangents: np.ndarray, step: float) -> float:
    """Approximate the tightest corner: radius ≈ ds / dθ between checkpoints."""
    dots = (tangents * np.roll(tangents, -1, axis=0)).sum(axis=1)
    dtheta = np.arccos(np.clip(dots, -1.0, 1.0))
    dtheta = np.maximum(dtheta, 1e-9)
    return float((step / dtheta).min())


def _has_pinch(centerline: np.ndarray, min_dist: float, skip: int) -> bool:
    """True if two non-adjacent sections of track come closer than min_dist.

    "Non-adjacent" = circular index distance > skip. O(M^2) but M is small
    and this runs once per candidate track, not per simulation step.
    """
    m = centerline.shape[0]
    diff = centerline[:, None, :] - centerline[None, :, :]
    d2 = (diff**2).sum(axis=2)
    idx = np.arange(m)
    idx_dist = np.abs(idx[:, None] - idx[None, :])
    idx_dist = np.minimum(idx_dist, m - idx_dist)  # circular distance
    non_adjacent = idx_dist > skip
    return bool((d2[non_adjacent] < min_dist**2).any())


def _disc_offsets(radius: float) -> np.ndarray:
    """Integer (dx, dy) offsets covering a disc — the rasterization brush."""
    r = int(np.ceil(radius))
    dx, dy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    mask = dx**2 + dy**2 <= radius**2
    return np.stack([dx[mask], dy[mask]], axis=1).astype(np.int32)


def _rasterize_corridor(
    centerline: np.ndarray, cum_s: np.ndarray, total: float,
    radius: float, world_size: int,
) -> np.ndarray:
    """Stamp the drivable corridor into a grid. True = wall, False = drivable.

    We walk the loop at 2 px spacing and stamp a disc of the corridor radius
    at each step. Overlapping stamps leave a scallop depth of s^2/(8r) — a
    tiny fraction of a pixel at 2 px spacing — so the corridor edge is smooth.
    """
    grid = np.ones((world_size, world_size), dtype=bool)

    # Dense, evenly spaced points along the closed loop (2 px apart).
    closed_pts = np.vstack([centerline, centerline[:1]])
    closed_s = np.concatenate([cum_s, [total]])
    targets = np.arange(0.0, total, 2.0)
    px = np.interp(targets, closed_s, closed_pts[:, 0])
    py = np.interp(targets, closed_s, closed_pts[:, 1])
    pts = np.stack([px, py], axis=1)

    offsets = _disc_offsets(radius)  # (K, 2)
    pts_int = np.round(pts).astype(np.int32)
    # Chunked so the (points x brush) index array stays modest in memory.
    for start in range(0, len(pts_int), 256):
        chunk = pts_int[start:start + 256]
        cells = chunk[:, None, :] + offsets[None, :, :]  # (chunk, K, 2)
        xs = np.clip(cells[..., 0], 0, world_size - 1)
        ys = np.clip(cells[..., 1], 0, world_size - 1)
        grid[ys, xs] = False
    return grid


def make_track(seed: int, difficulty: float, cfg: TrackConfig, car_radius: float) -> Track:
    """Generate a valid track for (seed, difficulty). Deterministic.

    Rejection sampling: invalid candidates re-roll from a derived seed. If a
    difficulty produces no valid track after cfg.max_attempts, we back off
    difficulty by 0.1 (with a fresh seed offset per level, so retries are not
    identical) rather than crash a training run — and the returned
    Track.difficulty reports the difficulty actually used, so callers can log
    the truth. If even difficulty 0 fails, the config is unsatisfiable and we
    raise instead of retrying forever.
    """
    requested = float(np.clip(difficulty, 0.0, cfg.max_difficulty))
    for backoff in range(12):
        d = max(0.0, requested - 0.1 * backoff)
        track = _generate(seed + 999_983 * backoff, d, cfg, car_radius)
        if track is not None:
            track.seed = seed
            return track
        print(f"[track] seed={seed} d={d:.2f}: no valid track in "
              f"{cfg.max_attempts} attempts, backing off difficulty")
        if d == 0.0:
            break
    raise RuntimeError(
        f"track generation failed for seed {seed} even at difficulty 0 — "
        f"the TrackConfig constraints are unsatisfiable (base_radius, "
        f"half_width_easy, min_radius_margin, world_size)")


def _generate(rng_seed: int, difficulty: float, cfg: TrackConfig,
              car_radius: float) -> Track | None:
    """One rejection-sampling pass at a fixed difficulty; None if all fail."""
    half_width = _knob(cfg.half_width_easy, cfg.half_width_hard,
                       cfg.half_width_extreme, difficulty, cfg.max_difficulty)
    n_control = round(_knob(cfg.control_points_easy, cfg.control_points_hard,
                            cfg.control_points_extreme, difficulty, cfg.max_difficulty))
    jitter = _knob(cfg.radial_jitter_easy, cfg.radial_jitter_hard,
                   cfg.radial_jitter_extreme, difficulty, cfg.max_difficulty)
    center = cfg.world_size / 2.0
    margin = half_width + 2.0

    for attempt in range(cfg.max_attempts):
        rng = np.random.default_rng(rng_seed + 100_003 * attempt)

        # 1. Random control points around a circle.
        spacing = 2.0 * np.pi / n_control
        angles = (np.arange(n_control) * spacing
                  + rng.uniform(-cfg.angle_jitter, cfg.angle_jitter, n_control) * spacing)
        radii = cfg.base_radius * (1.0 + rng.uniform(-jitter, jitter, n_control))
        control = center + np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

        # 2-3. Smooth closed spline, re-spaced to equal arc-length checkpoints.
        dense = _catmull_rom_closed(control, cfg.samples_per_segment)
        centerline, cum_s, total = _resample_arclength(dense, cfg.n_checkpoints)
        tangents, normals = _tangents_normals(centerline)

        # 4. Validity checks (reject and re-roll on failure).
        if centerline.min() < margin or centerline.max() > cfg.world_size - margin:
            continue  # walls would leave the grid
        step = total / cfg.n_checkpoints
        if _min_curvature_radius(tangents, step) < cfg.min_radius_margin * half_width:
            continue  # a corner is too tight for the corridor width
        if _has_pinch(centerline, cfg.pinch_margin * half_width, cfg.pinch_skip):
            continue  # two sections of track overlap/pinch

        occ_sensor = _rasterize_corridor(centerline, cum_s, total, half_width, cfg.world_size)
        # Collision grid is inflated by the car radius ("configuration space"):
        # testing the car's center against it equals testing the car's disc
        # against the true walls — one array lookup instead of corner math.
        occ_coll = _rasterize_corridor(centerline, cum_s, total,
                                       half_width - car_radius, cfg.world_size)

        return Track(
            seed=rng_seed, difficulty=difficulty, half_width=float(half_width),
            centerline=centerline.astype(np.float32),
            tangents=tangents.astype(np.float32),
            normals=normals.astype(np.float32),
            cum_s=cum_s.astype(np.float32), total_length=total,
            occ_sensor=occ_sensor, occ_coll=occ_coll,
            start_pos=centerline[0].astype(np.float32),
            start_heading=float(np.arctan2(tangents[0, 1], tangents[0, 0])),
        )

    return None  # every candidate rejected; make_track handles the backoff


def project_progress(
    track: Track, pos: np.ndarray, cl_idx: np.ndarray,
    window_back: int, window_fwd: int,
) -> np.ndarray:
    """Nearest centerline checkpoint for every car — searched in a window.

    Each car remembers its last checkpoint index and we only search
    [idx - window_back, idx + window_fwd] around it, as a (P, W) argmin.
    Besides being cheap, the window is an anti-cheat device: progress
    physically cannot jump across a nearby fold of the track, because the
    faraway checkpoints are simply not candidates.

    Requires v_max * dt < window_fwd * checkpoint spacing (asserted in tests).
    Returns the new (P,) int32 checkpoint indices.
    """
    m = track.centerline.shape[0]
    offsets = np.arange(-window_back, window_fwd + 1)
    cand = (cl_idx[:, None] + offsets[None, :]) % m          # (P, W)
    diff = track.centerline[cand] - pos[:, None, :]          # (P, W, 2)
    d2 = (diff**2).sum(axis=2)
    best = d2.argmin(axis=1)                                 # (P,)
    return cand[np.arange(pos.shape[0]), best].astype(np.int32)
