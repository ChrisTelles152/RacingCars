"""Tests for racing/sensors.py — grid-marching ray sensors.

The sensors are the network's ONLY view of the world, so their geometry must
be exactly right: a systematic bias here (wrong angle sign, off-by-one in the
grid lookup, bad normalization) would silently corrupt every observation the
evolved brains ever see. All grids here are synthetic and hand-built so every
expected distance is known in closed form.

Distance readings are quantized: samples march in steps of
ray_length / n_samples along each ray, and sample coordinates truncate to
integer grid cells, adding up to ~1 px more. Tests therefore allow a
tolerance of (ray_length / n_samples + 1) px unless exactness is guaranteed.
"""

from __future__ import annotations

import numpy as np

from racing.config import SensorConfig
from racing.sensors import ray_geometry, sense

G = 512  # synthetic grid size used throughout (same order as real tracks)


def _tol_px(cfg: SensorConfig) -> float:
    """Marching quantization (one sample step) + 1 px int-truncation slack."""
    return cfg.ray_length / cfg.n_samples + 1.0


def _sense_one(x: float, y: float, heading: float, occ: np.ndarray,
               cfg: SensorConfig) -> np.ndarray:
    """Convenience: sense a single car, return its (R,) reading."""
    rel_angles, sample_ts = ray_geometry(cfg)
    pos = np.array([[x, y]], dtype=np.float32)
    hdg = np.array([heading], dtype=np.float32)
    return sense(pos, hdg, occ, cfg, rel_angles, sample_ts)[0]


# ---------------------------------------------------------------------------
# ray_geometry
# ---------------------------------------------------------------------------

def test_ray_geometry_shapes_and_sample_spacing():
    """Downstream code (brain input width, sense broadcasting) relies on the
    exact shapes (R,) and (S,), on samples increasing monotonically (else
    'first hit' would not mean 'nearest wall'), and on the last sample landing
    exactly at ray_length (else max range would be silently shorter)."""
    cfg = SensorConfig()
    rel_angles, sample_ts = ray_geometry(cfg)

    assert rel_angles.shape == (len(cfg.ray_angles_deg),)
    assert sample_ts.shape == (cfg.n_samples,)
    assert rel_angles.dtype == np.float32
    assert sample_ts.dtype == np.float32

    # Angles are just deg->rad of the configured fan.
    np.testing.assert_allclose(
        rel_angles, np.deg2rad(cfg.ray_angles_deg), rtol=1e-6)

    # Strictly increasing march, uniform spacing, ending exactly at max range.
    assert np.all(np.diff(sample_ts) > 0)
    np.testing.assert_allclose(np.diff(sample_ts),
                               cfg.ray_length / cfg.n_samples, rtol=1e-5)
    assert sample_ts[-1] == np.float32(cfg.ray_length)
    # First sample is one step out, NOT 0 — the car's own cell never counts
    # as a wall hit even when the car hugs a wall.
    np.testing.assert_allclose(sample_ts[0], cfg.ray_length / cfg.n_samples,
                               rtol=1e-5)


# ---------------------------------------------------------------------------
# basic hit / no-hit geometry
# ---------------------------------------------------------------------------

def test_clear_rays_read_exactly_one():
    """1.0 must mean *exactly* 'nothing in range': the brain treats saturated
    rays as open road, and an almost-1.0 value would leak a phantom wall."""
    cfg = SensorConfig()
    occ = np.zeros((G, G), dtype=bool)  # empty world, no walls at all
    out = _sense_one(G / 2, G / 2, 0.7, occ, cfg)
    np.testing.assert_array_equal(out, np.ones(len(cfg.ray_angles_deg),
                                               dtype=np.float32))


def test_forward_ray_reads_wall_at_known_distance():
    """The 0-degree ray is the car's forward eye; its reading must equal the
    true distance to the wall (normalized), up to marching quantization."""
    cfg = SensorConfig()
    occ = np.zeros((G, G), dtype=bool)
    x_wall = 320
    # A THICK slab, not a 1-px column: angled rays advance several px in x
    # per sample and can step clean over a wall thinner than that stride.
    # Real track walls are solid occupancy regions, so thickness is realistic.
    occ[:, x_wall:x_wall + 24] = True

    x_car, y_car = 200.0, 256.0
    d_true = x_wall - x_car        # 120 px to the slab face, inside 160 range
    out = _sense_one(x_car, y_car, 0.0, occ, cfg)  # heading +x

    i0 = cfg.ray_angles_deg.index(0.0)
    tol = _tol_px(cfg) / cfg.ray_length
    assert abs(out[i0] - d_true / cfg.ray_length) <= tol

    # Geometry cross-checks on the same scene:
    # +-20 deg rays hit the same column at d / cos(20 deg) ~ 127.7 px.
    d20 = d_true / np.cos(np.deg2rad(20.0))
    for a in (-20.0, 20.0):
        i = cfg.ray_angles_deg.index(a)
        assert abs(out[i] - d20 / cfg.ray_length) <= tol + 1.0 / cfg.ray_length
    # +-45 deg rays would need d / cos(45 deg) ~ 169.7 px > range: clear.
    # +-90 deg rays run parallel to the wall column: clear.
    for a in (-45.0, 45.0, -90.0, 90.0):
        i = cfg.ray_angles_deg.index(a)
        assert out[i] == np.float32(1.0)


def test_side_rays_measure_corridor_half_width():
    """Corridor centering is the core evolved behavior: the +-90 rays must
    report the true lateral clearance or the 'stay centered' gradient the
    brains exploit would point the wrong way."""
    cfg = SensorConfig()
    occ = np.ones((G, G), dtype=bool)      # all wall...
    y0, y1 = 100, 160                       # ...with a free horizontal band
    occ[y0:y1 + 1, :] = False
    w = y1 - y0 + 1                         # corridor width, 61 px
    y_mid = (y0 + y1) / 2.0                 # 130.0

    out = _sense_one(256.0, y_mid, 0.0, occ, cfg)  # facing along the corridor

    tol = _tol_px(cfg) / cfg.ray_length
    expected = (w / 2.0) / cfg.ray_length
    for a in (-90.0, 90.0):
        i = cfg.ray_angles_deg.index(a)
        assert abs(out[i] - expected) <= tol
    # Symmetric corridor -> symmetric readings.
    assert (out[cfg.ray_angles_deg.index(-90.0)]
            == out[cfg.ray_angles_deg.index(90.0)])
    # Straight ahead the corridor runs past sensing range: clear.
    assert out[cfg.ray_angles_deg.index(0.0)] == np.float32(1.0)


def test_heading_rotation_gives_same_corridor_readings():
    """Sensors are defined relative to the heading, so a vertical corridor
    seen by a car heading +y must look identical to a horizontal corridor
    seen by a car heading +x — any asymmetry means a broken angle convention."""
    cfg = SensorConfig()
    y0, y1 = 100, 160

    occ_h = np.ones((G, G), dtype=bool)
    occ_h[y0:y1 + 1, :] = False
    out_h = _sense_one(256.0, (y0 + y1) / 2.0, 0.0, occ_h, cfg)

    occ_v = np.ones((G, G), dtype=bool)
    occ_v[:, y0:y1 + 1] = False            # same band, carved along x instead
    out_v = _sense_one((y0 + y1) / 2.0, 256.0, np.pi / 2.0, occ_v, cfg)

    # float32 trig of pi/2 vs 0 differs by ~1e-7 rad; over 160 px that moves
    # samples by ~2e-5 px, far below one grid cell, so cells (and therefore
    # quantized readings) must match exactly.
    np.testing.assert_array_equal(out_h, out_v)


# ---------------------------------------------------------------------------
# grid-edge behavior
# ---------------------------------------------------------------------------

def test_ray_leaving_grid_hits_border_wall():
    """Samples past the grid edge clamp to the border cells; when those are
    wall (as real tracks guarantee), a ray pointing off-world must read the
    border as a wall, never as open road — otherwise cars would happily
    drive off the map."""
    cfg = SensorConfig()
    occ = np.zeros((G, G), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True  # border = wall

    x_car = 508.0                          # 3 px from the wall column at 511
    d_true = (G - 1) - x_car
    out = _sense_one(x_car, 256.0, 0.0, occ, cfg)  # facing off-grid (+x)

    i0 = cfg.ray_angles_deg.index(0.0)
    tol = _tol_px(cfg) / cfg.ray_length
    assert out[i0] < 1.0                   # definitely NOT clear
    assert abs(out[i0] - d_true / cfg.ray_length) <= tol


def test_small_grid_all_rays_clamped_to_border():
    """With ray_length (160 px) larger than the whole grid (64 px), most
    samples land off-world. Every ray must still report the border wall at
    its true geometric distance — off-grid samples clamp to wall, they do
    not wrap around or read clear."""
    cfg = SensorConfig()
    g = 64
    occ = np.zeros((g, g), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True

    c = g / 2.0                            # dead center: 31 px to every wall
    out = _sense_one(c, c, 0.0, occ, cfg)

    assert np.all(out < 1.0)               # nothing reads clear in a 64px box
    tol = _tol_px(cfg) / cfg.ray_length
    d_true = (g - 1) - c                   # 31 px straight ahead
    i0 = cfg.ray_angles_deg.index(0.0)
    assert abs(out[i0] - d_true / cfg.ray_length) <= tol
    # +-90 rays see the row walls at the same 31 px.
    for a in (-90.0, 90.0):
        i = cfg.ray_angles_deg.index(a)
        assert abs(out[i] - d_true / cfg.ray_length) <= tol


# ---------------------------------------------------------------------------
# vectorization and output contract
# ---------------------------------------------------------------------------

def test_vectorized_sense_matches_per_car_loop():
    """The whole point of sense() is one fancy-index gather for all P cars;
    if batching changed any reading versus sensing cars one at a time, the
    population's fitness would depend on batch composition — a subtle,
    determinism-breaking bug."""
    cfg = SensorConfig()
    rng = np.random.default_rng(1234)

    occ = np.zeros((G, G), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    for _ in range(40):                    # scatter random 16x16 wall blobs
        bx, by = rng.integers(0, G - 16, size=2)
        occ[by:by + 16, bx:bx + 16] = True

    p = 6
    pos = rng.uniform(40.0, G - 40.0, size=(p, 2)).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, size=p).astype(np.float32)

    rel_angles, sample_ts = ray_geometry(cfg)
    batched = sense(pos, heading, occ, cfg, rel_angles, sample_ts)
    assert batched.shape == (p, len(cfg.ray_angles_deg))

    for i in range(p):
        solo = sense(pos[i:i + 1], heading[i:i + 1], occ, cfg,
                     rel_angles, sample_ts)
        np.testing.assert_array_equal(batched[i], solo[0])


def test_output_dtype_and_range():
    """build_obs feeds these values straight into a float32 MLP; a dtype
    upcast would silently double memory/compute, and any value outside
    [0, 1] would break the normalization contract the network trains on."""
    cfg = SensorConfig()
    rng = np.random.default_rng(7)

    occ = rng.random((G, G)) < 0.3         # 30% random wall speckle
    pos = rng.uniform(10.0, G - 10.0, size=(8, 2)).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, size=8).astype(np.float32)

    rel_angles, sample_ts = ray_geometry(cfg)
    out = sense(pos, heading, occ, cfg, rel_angles, sample_ts)

    assert out.dtype == np.float32
    assert out.shape == (8, len(cfg.ray_angles_deg))
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    # Readings are quantized to the sample grid: every non-clear value must
    # be one of the normalized sample distances.
    allowed = np.concatenate([sample_ts / cfg.ray_length,
                              [np.float32(1.0)]]).astype(np.float32)
    assert np.all(np.isin(out, allowed))
