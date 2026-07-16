"""Tests for racing/sensors.py — grid-marching ray sensors.

The sensors are the network's ONLY view of the world, so their geometry must
be exactly right: a systematic bias here (wrong angle sign, off-by-one in the
grid lookup, bad normalization) would silently corrupt every observation the
evolved brains ever see. All grids here are synthetic and hand-built so every
expected distance is known in closed form.

Most tests use UNIFORM — a 7-ray fan with one shared range (ray_lengths=None,
the fallback older checkpoints rely on) — because closed-form geometry is
easiest to state there. Per-ray-length behavior gets its own tests at the end.

Distance readings are quantized: samples march in steps of range / n_samples
along each ray, and sample coordinates truncate to integer grid cells, adding
up to ~1 px more. Tests therefore allow a tolerance of
(range / n_samples + 1) px unless exactness is guaranteed.
"""

from __future__ import annotations

import numpy as np
import pytest

from racing.config import SensorConfig
from racing.sensors import ray_geometry, sense

G = 512  # synthetic grid size used throughout (same order as real tracks)

# One shared range for every ray — the closed-form-friendly configuration.
UNIFORM = SensorConfig(
    ray_angles_deg=(-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0),
    ray_lengths=None, ray_length=160.0, n_samples=24)


def _tol_px(cfg: SensorConfig) -> float:
    """Marching quantization (one sample step) + 1 px int-truncation slack."""
    return cfg.ray_length / cfg.n_samples + 1.0


def _sense_one(x: float, y: float, heading: float, occ: np.ndarray,
               cfg: SensorConfig) -> np.ndarray:
    """Convenience: sense a single car, return its (R,) reading."""
    rel_angles, sample_ts, lengths = ray_geometry(cfg)
    pos = np.array([[x, y]], dtype=np.float32)
    hdg = np.array([heading], dtype=np.float32)
    return sense(pos, hdg, occ, cfg, rel_angles, sample_ts, lengths)[0]


# ---------------------------------------------------------------------------
# ray_geometry
# ---------------------------------------------------------------------------

def test_ray_geometry_shapes_and_sample_spacing():
    """Downstream code (brain input width, sense broadcasting) relies on the
    exact shapes (R,), (R, S), (R,); on samples increasing monotonically
    (else 'first hit' would not mean 'nearest wall'); and on each ray's last
    sample landing exactly at its own range."""
    cfg = UNIFORM
    rel_angles, sample_ts, lengths = ray_geometry(cfg)
    r = len(cfg.ray_angles_deg)

    assert rel_angles.shape == (r,)
    assert sample_ts.shape == (r, cfg.n_samples)
    assert lengths.shape == (r,)
    assert rel_angles.dtype == np.float32
    assert sample_ts.dtype == np.float32

    np.testing.assert_allclose(
        rel_angles, np.deg2rad(cfg.ray_angles_deg), rtol=1e-6)
    np.testing.assert_array_equal(lengths, np.full(r, cfg.ray_length, np.float32))

    # Strictly increasing march per ray, uniform spacing, ending exactly at
    # that ray's range; first sample one step out, NOT 0 — the car's own cell
    # never counts as a wall hit even when the car hugs a wall.
    assert np.all(np.diff(sample_ts, axis=1) > 0)
    for i in range(r):
        np.testing.assert_allclose(np.diff(sample_ts[i]),
                                   lengths[i] / cfg.n_samples, rtol=1e-5)
        np.testing.assert_allclose(sample_ts[i, -1], lengths[i], rtol=1e-6)
        np.testing.assert_allclose(sample_ts[i, 0], lengths[i] / cfg.n_samples,
                                   rtol=1e-5)


def test_ray_geometry_default_config_is_forward_heavy():
    """The default fan implements the perception-horizon design: long dense
    rays ahead (where braking decisions live), short rays to the sides."""
    cfg = SensorConfig()
    rel_angles, sample_ts, lengths = ray_geometry(cfg)
    assert len(cfg.ray_angles_deg) == len(cfg.ray_lengths)
    forward = lengths[np.abs(np.rad2deg(rel_angles)) <= 10.0]
    sideways = lengths[np.abs(np.rad2deg(rel_angles)) >= 90.0]
    assert forward.min() > sideways.max()


def test_ray_geometry_rejects_mismatched_lengths():
    """A ray_lengths tuple that doesn't match the angle fan is a config bug;
    it must fail loudly, not broadcast into silent nonsense."""
    cfg = SensorConfig(ray_angles_deg=(0.0, 45.0), ray_lengths=(100.0,))
    with pytest.raises(ValueError):
        ray_geometry(cfg)


# ---------------------------------------------------------------------------
# basic hit / no-hit geometry
# ---------------------------------------------------------------------------

def test_clear_rays_read_exactly_one():
    """1.0 must mean *exactly* 'nothing in range': the brain treats saturated
    rays as open road, and an almost-1.0 value would leak a phantom wall."""
    cfg = UNIFORM
    occ = np.zeros((G, G), dtype=bool)  # empty world, no walls at all
    out = _sense_one(G / 2, G / 2, 0.7, occ, cfg)
    np.testing.assert_array_equal(out, np.ones(len(cfg.ray_angles_deg),
                                               dtype=np.float32))


def test_forward_ray_reads_wall_at_known_distance():
    """The 0-degree ray is the car's forward eye; its reading must equal the
    true distance to the wall (normalized), up to marching quantization."""
    cfg = UNIFORM
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
    cfg = UNIFORM
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
    cfg = UNIFORM
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
# per-ray lengths
# ---------------------------------------------------------------------------

def test_per_ray_lengths_normalize_by_own_range():
    """Two rays at the same angle-to-wall but different ranges must report
    the same PHYSICAL distance as different normalized values (d / own range)
    — normalization by the wrong ray's range would warp the observation."""
    cfg = SensorConfig(ray_angles_deg=(0.0, 0.0), ray_lengths=(100.0, 200.0),
                       n_samples=50)
    occ = np.zeros((G, G), dtype=bool)
    occ[:, 300:330] = True                 # slab 80 px ahead of the car
    out = _sense_one(220.0, 256.0, 0.0, occ, cfg)
    tol0 = (100.0 / cfg.n_samples + 1.0) / 100.0
    tol1 = (200.0 / cfg.n_samples + 1.0) / 200.0
    assert abs(out[0] - 80.0 / 100.0) <= tol0
    assert abs(out[1] - 80.0 / 200.0) <= tol1


def test_short_ray_clear_while_long_ray_sees():
    """A wall beyond a short ray's range but inside a long ray's range must
    read clear (1.0) on the short ray and as a hit on the long one — this is
    exactly how the forward rays out-see the side rays."""
    cfg = SensorConfig(ray_angles_deg=(0.0, 0.0), ray_lengths=(100.0, 300.0),
                       n_samples=50)
    occ = np.zeros((G, G), dtype=bool)
    occ[:, 400:430] = True                 # slab 180 px ahead
    out = _sense_one(220.0, 256.0, 0.0, occ, cfg)
    assert out[0] == np.float32(1.0)
    assert out[1] < 1.0


# ---------------------------------------------------------------------------
# grid-edge behavior
# ---------------------------------------------------------------------------

def test_ray_leaving_grid_hits_border_wall():
    """Samples past the grid edge clamp to the border cells; when those are
    wall (as real tracks guarantee), a ray pointing off-world must read the
    border as a wall, never as open road — otherwise cars would happily
    drive off the map."""
    cfg = UNIFORM
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
    """With ray range (160 px) larger than the whole grid (64 px), most
    samples land off-world. Every ray must still report the border wall at
    its true geometric distance — off-grid samples clamp to wall, they do
    not wrap around or read clear."""
    cfg = UNIFORM
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
    cfg = SensorConfig()                   # the real 11-ray per-length fan
    rng = np.random.default_rng(1234)

    occ = np.zeros((G, G), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    for _ in range(40):                    # scatter random 16x16 wall blobs
        bx, by = rng.integers(0, G - 16, size=2)
        occ[by:by + 16, bx:bx + 16] = True

    p = 6
    pos = rng.uniform(40.0, G - 40.0, size=(p, 2)).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, size=p).astype(np.float32)

    rel_angles, sample_ts, lengths = ray_geometry(cfg)
    batched = sense(pos, heading, occ, cfg, rel_angles, sample_ts, lengths)
    assert batched.shape == (p, len(cfg.ray_angles_deg))

    for i in range(p):
        solo = sense(pos[i:i + 1], heading[i:i + 1], occ, cfg,
                     rel_angles, sample_ts, lengths)
        np.testing.assert_array_equal(batched[i], solo[0])


def test_output_dtype_and_range():
    """build_obs feeds these values straight into a float32 MLP; a dtype
    upcast would silently double memory/compute, and any value outside
    [0, 1] would break the normalization contract the network trains on."""
    cfg = SensorConfig()                   # the real 11-ray per-length fan
    rng = np.random.default_rng(7)

    occ = rng.random((G, G)) < 0.3         # 30% random wall speckle
    pos = rng.uniform(10.0, G - 10.0, size=(8, 2)).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, size=8).astype(np.float32)

    rel_angles, sample_ts, lengths = ray_geometry(cfg)
    out = sense(pos, heading, occ, cfg, rel_angles, sample_ts, lengths)

    assert out.dtype == np.float32
    assert out.shape == (8, len(cfg.ray_angles_deg))
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    # Readings are quantized to the sample grid: every non-clear value must
    # be one of that ray's normalized sample distances (up to float32
    # rounding of the (length * step) / length round-trip).
    steps = np.arange(1, cfg.n_samples + 1, dtype=np.float32) / cfg.n_samples
    allowed = np.concatenate([steps, [np.float32(1.0)]])
    diffs = np.abs(out.ravel()[:, None] - allowed[None, :]).min(axis=1)
    assert np.all(diffs < 1e-6)

def test_dense_fan_resolves_cone_in_flagship_gap():
    """The obstacle-fix rationale, pinned: a 12px cone at 7deg bearing sits in
    the flagship fan's 4-10deg gap and is INVISIBLE to it, but the dense fan
    (rays at 7deg) always sees it. Angular resolution is the fix, exactly as
    distance resolution ('precision') was for narrow corridors."""
    import dataclasses
    from experiments import apply_variant
    from racing.config import Config
    g = 512
    cx = cy = 256
    b = np.radians(7.0)
    ox, oy = cx + 130 * np.cos(b), cy + 130 * np.sin(b)
    yy, xx = np.mgrid[0:g, 0:g]
    occ = ((xx - ox) ** 2 + (yy - oy) ** 2 <= 36)
    pos = np.array([[cx, cy]], np.float32)
    hd = np.zeros(1, np.float32)

    def hits(variant):
        c = apply_variant(Config(), variant)
        rel, ts, ln = ray_geometry(c.sensor)
        d = sense(pos, hd, occ, c.sensor, rel, ts, ln)[0]
        return int((d < 0.999).sum())

    assert hits("precision") == 0   # the blind spot
    assert hits("densefan") >= 1    # fixed


# ---------------------------------------------------------------------------
# obstacle radar
# ---------------------------------------------------------------------------

def test_radar_reads_nearest_frontal_obstacle():
    """The radar contract: nearest in-range frontal cone as [d/range,
    sin(bearing), cos(bearing)], bearing in the CAR frame. This is the
    alignment-independent signal rays cannot provide."""
    from racing.sensors import radar_nearest
    cfg = SensorConfig(obstacle_radar=True, radar_range=300.0)
    pos = np.array([[100.0, 100.0]], np.float32)
    heading = np.array([0.0], np.float32)  # facing +x
    # two cones: one 50px dead ahead, one 40px behind (must be ignored)
    obstacles = np.array([[150.0, 100.0], [60.0, 100.0]], np.float32)
    out = radar_nearest(pos, heading, obstacles, cfg)
    np.testing.assert_allclose(out[0], [50/300, 0.0, 1.0], atol=1e-6)

    # cone 30 degrees to the left at 100px, car heading +x
    th = np.radians(30.0)
    obstacles = np.array([[100 + 100*np.cos(th), 100 + 100*np.sin(th)]],
                         np.float32)
    out = radar_nearest(pos, heading, obstacles, cfg)
    np.testing.assert_allclose(out[0], [100/300, np.sin(th), np.cos(th)],
                               atol=1e-5)

    # heading rotation moves the bearing: same cone, car now facing +y
    out = radar_nearest(pos, np.array([np.pi/2], np.float32), obstacles, cfg)
    np.testing.assert_allclose(out[0, 1], np.sin(th - np.pi/2), atol=1e-5)


def test_radar_none_in_range_reads_far_and_directionless():
    """[1, 0, 0] is the 'no threat' word: max distance, zero direction —
    constant on every obstacle-free track, so the channels are learnable
    as 'ignore unless they move'."""
    from racing.sensors import radar_nearest
    cfg = SensorConfig(obstacle_radar=True, radar_range=300.0)
    pos = np.array([[100.0, 100.0]], np.float32)
    heading = np.array([0.0], np.float32)
    out_none = radar_nearest(pos, heading, None, cfg)
    out_empty = radar_nearest(pos, heading, np.zeros((0, 2), np.float32), cfg)
    out_far = radar_nearest(pos, heading,
                            np.array([[900.0, 100.0]], np.float32), cfg)
    for out in (out_none, out_empty, out_far):
        np.testing.assert_array_equal(out, [[1.0, 0.0, 0.0]])


def test_radar_grows_spec_and_episode_runs():
    """make_spec adds exactly 3 inputs; a radar config drives a real obstacle
    track end-to-end deterministically."""
    import dataclasses
    from experiments import apply_variant
    from racing.config import Config
    from racing.brain import make_spec, init_population
    from racing.track import make_track
    from racing.simulation import run_episode
    c = apply_variant(Config(), "radar")
    plain = apply_variant(Config(), "precision")
    spec_r = make_spec(c.brain, c.sensor)
    spec_p = make_spec(plain.brain, plain.sensor)
    assert spec_r.n_in == spec_p.n_in + 3
    assert spec_r.genome_size == 290
    track = make_track(29500, 1.0, c.track, c.car.car_radius)
    assert track.obstacle_pos is not None and len(track.obstacle_pos) >= 1
    g = init_population(16, spec_r, np.random.default_rng(2))
    r1 = run_episode(g, track, c)
    r2 = run_episode(g, track, c)
    np.testing.assert_array_equal(r1.fitness, r2.fitness)


def test_radar_respects_line_of_sight():
    """Radar must not see through walls: a nearer cone behind a slab is
    invisible (else the channel reports phantom threats from other track
    folds — measured at ~62% of reports before this mask existed), and the
    farther VISIBLE cone reports instead. Without the grid the check is
    skipped (unit-geometry mode)."""
    from racing.sensors import radar_nearest
    cfg = SensorConfig(obstacle_radar=True, radar_range=300.0)
    pos = np.array([[100.0, 256.0]], np.float32)
    heading = np.array([0.0], np.float32)
    occ = np.zeros((512, 512), bool)
    occ[:, 160:172] = True                     # wall slab at x=160-171
    # cone A: 80px ahead but BEHIND the slab; cone B: 40px ahead, visible
    cones = np.array([[180.0, 256.0], [140.0, 256.0]], np.float32)

    out = radar_nearest(pos, heading, cones, cfg, occ)
    np.testing.assert_allclose(out[0], [40/300, 0.0, 1.0], atol=1e-5)

    # remove the visible cone: the through-wall one must NOT report
    out2 = radar_nearest(pos, heading, cones[:1], cfg, occ)
    np.testing.assert_array_equal(out2, [[1.0, 0.0, 0.0]])

    # no grid passed -> LOS skipped -> nearest wins regardless
    out3 = radar_nearest(pos, heading, cones, cfg)
    np.testing.assert_allclose(out3[0, 0], 40/300, atol=1e-5)

    # a cone right against its own wall stamping must still be visible
    # (sight line sampled short of the cone): cone at the slab's front face
    cone_on_wall = np.array([[158.0, 256.0]], np.float32)
    out4 = radar_nearest(pos, heading, cone_on_wall, cfg, occ)
    assert out4[0, 0] < 1.0, "cone touching a wall must not self-occlude"
