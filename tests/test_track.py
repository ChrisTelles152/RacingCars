"""Tests for racing/track.py — procedural track generation and progress lookup.

The track generator is the foundation everything else stands on: sensors and
collisions read its occupancy grids, fitness reads its checkpoints, and the
whole training pipeline assumes tracks are deterministic per (seed, difficulty).
These tests pin down those contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from racing.config import CarConfig, SimConfig, TrackConfig
from racing.track import make_track, project_progress

TRACK_CFG = TrackConfig()
CAR_CFG = CarConfig()
SIM_CFG = SimConfig()


@pytest.fixture(scope="module")
def track():
    """One full-size default-config track, shared by the read-only tests."""
    return make_track(seed=3, difficulty=0.4, cfg=TRACK_CFG,
                      car_radius=CAR_CFG.car_radius)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_make_track_is_deterministic():
    """Same (seed, difficulty, config) must yield the identical track.

    Training re-generates tracks from seeds instead of storing them, and
    replay/validation depend on getting bit-identical geometry back.
    """
    a = make_track(seed=3, difficulty=0.4, cfg=TRACK_CFG, car_radius=CAR_CFG.car_radius)
    b = make_track(seed=3, difficulty=0.4, cfg=TRACK_CFG, car_radius=CAR_CFG.car_radius)

    np.testing.assert_array_equal(a.centerline, b.centerline)
    np.testing.assert_array_equal(a.tangents, b.tangents)
    np.testing.assert_array_equal(a.normals, b.normals)
    np.testing.assert_array_equal(a.cum_s, b.cum_s)
    np.testing.assert_array_equal(a.occ_sensor, b.occ_sensor)
    np.testing.assert_array_equal(a.occ_coll, b.occ_coll)
    np.testing.assert_array_equal(a.start_pos, b.start_pos)
    assert a.total_length == b.total_length
    assert a.start_heading == b.start_heading
    assert a.half_width == b.half_width


# ---------------------------------------------------------------------------
# Checkpoint geometry
# ---------------------------------------------------------------------------

def test_checkpoints_evenly_spaced(track):
    """Consecutive checkpoints (wrap included) sit ~total_length/M apart.

    Equal spacing is what lets a checkpoint *index* stand in for arc-length
    progress: fitness = index / M only works if every step is the same size.
    Chord distance is slightly shorter than arc length on curves, hence the
    2% tolerance rather than exact equality.
    """
    cl = track.centerline.astype(np.float64)
    m = track.n_checkpoints
    # np.roll(-1) pairs each point with its successor, including M-1 -> 0.
    dists = np.linalg.norm(np.roll(cl, -1, axis=0) - cl, axis=1)
    spacing = track.total_length / m

    assert dists.shape == (m,)
    np.testing.assert_allclose(dists, spacing, rtol=0.02)


def test_cum_s_monotone_and_bounded(track):
    """cum_s starts at 0, strictly increases, and never reaches total_length.

    cum_s parameterizes the loop as a half-open interval [0, total): the last
    checkpoint must not alias the first, or interpolation along the loop
    (e.g. in rasterization) would double-count the seam.
    """
    s = track.cum_s.astype(np.float64)
    assert s[0] == 0.0
    assert np.all(np.diff(s) > 0.0)
    assert s[-1] < track.total_length


def test_tangents_normals_unit_and_perpendicular(track):
    """Tangents and normals are unit length and mutually perpendicular.

    Observations and steering geometry decompose vectors into along-track /
    across-track components; that only works with an orthonormal frame.
    Tolerances are loose-ish because the arrays are stored as float32.
    """
    t = track.tangents.astype(np.float64)
    n = track.normals.astype(np.float64)

    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose((t * n).sum(axis=1), 0.0, atol=1e-5)


def test_centerline_within_grid_bounds(track):
    """Every centerline point keeps at least half_width from the grid edge.

    If the corridor spilled off the grid, rasterization would clip walls at
    the border and cars could drive out of the world. make_track rejects such
    candidates; verify the accepted track really honors the margin.
    """
    cl = track.centerline
    lo = track.half_width
    hi = TRACK_CFG.world_size - track.half_width
    assert cl.min() >= lo
    assert cl.max() <= hi


# ---------------------------------------------------------------------------
# Robustness across seeds and difficulties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_tracks_build_sanely_across_seeds(difficulty):
    """Across seeds and the full difficulty range, tracks stay well-formed.

    Training throws thousands of (seed, difficulty) pairs at the generator,
    so it must not just avoid crashing but keep producing corridors of sane
    size. Also checks the configuration-space invariant: the collision grid
    (inflated by car radius) must call 'wall' everywhere the sensor grid
    does — otherwise a car could crash inside terrain its sensors report as
    drivable open space.
    """
    for seed in range(5):
        tr = make_track(seed=seed, difficulty=difficulty, cfg=TRACK_CFG,
                        car_radius=CAR_CFG.car_radius)

        drivable_frac = 1.0 - tr.occ_sensor.mean()
        assert 0.02 <= drivable_frac <= 0.40, (
            f"seed={seed} d={difficulty}: drivable fraction {drivable_frac:.3f} "
            "outside sane range")

        # occ_coll wall region is a superset of occ_sensor wall region.
        assert tr.occ_coll[tr.occ_sensor].all(), (
            f"seed={seed} d={difficulty}: sensor wall cell drivable in "
            "collision grid")

        assert tr.centerline.shape == (TRACK_CFG.n_checkpoints, 2)
        assert np.isfinite(tr.centerline).all()
        assert tr.total_length > 0.0


# ---------------------------------------------------------------------------
# Start pose
# ---------------------------------------------------------------------------

def test_start_pose_matches_checkpoint_zero(track):
    """start_pos is checkpoint 0 and start_heading points along tangents[0].

    Cars spawn from these fields with cl_idx = 0; if they disagreed with the
    centerline, progress tracking would start out of sync (or cars would
    spawn facing a wall).
    """
    np.testing.assert_array_equal(track.start_pos, track.centerline[0])

    heading_vec = np.array([np.cos(track.start_heading),
                            np.sin(track.start_heading)])
    np.testing.assert_allclose(heading_vec, track.tangents[0].astype(np.float64),
                               atol=1e-5)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def test_centerline_is_drivable_in_both_grids(track):
    """Sampled centerline cells are open (False) in both occupancy grids.

    The centerline is the middle of the corridor — if any of it rasterized as
    wall, a car following the ideal racing line would 'crash on air'. Grids
    are indexed grid[y, x], matching the module convention.
    """
    pts = track.centerline[::4]
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)

    assert not track.occ_sensor[ys, xs].any()
    assert not track.occ_coll[ys, xs].any()


# ---------------------------------------------------------------------------
# project_progress
# ---------------------------------------------------------------------------

def test_project_progress_exact_checkpoints(track):
    """Cars sitting exactly on checkpoints 10 and 50 map back to 10 and 50.

    The fixed point of progress projection: a car at a checkpoint whose
    cl_idx already points there must stay there (distance 0 wins the argmin).
    """
    pos = track.centerline[[10, 50]].astype(np.float32)
    cl_idx = np.array([10, 50], dtype=np.int32)
    out = project_progress(track, pos, cl_idx,
                           SIM_CFG.progress_window_back,
                           SIM_CFG.progress_window_fwd)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, [10, 50])


def test_project_progress_advances_within_window(track):
    """A car that moved 3 checkpoints ahead of its stale cl_idx is found there.

    This is the normal per-step case: the car advanced a little, and the
    windowed search must pick up the new nearest checkpoint from the stale
    starting index.
    """
    idx = 100
    pos = track.centerline[[idx + 3]].astype(np.float32)
    cl_idx = np.array([idx], dtype=np.int32)
    out = project_progress(track, pos, cl_idx,
                           SIM_CFG.progress_window_back,
                           SIM_CFG.progress_window_fwd)
    np.testing.assert_array_equal(out, [idx + 3])


def test_project_progress_wraps_around_lap(track):
    """A car crossing the start/finish line wraps from index M-2 to 2.

    The candidate window is computed modulo M, so finishing a lap must roll
    the index over to the small numbers instead of getting stuck at M-1 —
    this is what makes multi-lap progress counting work.
    """
    m = track.n_checkpoints
    pos = track.centerline[[2]].astype(np.float32)
    cl_idx = np.array([m - 2], dtype=np.int32)
    out = project_progress(track, pos, cl_idx,
                           SIM_CFG.progress_window_back,
                           SIM_CFG.progress_window_fwd)
    np.testing.assert_array_equal(out, [2])


def test_progress_window_outruns_top_speed(track):
    """One physics step at v_max stays well inside the forward search window.

    project_progress only searches window_fwd checkpoints ahead. If a car
    could out-drive that window in a single step, its progress index would
    silently fall behind reality and fitness would under-count. Documented in
    project_progress's docstring as 'asserted in tests' — this is that test.
    """
    max_step_dist = CAR_CFG.v_max * CAR_CFG.dt
    spacing = track.total_length / track.n_checkpoints
    window_reach = SIM_CFG.progress_window_fwd * spacing
    assert max_step_dist < window_reach


# ---------------------------------------------------------------------------
# Sprint-3 generator features: variable width + hairpin traps
# ---------------------------------------------------------------------------

import dataclasses
import hashlib

from racing.track import make_track as _mk


def _track_hash(t):
    return hashlib.sha256(t.centerline.tobytes() + t.occ_sensor.tobytes()
                          + t.occ_coll.tobytes()).hexdigest()[:16]


def test_golden_hashes_features_off():
    """With width/trap features at their defaults (off), the generator must
    reproduce PRE-FEATURE tracks bit-for-bit: every frozen suite and every
    old checkpoint's world depends on it. These hashes were recorded from
    the generator before the features existed."""
    golden = [("d0269f49f37795d4", 7, 0.0), ("61bdd872507440ca", 7, 0.5),
              ("93af6ac4f617f31c", 13, 1.0), ("6c3dc0d3e6bda412", 20004, 1.0)]
    cfg = TrackConfig()
    for want, seed, d in golden:
        assert _track_hash(_mk(seed, d, cfg, 6.0)) == want, (seed, d)


def test_width_profile_varies_and_respects_bounds():
    """The width profile must actually vary the corridor (else the feature is
    dead), stay within +-amp of nominal, and every section must leave the car
    at least the traversability slack the validity check promises."""
    cfg = dataclasses.replace(TrackConfig(), width_profile_amp=0.3)
    t = _mk(29100, 1.0, cfg, 6.0)
    assert t.half_widths.std() > 0.5           # it varies
    assert t.half_widths.min() >= t.half_width * 0.7 - 1e-6
    assert t.half_widths.max() <= t.half_width * 1.3 + 1e-6
    assert (t.half_widths - 6.0).min() >= 5.0  # traversability floor
    # deterministic
    t2 = _mk(29100, 1.0, cfg, 6.0)
    np.testing.assert_array_equal(t.half_widths, t2.half_widths)
    np.testing.assert_array_equal(t.occ_sensor, t2.occ_sensor)


def test_constant_width_track_has_uniform_half_widths():
    """With the profile off, half_widths must equal the scalar everywhere —
    downstream code may consult either representation."""
    t = _mk(7, 0.5, TrackConfig(), 6.0)
    np.testing.assert_allclose(t.half_widths, t.half_width, rtol=1e-6)


def test_trap_track_changes_geometry_and_is_deterministic():
    """Trap insertion must produce a different (but still valid) track at its
    labeled difficulty, deterministically per seed."""
    cfg = dataclasses.replace(TrackConfig(), trap_prob=1.0)
    plain = _mk(29200, 1.0, TrackConfig(), 6.0)
    trapped = _mk(29200, 1.0, cfg, 6.0)
    assert not np.array_equal(plain.centerline, trapped.centerline)
    assert trapped.difficulty == 1.0  # generated at its label, no backoff
    again = _mk(29200, 1.0, cfg, 6.0)
    np.testing.assert_array_equal(trapped.centerline, again.centerline)


def test_grip_zones_painted_and_bounded():
    """Low-grip zones must actually appear in the surface grid, only inside
    the corridor region, with grip values in [grip_min, 0.85] — and the
    feature-off surface must be None (no cost, no behavior change)."""
    cfg = dataclasses.replace(TrackConfig(), grip_zones=2)
    t = _mk(29300, 1.0, cfg, 6.0)
    assert t.surface is not None
    low = t.surface < 0.999
    assert low.sum() > 500              # zones exist and have area
    vals = t.surface[low]
    assert vals.min() >= cfg.grip_min - 1e-6 and vals.max() <= 0.85 + 1e-6
    assert _mk(29300, 1.0, TrackConfig(), 6.0).surface is None
    # deterministic
    t2 = _mk(29300, 1.0, cfg, 6.0)
    np.testing.assert_array_equal(t.surface, t2.surface)


def test_grip_scales_acceleration_and_steering():
    """On grip g, drive force and steering authority both scale by g — the
    physics contract the 'drive with margins' lesson depends on."""
    from racing.car import step_cars
    from racing.config import CarConfig
    ccfg = CarConfig()
    def run(grip):
        pos = np.zeros((1, 2), np.float32)
        heading = np.zeros(1, np.float32)
        speed = np.zeros(1, np.float32)
        alive = np.ones(1, bool)
        ctrl = np.array([[1.0, 1.0]], np.float32)
        for _ in range(30):
            step_cars(pos, heading, speed, alive, ctrl, ccfg,
                      None if grip is None else np.full(1, grip, np.float32))
        return float(speed[0]), float(heading[0])
    v_full, h_full = run(None)
    v_ice, h_ice = run(0.5)
    assert v_ice < v_full * 0.75
    assert h_ice < h_full * 0.85


def test_obstacles_stamped_inside_corridor():
    """Cones must add wall cells INSIDE the previously drivable corridor
    (that is what makes them obstacles), in both grids, deterministically,
    and never near the spawn."""
    cfg = dataclasses.replace(TrackConfig(), obstacles=3)
    plain = _mk(29400, 1.0, TrackConfig(), 6.0)
    coned = _mk(29400, 1.0, cfg, 6.0)
    added_sensor = coned.occ_sensor & ~plain.occ_sensor
    added_coll = coned.occ_coll & ~plain.occ_coll
    assert added_sensor.sum() > 50          # cones visible to sensors
    assert added_coll.sum() > added_sensor.sum()  # coll cones are inflated
    # nothing new near the spawn point
    sx, sy = int(round(plain.start_pos[0])), int(round(plain.start_pos[1]))
    assert not added_coll[sy - 40:sy + 40, sx - 40:sx + 40].any()
    again = _mk(29400, 1.0, cfg, 6.0)
    np.testing.assert_array_equal(coned.occ_sensor, again.occ_sensor)
