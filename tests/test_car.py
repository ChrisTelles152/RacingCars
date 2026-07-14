"""Tests for racing/car.py step_cars — the vectorized kinematic model.

step_cars is the physics core of the whole simulator: every fitness number
the GA sees is downstream of these few array operations. The update order
matters and is pinned by these tests:

    1. heading += steer * steer_rate * authority(old_speed) * dt
    2. speed   += (throttle * accel - drag * speed) * dt, clipped to [0, v_max]
    3. pos     += new_speed * (cos(new_heading), sin(new_heading)) * dt

All tests use small hand-built arrays (1-5 cars) and the default CarConfig,
so they run in milliseconds and every expected value is derivable by hand.
"""

from __future__ import annotations

import numpy as np

from racing.car import step_cars
from racing.config import CarConfig


CFG = CarConfig()


def _make_state(n: int):
    """Fresh zeroed state arrays for n cars (pos, heading, speed, alive)."""
    pos = np.zeros((n, 2), dtype=np.float32)
    heading = np.zeros(n, dtype=np.float32)
    speed = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=bool)
    return pos, heading, speed, alive


def _controls(steer: float, throttle: float, n: int = 1) -> np.ndarray:
    return np.tile(np.array([[steer, throttle]], dtype=np.float32), (n, 1))


def test_full_throttle_straight_matches_closed_form():
    """Speed must follow the documented discrete recurrence exactly, and the
    car must travel along +x when heading=0 — this pins the physics contract
    the rest of the simulator (and any learned policy) depends on."""
    n_steps = 20  # stays well below v_max (speed_20 ~ 142 px/s < 300)
    pos, heading, speed, alive = _make_state(1)
    controls = _controls(steer=0.0, throttle=1.0)

    # Reference: the same recurrence in float64. Position integrates the
    # *post-update* speed each step, because step_cars moves after accelerating.
    ref_speed, ref_x = 0.0, 0.0
    for _ in range(n_steps):
        ref_speed += (CFG.accel - CFG.drag * ref_speed) * CFG.dt
        ref_x += ref_speed * CFG.dt

    for _ in range(n_steps):
        step_cars(pos, heading, speed, alive, controls, CFG)

    # Closed form of s_{k+1} = s_k * (1 - drag*dt) + accel*dt from s_0 = 0:
    # s_n = (accel/drag) * (1 - (1 - drag*dt)^n).
    closed = (CFG.accel / CFG.drag) * (1.0 - (1.0 - CFG.drag * CFG.dt) ** n_steps)
    np.testing.assert_allclose(speed[0], closed, rtol=1e-5)
    np.testing.assert_allclose(speed[0], ref_speed, rtol=1e-5)
    np.testing.assert_allclose(pos[0, 0], ref_x, rtol=1e-5)
    # heading=0, steer=0: motion must be purely along +x.
    assert heading[0] == 0.0
    np.testing.assert_allclose(pos[0, 1], 0.0, atol=1e-6)
    assert pos[0, 0] > 0.0


def test_speed_clamps_at_v_max_under_sustained_throttle():
    """The unclamped terminal speed accel/drag = 500 exceeds v_max = 300, so
    the clip is what actually caps lap speed — without it, fitness would be
    driven by a physics exploit rather than driving skill."""
    pos, heading, speed, alive = _make_state(1)
    controls = _controls(steer=0.0, throttle=1.0)

    assert CFG.accel / CFG.drag > CFG.v_max  # sanity: the clamp must engage
    for _ in range(200):  # far past the ~55 steps needed to hit v_max
        step_cars(pos, heading, speed, alive, controls, CFG)
        assert speed[0] <= CFG.v_max

    np.testing.assert_allclose(speed[0], CFG.v_max, rtol=1e-6)


def test_braking_never_goes_below_zero():
    """No reverse gear: speed is clipped at 0 so 'drive backwards over the
    start line' can never emerge as a degenerate fitness strategy."""
    pos, heading, speed, alive = _make_state(1)
    speed[:] = 5.0  # low speed; one full-brake step would overshoot negative:
    # 5 + (-250 - 0.5*5)/30 = -3.4 without the clip.
    controls = _controls(steer=0.0, throttle=-1.0)

    for _ in range(10):
        step_cars(pos, heading, speed, alive, controls, CFG)
        assert speed[0] >= 0.0

    assert speed[0] == 0.0
    # A stopped car must stay put (guards against residual drift).
    x_before = pos[0, 0]
    step_cars(pos, heading, speed, alive, controls, CFG)
    assert pos[0, 0] == x_before


def test_positive_steer_increases_heading():
    """Steering sign convention: positive steer -> heading increases (a
    rightward turn on screen in this y-down world). Sensors and rendering
    both assume this convention, so it must never silently flip."""
    pos, heading, speed, alive = _make_state(1)
    controls = _controls(steer=1.0, throttle=0.0)

    for _ in range(5):
        step_cars(pos, heading, speed, alive, controls, CFG)

    assert heading[0] > 0.0

    # And the mirror image: negative steer decreases heading.
    pos, heading, speed, alive = _make_state(1)
    step_cars(pos, heading, speed, alive, _controls(-1.0, 0.0), CFG)
    assert heading[0] < 0.0


def test_steering_authority_scales_with_speed():
    """Authority = floor + (1 - floor) * speed / v_max is the anti-spin rule:
    at rest a car turns at only steer_rate * floor, at v_max at the full
    steer_rate. Without it, spinning in place would be a viable policy."""
    # At speed = 0 with no throttle, speed stays 0, so every step turns by
    # exactly steer_rate * floor * dt.
    pos, heading, speed, alive = _make_state(1)
    controls = _controls(steer=1.0, throttle=0.0)
    step_cars(pos, heading, speed, alive, controls, CFG)
    expected_floor = CFG.steer_rate * CFG.steer_speed_floor * CFG.dt
    np.testing.assert_allclose(heading[0], expected_floor, rtol=1e-6)

    # At speed = v_max the authority is exactly 1 (heading update uses the
    # speed from *before* this step's acceleration), so one step turns by
    # steer_rate * dt.
    pos, heading, speed, alive = _make_state(1)
    speed[:] = CFG.v_max
    step_cars(pos, heading, speed, alive, _controls(1.0, 1.0), CFG)
    expected_full = CFG.steer_rate * CFG.dt
    np.testing.assert_allclose(heading[0], expected_full, rtol=1e-6)

    # The floor is well below full authority — the scaling is real.
    assert expected_floor < 0.5 * expected_full


def test_dead_cars_are_frozen():
    """Dead cars are masked, not removed: pos and heading must not move and
    speed is forced to 0 even under full controls. If a crashed car kept
    drifting, it would keep earning progress and corrupt fitness."""
    pos, heading, speed, alive = _make_state(1)
    pos[0] = (100.0, 200.0)
    heading[0] = 0.7
    speed[0] = 150.0  # was moving when it died
    alive[0] = False
    controls = _controls(steer=1.0, throttle=1.0)

    for _ in range(10):
        step_cars(pos, heading, speed, alive, controls, CFG)

    np.testing.assert_array_equal(pos[0], np.array([100.0, 200.0], dtype=np.float32))
    assert heading[0] == np.float32(0.7)
    assert speed[0] == 0.0  # forced to zero, not left at 150


def test_vectorized_batch_matches_individual_runs():
    """step_cars must be a pure per-car map: a 5-car batch with mixed controls
    and alive flags has to produce bit-identical results to five 1-car runs.
    Any cross-car leakage (bad broadcasting, wrong axis) breaks the GA's
    assumption that each genome is evaluated independently."""
    rng = np.random.default_rng(42)
    n = 5

    pos0 = rng.uniform(-50.0, 50.0, size=(n, 2)).astype(np.float32)
    heading0 = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    speed0 = rng.uniform(0.0, CFG.v_max, size=n).astype(np.float32)
    alive0 = np.array([True, True, False, True, True])
    controls = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)

    # Batch run.
    pos_b, heading_b, speed_b = pos0.copy(), heading0.copy(), speed0.copy()
    for _ in range(15):
        step_cars(pos_b, heading_b, speed_b, alive0, controls, CFG)

    # One car at a time, same 15 steps each.
    for i in range(n):
        pos_i = pos0[i : i + 1].copy()
        heading_i = heading0[i : i + 1].copy()
        speed_i = speed0[i : i + 1].copy()
        alive_i = alive0[i : i + 1].copy()
        controls_i = controls[i : i + 1].copy()
        for _ in range(15):
            step_cars(pos_i, heading_i, speed_i, alive_i, controls_i, CFG)

        # Same float32 ops in both paths -> exact equality, not just close.
        np.testing.assert_array_equal(pos_b[i], pos_i[0])
        np.testing.assert_array_equal(heading_b[i], heading_i[0])
        np.testing.assert_array_equal(speed_b[i], speed_i[0])
