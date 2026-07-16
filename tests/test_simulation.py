"""Tests for racing/simulation.py — the episode loop and its rules of the game.

These tests drive the REAL simulation stack (real track, real physics, real
sensors) with scripted controls where possible, so they pin down the game
rules — monotone progress, stagnation kills, crash freezing, fitness math,
determinism — rather than mocking them away.

One easy track (seed 7, difficulty 0.0) is built once at module import and
shared by every test: track generation is deterministic, and reusing it keeps
the suite fast.
"""

from __future__ import annotations

import numpy as np

from racing.brain import init_population, make_spec
from racing.config import Config
from racing.sensors import ray_geometry
from racing.simulation import EpisodeResult, init_state, run_episode, step
from racing.track import make_track

CONFIG = Config()
TRACK = make_track(seed=7, difficulty=0.0, cfg=CONFIG.track,
                   car_radius=CONFIG.car.car_radius)
SPEC = make_spec(CONFIG.brain, CONFIG.sensor)
RAYS = ray_geometry(CONFIG.sensor)


def _scripted_step(state, steer: float, throttle: float) -> None:
    """Advance one tick with fixed controls for every car (genomes unused)."""
    p = state.pos.shape[0]
    controls = np.tile(np.array([steer, throttle], dtype=np.float32), (p, 1))
    step(state, TRACK, None, SPEC, CONFIG, RAYS, controls=controls)


def test_init_state_everyone_at_start():
    """Fair racing requires identical initial conditions: every car must start
    at the start line, pointing along the track, at rest, alive, with zero
    progress — otherwise fitness comparisons across the population are biased.
    """
    pop = 5
    state = init_state(pop, TRACK)

    np.testing.assert_array_equal(state.pos,
                                  np.tile(TRACK.start_pos, (pop, 1)))
    np.testing.assert_array_equal(state.heading,
                                  np.full(pop, TRACK.start_heading,
                                          dtype=np.float32))
    np.testing.assert_array_equal(state.speed, np.zeros(pop))
    assert state.alive.all()
    assert not state.crashed.any()
    np.testing.assert_array_equal(state.progress, np.zeros(pop))
    np.testing.assert_array_equal(state.cl_idx, np.zeros(pop, dtype=np.int32))
    np.testing.assert_array_equal(state.laps, np.zeros(pop, dtype=np.int32))
    np.testing.assert_array_equal(state.stall_ref, np.zeros(pop))
    np.testing.assert_array_equal(state.steps_alive, np.zeros(pop))
    assert state.t == 0


def test_scripted_full_throttle_gains_monotone_progress():
    """The core reward signal: driving forward along the track must register
    as progress, and progress must be monotone non-decreasing (best point ever
    reached). If progress could decrease, wiggling backwards would be scored,
    reopening reward-hacking exploits the design explicitly closes.
    """
    state = init_state(1, TRACK)
    history = []
    for _ in range(120):
        _scripted_step(state, steer=0.0, throttle=1.0)
        history.append(float(state.progress[0]))
    history = np.asarray(history)

    # Straight full throttle from the start line follows the track for at
    # least a short while, so progress accumulates within a few steps.
    assert history[19] > 10.0, "full throttle straight should gain progress"
    # Monotone non-decreasing over the whole episode, even if the car later
    # drifts off the (curved) track and crashes.
    assert np.all(np.diff(history) >= 0.0), "progress must never decrease"


def test_idle_car_killed_at_first_stall_check():
    """The stagnation rule ends episodes early: a car that gains no progress
    over a trailing window (parked, circling, wall-wiggling) must be frozen at
    exactly the first stall check, or dud genomes would waste max_steps of
    compute every generation.
    """
    n = CONFIG.sim.stall_check_every
    state = init_state(1, TRACK)
    for _ in range(n):
        _scripted_step(state, steer=0.0, throttle=0.0)

    assert state.t == n
    assert not state.alive[0], "idle car should be stall-killed at step 120"
    assert not state.crashed[0], "stall kill is not a crash"
    # Zero throttle means zero speed means the car literally never moved.
    np.testing.assert_array_equal(state.pos[0], TRACK.start_pos)
    assert state.progress[0] == 0.0


def test_crash_kills_and_freezes_position():
    """Crashing must be terminal and freezing: crashed flag set, alive
    cleared, and the position pinned thereafter. If dead cars kept moving,
    they could keep earning progress after hitting a wall.
    """
    state = init_state(1, TRACK)
    for _ in range(600):
        _scripted_step(state, steer=1.0, throttle=1.0)  # spiral into a wall
        if state.crashed[0]:
            break

    assert state.crashed[0], "hard constant steer should hit a wall"
    assert not state.alive[0], "crashed cars are no longer alive"

    frozen_pos = state.pos.copy()
    frozen_progress = state.progress.copy()
    for _ in range(10):
        _scripted_step(state, steer=1.0, throttle=1.0)
    np.testing.assert_array_equal(state.pos, frozen_pos,
                                  err_msg="dead car must not move")
    np.testing.assert_array_equal(state.progress, frozen_progress)
    assert state.crashed[0] and not state.alive[0]


def test_run_episode_zero_genomes_all_stall():
    """End-to-end sanity for the scoring rules on a degenerate population:
    all-zero genomes yield tanh(0)=0 controls, so nobody moves, nobody
    crashes, everyone earns exactly 0 fitness, and the episode terminates at
    the first stall check instead of burning all max_steps.
    """
    pop = 4
    genomes = np.zeros((pop, SPEC.genome_size), dtype=np.float32)
    result = run_episode(genomes, TRACK, CONFIG)

    assert isinstance(result, EpisodeResult)
    np.testing.assert_array_equal(result.fitness, np.zeros(pop))
    np.testing.assert_array_equal(result.progress, np.zeros(pop))
    assert not result.crashed.any(), "standing still is not a crash"
    assert result.steps_run == CONFIG.sim.stall_check_every, \
        "episode should end the moment the whole population is stall-killed"


def test_fitness_equals_laps_minus_crash_penalty():
    """The fitness definition is the contract with the GA: lap fraction minus
    a small penalty only for crashers. Verifying it elementwise from the
    EpisodeResult fields guards against the penalty silently leaking onto
    non-crashed cars (or crashers escaping it).
    """
    rng = np.random.default_rng(123)
    genomes = init_population(8, SPEC, rng, scale=CONFIG.evo.init_scale)
    result = run_episode(genomes, TRACK, CONFIG, max_steps=400)

    # Mirror the module's arithmetic exactly (float32 progress -> float64
    # ratio -> penalty -> cast) for a bit-exact comparison.
    lap_frac = result.progress / TRACK.total_length
    expected = (lap_frac
                - CONFIG.sim.crash_penalty * result.crashed).astype(np.float32)
    np.testing.assert_array_equal(result.fitness, expected)
    np.testing.assert_array_equal(result.laps, lap_frac.astype(np.float32))
    # Same law through the reported (rounded) laps field, within float32 noise.
    np.testing.assert_allclose(
        result.fitness,
        result.laps - CONFIG.sim.crash_penalty * result.crashed,
        rtol=0.0, atol=1e-6)


def test_on_step_callback_aborts_early_and_sees_time_advance():
    """The on_step hook is how viewers observe (and close) an episode. It must
    see time moving forward and returning False must abort immediately —
    that is the only way a pygame window close can stop a run cleanly.
    """
    genomes = np.zeros((3, SPEC.genome_size), dtype=np.float32)
    seen_t: list[int] = []

    def on_step(state) -> bool | None:
        seen_t.append(state.t)
        return False if state.t >= 5 else None  # None (not False) continues

    result = run_episode(genomes, TRACK, CONFIG, on_step=on_step)

    assert result.steps_run == 5 < CONFIG.sim.max_steps
    assert seen_t == [1, 2, 3, 4, 5], \
        "callback runs once per step with strictly increasing state.t"


def test_run_episode_is_deterministic():
    """Reproducibility is a design pillar: the sim uses no RNG, so identical
    (genomes, track, config) must give bit-identical results. Any drift here
    would make training runs impossible to replay or debug.
    """
    rng = np.random.default_rng(99)
    genomes = init_population(6, SPEC, rng, scale=CONFIG.evo.init_scale)

    r1 = run_episode(genomes, TRACK, CONFIG, max_steps=300)
    r2 = run_episode(genomes, TRACK, CONFIG, max_steps=300)

    np.testing.assert_array_equal(r1.fitness, r2.fitness)
    np.testing.assert_array_equal(r1.progress, r2.progress)
    np.testing.assert_array_equal(r1.steps_alive, r2.steps_alive)
    np.testing.assert_array_equal(r1.crashed, r2.crashed)
    assert r1.steps_run == r2.steps_run


# ---------------------------------------------------------------------------
# delta-ray (closure-rate) observations and the alive-subset optimization
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402

from racing.simulation import _assemble_obs, build_obs, init_ray_history  # noqa: E402

DELTA_CONFIG = dataclasses.replace(
    CONFIG, sensor=dataclasses.replace(CONFIG.sensor, delta_rays=True))
DELTA_SPEC = make_spec(DELTA_CONFIG.brain, DELTA_CONFIG.sensor)


def _delta_track(difficulty=0.6):
    # A harder track so cars die at different times, exercising the
    # alive-subset code path (not everyone alive to the end).
    return make_track(seed=13, difficulty=difficulty, cfg=DELTA_CONFIG.track,
                      car_radius=DELTA_CONFIG.car.car_radius)


def test_delta_rays_dont_leak_across_cars():
    """THE load-bearing invariant for the alive-subset speedup with a shared
    ray-history buffer: because cars never interact, one car's fitness must be
    identical whether it is simulated alone or inside a batch. If the closure-
    rate buffer leaked a neighbor's history (or a skipped dead row corrupted
    it), a car's result would depend on its batch — silently breaking both
    determinism and fair selection.
    """
    track = _delta_track()
    genomes = init_population(24, DELTA_SPEC, np.random.default_rng(3),
                              scale=DELTA_CONFIG.evo.init_scale)
    batch = run_episode(genomes, track, DELTA_CONFIG)
    for i in (0, 7, 15, 23):
        solo = run_episode(genomes[i:i + 1], track, DELTA_CONFIG)
        assert solo.fitness[0] == batch.fitness[i], f"car {i} batch-dependent"
        assert solo.crashed[0] == batch.crashed[i]


def test_delta_channel_is_zero_at_first_step():
    """The buffer is seeded by sensing at the start pose, so the closure-rate
    channel must be ~0 on step 0 (the car hasn't moved) — NOT the saturated
    garbage a zero-fill would feed tanh on the first, decisive step."""
    track = _delta_track(0.3)
    state = init_state(4, track)
    init_ray_history(state, track, DELTA_CONFIG, RAYS_DELTA)
    obs = build_obs(state, track, DELTA_CONFIG, RAYS_DELTA)
    n_rays = len(DELTA_CONFIG.sensor.ray_angles_deg)
    delta_channel = obs[:, n_rays:2 * n_rays]
    np.testing.assert_array_equal(delta_channel, np.zeros_like(delta_channel))


def test_delta_obs_width_and_capacity_control():
    """Delta and its capacity control must share input width and genome size
    (so an A/B isolates information from parameters), and the plain obs must
    stay narrow."""
    n_rays = len(CONFIG.sensor.ray_angles_deg)
    dists = np.ones((3, n_rays), dtype=np.float32)
    speed = np.full((3, 1), 0.5, dtype=np.float32)
    prev = dists - 0.01
    plain = _assemble_obs(dists, speed, None, None, CONFIG.sensor)
    delta = _assemble_obs(dists, speed, prev, None, DELTA_CONFIG.sensor)
    cap_cfg = dataclasses.replace(
        CONFIG.sensor, capacity_control=True)
    cap = _assemble_obs(dists, speed, None, None, cap_cfg)
    assert plain.shape[1] == n_rays + 1
    assert delta.shape[1] == cap.shape[1] == 2 * n_rays + 1
    # closure-rate channel = gain * (dists - prev)
    np.testing.assert_allclose(
        delta[:, n_rays:2 * n_rays],
        DELTA_CONFIG.sensor.delta_gain * (dists - prev), rtol=1e-5)


RAYS_DELTA = ray_geometry(DELTA_CONFIG.sensor)


# ---------------------------------------------------------------------------
# recurrent (Elman) brains
# ---------------------------------------------------------------------------

from racing.brain import forward_recurrent  # noqa: E402

REC_CONFIG = dataclasses.replace(
    CONFIG, brain=dataclasses.replace(CONFIG.brain, recurrent=True))
REC_SPEC = make_spec(REC_CONFIG.brain, REC_CONFIG.sensor)


def test_recurrent_genome_size_and_forward_shapes():
    """W_rec adds hidden^2 genes; the recurrent step must return controls in
    [-1,1] and a bounded hidden state the caller threads forward."""
    assert REC_SPEC.genome_size == SPEC.genome_size + REC_SPEC.hidden ** 2
    genomes = init_population(4, REC_SPEC, np.random.default_rng(0))
    obs = np.random.default_rng(1).uniform(0, 1, (4, REC_SPEC.n_in)).astype(np.float32)
    h0 = np.zeros((4, REC_SPEC.hidden), dtype=np.float32)
    ctrl, h1 = forward_recurrent(genomes, obs, h0, REC_SPEC)
    assert ctrl.shape == (4, 2) and h1.shape == (4, REC_SPEC.hidden)
    assert np.all(np.abs(ctrl) < 1.0) and np.all(np.abs(h1) < 1.0)
    # memory actually matters: a different h_prev changes the output
    ctrl2, _ = forward_recurrent(genomes, obs, h1, REC_SPEC)
    assert not np.array_equal(ctrl, ctrl2)


def test_recurrent_batch_independence():
    """Same invariant as delta-rays: with the alive-subset optimization
    gathering/scattering h rows, a car's episode must not depend on its
    batch — if a dead neighbor's stale h ever leaked into a living car's
    update, fitness would become batch-dependent."""
    track = make_track(seed=13, difficulty=0.6, cfg=REC_CONFIG.track,
                       car_radius=REC_CONFIG.car.car_radius)
    genomes = init_population(24, REC_SPEC, np.random.default_rng(5))
    batch = run_episode(genomes, track, REC_CONFIG)
    for i in (0, 9, 23):
        solo = run_episode(genomes[i:i + 1], track, REC_CONFIG)
        assert solo.fitness[0] == batch.fitness[i], f"car {i} batch-dependent"


# ---------------------------------------------------------------------------
# multi-car heats
# ---------------------------------------------------------------------------

HEAT_CONFIG = dataclasses.replace(
    CONFIG, sim=dataclasses.replace(CONFIG.sim, heat_size=4))


def test_heat_spawn_grid_no_contact():
    """The staggered grid must spawn every car clear of its heat-mates —
    starting inside a rival is a crash before the race begins."""
    from racing.simulation import _car_contacts
    state = init_state(8, TRACK, heat_size=4)
    assert not _car_contacts(state.pos, 4, CONFIG.car.car_radius).any()
    # back rows start slightly behind the line: progress <= front row's
    assert state.progress.min() < state.progress.max() + 1e-6
    assert state.alive.all()


def test_car_contact_crashes_both():
    """Contact between two living cars must crash them both — collision is
    symmetric, and the physics freeze then keeps them as wreckage."""
    state = init_state(4, TRACK, heat_size=4)
    state.pos[1] = state.pos[0] + np.array([CONFIG.car.car_radius, 0.0],
                                           dtype=np.float32)  # overlapping
    controls = np.zeros((4, 2), dtype=np.float32)
    step(state, TRACK, None, SPEC, HEAT_CONFIG, RAYS, controls=controls)
    assert state.crashed[0] and state.crashed[1]
    assert not state.crashed[2] and not state.crashed[3]


def test_rays_see_a_heatmate_as_wall():
    """A rival parked dead ahead must shorten the forward ray reading —
    that is the only channel through which cars perceive each other."""
    from racing.simulation import _sense_other_cars
    state = init_state(4, TRACK, heat_size=4)
    # place car 1 exactly 60 px ahead of car 0 along its heading
    fwd = np.array([np.cos(state.heading[0]), np.sin(state.heading[0])])
    state.pos[1] = state.pos[0] + (60.0 * fwd).astype(np.float32)
    state.pos[2] += 300.0  # move the others far away
    state.pos[3] += 300.0
    rel, ts, lengths = RAYS
    car_d = _sense_other_cars(state, 4, rel, lengths, CONFIG.car.car_radius)
    i0 = list(CONFIG.sensor.ray_angles_deg).index(0.0)
    expected = (60.0 - 2 * CONFIG.car.car_radius) / lengths[i0]
    assert abs(car_d[0, i0] - expected) < 0.02
    # the car behind reads nothing ahead of it from car 0's rear
    assert car_d[1, i0] == 1.0
