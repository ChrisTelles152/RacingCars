"""The episode: run a whole population on one track, headlessly, and score it.

This module is the seam between everything else — sensors feed brains, brains
feed physics, physics feeds progress tracking — and it owns the rules of the
game:

- **Fitness = progress / track_length** (lap fraction). Progress is monotone:
  a car is credited with the *furthest* point it ever reached, measured along
  the centerline. Lap fractions are comparable across tracks of different
  lengths, which matters because training uses fresh random tracks constantly.
- **No explicit speed bonus.** Faster cars simply get further before the step
  cap — rewarding speed directly invites fitness hacking.
- **Crash** = the car center enters the radius-inflated collision grid
  (one array lookup). Crashed cars keep their progress minus a small penalty:
  crashing just short of X must score worse than calmly stopping at X.
- **Stagnation kill**: a car that gains almost no progress over a trailing
  window is parked, circling, or wiggling against a wall — it is frozen so
  the episode can end early. Combined with monotone progress and no reverse
  gear, this closes the known reward-hacking exploits.

`run_episode` is the ONLY episode loop in the codebase. The pygame viewers
pass an `on_step` callback and merely observe the arrays; headless training
passes nothing. Same code path, so watched and headless runs cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .brain import BrainSpec, forward, make_spec
from .car import step_cars
from .config import Config
from .sensors import ray_geometry, sense
from .track import Track, project_progress


@dataclass
class SimState:
    """Structure-of-arrays state for P cars on one track."""
    pos: np.ndarray        # (P, 2) float32
    heading: np.ndarray    # (P,)   float32
    speed: np.ndarray      # (P,)   float32
    alive: np.ndarray      # (P,)   bool
    crashed: np.ndarray    # (P,)   bool (subset of not-alive)
    cl_idx: np.ndarray     # (P,)   int32, current nearest checkpoint
    laps: np.ndarray       # (P,)   int32, signed start-line crossings
    progress: np.ndarray   # (P,)   float32, monotone-max arc length incl. laps
    stall_ref: np.ndarray  # (P,)   float32, progress at last stall check
    steps_alive: np.ndarray  # (P,) int32
    t: int = 0
    # Ray-history ring buffer for delta-ray (closure-rate) observations —
    # (stride, P, R), holding the last `stride` ray-distance snapshots; None
    # unless config.sensor.delta_rays. hist_pos points at the oldest slot.
    ray_hist: np.ndarray | None = None
    hist_pos: int = 0
    # Populated by run_episode for observers (viewers draw sensor rays etc.).
    last_obs: np.ndarray | None = None
    last_controls: np.ndarray | None = None


def init_state(pop: int, track: Track) -> SimState:
    """Everyone starts at the start line, pointed along the track, at rest."""
    return SimState(
        pos=np.tile(track.start_pos, (pop, 1)).astype(np.float32),
        heading=np.full(pop, track.start_heading, dtype=np.float32),
        speed=np.zeros(pop, dtype=np.float32),
        alive=np.ones(pop, dtype=bool),
        crashed=np.zeros(pop, dtype=bool),
        cl_idx=np.zeros(pop, dtype=np.int32),
        laps=np.zeros(pop, dtype=np.int32),
        progress=np.zeros(pop, dtype=np.float32),
        stall_ref=np.zeros(pop, dtype=np.float32),
        steps_alive=np.zeros(pop, dtype=np.int32),
    )


def init_ray_history(state: SimState, track: Track, config: Config,
                     rays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Seed the closure-rate buffer by sensing once at the start pose.

    Filling every slot with the initial reading makes the closure rate ~0 at
    t=0 (the car hasn't moved) instead of the garbage a zero-fill would feed
    tanh on the first, most important decision.
    """
    if not config.sensor.delta_rays:
        return
    rel_angles, sample_ts, lengths = rays
    dists = sense(state.pos, state.heading, track.occ_sensor,
                  config.sensor, rel_angles, sample_ts, lengths)
    state.ray_hist = np.repeat(dists[None], config.sensor.delta_stride, axis=0)
    state.hist_pos = 0


def _assemble_obs(dists: np.ndarray, speed_n: np.ndarray,
                  prev_dists: np.ndarray | None, sensor_cfg) -> np.ndarray:
    """Build observation rows from ray distances + normalized speed.

    Plain:    [rays, speed].
    Delta:    [rays, gain*(rays - rays_stride_ago), speed] — appends the
              closure rate, which is what distance-alone can't convey
              (time-to-collision needs velocity, not just position).
    Capacity: [rays, rays, speed] — the delta-arm's control: same width and
              genome size, zero new information.
    """
    if sensor_cfg.delta_rays:
        delta = (sensor_cfg.delta_gain * (dists - prev_dists)).astype(np.float32)
        return np.concatenate([dists, delta, speed_n], axis=1)
    if sensor_cfg.capacity_control:
        return np.concatenate([dists, dists, speed_n], axis=1)
    return np.concatenate([dists, speed_n], axis=1)


def _sense_and_assemble(state: SimState, track: Track, config: Config,
                        rays: tuple[np.ndarray, np.ndarray, np.ndarray],
                        idx: np.ndarray | None) -> np.ndarray:
    """Full-width (P, n_in) observation, sensing only rows in `idx` (None=all).

    For delta-rays this also advances the ring buffer: it reads each computed
    row's snapshot from `stride` steps ago, then overwrites that slot with the
    current reading. Only computed (alive) rows are touched, and because a
    living car was alive at every earlier step too, its buffer history is
    identical whether or not dead peers were skipped — so the alive-subset
    optimization stays bit-for-bit faithful to the full-population path.
    """
    rel_angles, sample_ts, lengths = rays
    sc = config.sensor
    pop = state.pos.shape[0]
    rows = slice(None) if idx is None else idx

    dists = sense(state.pos[rows], state.heading[rows], track.occ_sensor,
                  sc, rel_angles, sample_ts, lengths)
    speed_n = (state.speed[rows] / config.car.v_max).astype(np.float32)[:, None]

    prev = None
    if sc.delta_rays:
        # Copy, not a view: for idx=None `ray_hist[hist_pos][:]` is a view of
        # the slot we overwrite on the next line, which would zero every
        # closure rate in the full-population path (and silently diverge from
        # the alive-subset path, where fancy indexing already copies).
        prev = np.array(state.ray_hist[state.hist_pos][rows])  # rays `stride` steps ago
        state.ray_hist[state.hist_pos][rows] = dists           # overwrite that slot
        state.hist_pos = (state.hist_pos + 1) % sc.delta_stride

    row_obs = _assemble_obs(dists, speed_n, prev, sc)
    if idx is None:
        return row_obs
    obs = np.zeros((pop, row_obs.shape[1]), dtype=np.float32)
    obs[idx] = row_obs
    return obs


def build_obs(state: SimState, track: Track, config: Config,
              rays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Full-population observation vector (P, n_in)."""
    return _sense_and_assemble(state, track, config, rays, idx=None)


def step(state: SimState, track: Track, genomes: np.ndarray | None, spec: BrainSpec,
         config: Config, rays: tuple[np.ndarray, np.ndarray, np.ndarray],
         controls: np.ndarray | None = None) -> None:
    """Advance the whole population one timestep: sense -> think -> move -> score.

    Normally the brains produce the controls; pass `controls` explicitly to
    drive the cars from outside (human keyboard input in play.py) while
    keeping every other rule — physics, collision, progress, stalls —
    identical to what the evolved cars experience.
    """
    sim = config.sim
    g = track.occ_coll.shape[0]
    pop = state.pos.shape[0]

    # Sense and think only for the living. Early generations are dominated by
    # dead cars (most crash within seconds) and ray sensing is ~3/4 of step
    # cost, so subsetting the two expensive stages is a large speedup — with
    # bit-identical results, because every car's rows are computed
    # independently of the others'.
    if controls is None:
        alive_idx = np.flatnonzero(state.alive)
        if alive_idx.size == pop:
            obs = _sense_and_assemble(state, track, config, rays, idx=None)
            controls = forward(genomes, obs, spec)
        else:
            obs = _sense_and_assemble(state, track, config, rays, idx=alive_idx)
            controls = np.zeros((pop, 2), dtype=np.float32)
            controls[alive_idx] = forward(genomes[alive_idx], obs[alive_idx], spec)
    else:
        obs = build_obs(state, track, config, rays)
    step_cars(state.pos, state.heading, state.speed, state.alive, controls, config.car)

    # Collision: one gather in the configuration-space grid.
    xi = np.clip(np.round(state.pos[:, 0]).astype(np.int32), 0, g - 1)
    yi = np.clip(np.round(state.pos[:, 1]).astype(np.int32), 0, g - 1)
    crashed_now = track.occ_coll[yi, xi] & state.alive  # grid indexes [y, x]
    state.crashed |= crashed_now
    state.alive &= ~crashed_now

    # Progress: windowed nearest-checkpoint search + wrap-aware lap counting.
    new_idx = project_progress(track, state.pos, state.cl_idx,
                               sim.progress_window_back, sim.progress_window_fwd)
    m = track.n_checkpoints
    state.laps += (new_idx < state.cl_idx - m // 2).astype(np.int32)  # crossed start forward
    state.laps -= (new_idx > state.cl_idx + m // 2).astype(np.int32)  # crossed it backward
    state.cl_idx = new_idx
    raw = state.laps.astype(np.float32) * track.total_length + track.cum_s[new_idx]
    np.maximum(state.progress, raw, out=state.progress)  # monotone: best point ever reached

    # Stagnation: periodically freeze cars that stopped making progress.
    state.t += 1
    if state.t % sim.stall_check_every == 0:
        stalled = (state.progress - state.stall_ref) < sim.stall_min_progress
        state.alive &= ~stalled
        state.stall_ref = state.progress.copy()

    state.steps_alive += state.alive
    state.last_obs = obs
    state.last_controls = controls


@dataclass
class EpisodeResult:
    fitness: np.ndarray      # (P,) float32, lap fractions minus crash penalty
    progress: np.ndarray     # (P,) float32, px along centerline incl. laps
    laps: np.ndarray         # (P,) float32, progress / track_length
    steps_alive: np.ndarray  # (P,) int32
    crashed: np.ndarray      # (P,) bool
    final_ck: np.ndarray | None = None  # (P,) int32 last checkpoint index —
    # for crashed cars this is (approximately) where they died
    steps_run: int = 0


def run_episode(genomes: np.ndarray, track: Track, config: Config,
                on_step=None, max_steps: int | None = None) -> EpisodeResult:
    """Score every genome on this track. Deterministic; the sim itself uses no RNG.

    `on_step(state)` is an observe-only hook for renderers; returning False
    from it aborts the episode (viewer window closed).
    """
    spec = make_spec(config.brain, config.sensor)
    rays = ray_geometry(config.sensor)
    if max_steps is None:
        max_steps = config.sim.max_steps

    state = init_state(genomes.shape[0], track)
    init_ray_history(state, track, config, rays)
    while state.t < max_steps and state.alive.any():
        step(state, track, genomes, spec, config, rays)
        if on_step is not None and on_step(state) is False:
            break

    lap_frac = state.progress / track.total_length
    fitness = lap_frac - config.sim.crash_penalty * state.crashed
    return EpisodeResult(
        fitness=fitness.astype(np.float32),
        progress=state.progress.copy(),
        laps=lap_frac.astype(np.float32),
        steps_alive=state.steps_alive.copy(),
        crashed=state.crashed.copy(),
        final_ck=state.cl_idx.copy(),
        steps_run=state.t,
    )
