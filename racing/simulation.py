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


def build_obs(state: SimState, track: Track, config: Config,
              rel_angles: np.ndarray, sample_ts: np.ndarray) -> np.ndarray:
    """Observation vector (P, R+1): normalized ray distances + own speed."""
    rays = sense(state.pos, state.heading, track.occ_sensor,
                 config.sensor, rel_angles, sample_ts)
    speed_n = (state.speed / config.car.v_max).astype(np.float32)[:, None]
    return np.concatenate([rays, speed_n], axis=1)


def step(state: SimState, track: Track, genomes: np.ndarray | None, spec: BrainSpec,
         config: Config, rel_angles: np.ndarray, sample_ts: np.ndarray,
         controls: np.ndarray | None = None) -> None:
    """Advance the whole population one timestep: sense -> think -> move -> score.

    Normally the brains produce the controls; pass `controls` explicitly to
    drive the cars from outside (human keyboard input in play.py) while
    keeping every other rule — physics, collision, progress, stalls —
    identical to what the evolved cars experience.
    """
    sim = config.sim
    g = track.occ_coll.shape[0]

    obs = build_obs(state, track, config, rel_angles, sample_ts)
    if controls is None:
        controls = forward(genomes, obs, spec)
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
    steps_run: int = 0


def run_episode(genomes: np.ndarray, track: Track, config: Config,
                on_step=None, max_steps: int | None = None) -> EpisodeResult:
    """Score every genome on this track. Deterministic; the sim itself uses no RNG.

    `on_step(state)` is an observe-only hook for renderers; returning False
    from it aborts the episode (viewer window closed).
    """
    spec = make_spec(config.brain, config.sensor)
    rel_angles, sample_ts = ray_geometry(config.sensor)
    if max_steps is None:
        max_steps = config.sim.max_steps

    state = init_state(genomes.shape[0], track)
    while state.t < max_steps and state.alive.any():
        step(state, track, genomes, spec, config, rel_angles, sample_ts)
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
        steps_run=state.t,
    )
