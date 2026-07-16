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

from .brain import BrainSpec, forward, forward_recurrent, make_spec
from .car import step_cars
from .config import Config
from .sensors import radar_nearest, ray_geometry, sense
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
    # Recurrent hidden state (P, hidden); None unless the brain is recurrent.
    h: np.ndarray | None = None
    # Populated by run_episode for observers (viewers draw sensor rays etc.).
    last_obs: np.ndarray | None = None
    last_controls: np.ndarray | None = None


def init_state(pop: int, track: Track, heat_size: int = 0) -> SimState:
    """Spawn the field.

    Ghost mode (heat_size=0): everyone at the start line, superimposed —
    cars don't interact, so sharing a point is fine and keeps comparisons
    fair. Heat mode: a staggered starting grid (rows of two, offset back
    along the track and laterally) so cars don't begin inside each other.
    """
    if heat_size <= 0:
        pos = np.tile(track.start_pos, (pop, 1)).astype(np.float32)
        heading = np.full(pop, track.start_heading, dtype=np.float32)
        cl_idx = np.zeros(pop, dtype=np.int32)
        laps = np.zeros(pop, dtype=np.int32)
        progress = np.zeros(pop, dtype=np.float32)
    else:
        if pop % heat_size:
            raise ValueError(f"population {pop} not divisible into heats of {heat_size}")
        m = track.n_checkpoints
        widths = (track.half_widths if track.half_widths is not None
                  else np.full(m, track.half_width, dtype=np.float32))
        j = np.arange(pop) % heat_size          # grid slot within the heat
        ck = (-(j // 2) * 5) % m                # rows of 2, ~24 px apart
        lateral = np.where(j % 2 == 0, -0.35, 0.35) * widths[ck]
        pos = (track.centerline[ck]
               + track.normals[ck] * lateral[:, None]).astype(np.float32)
        heading = np.arctan2(track.tangents[ck, 1],
                             track.tangents[ck, 0]).astype(np.float32)
        cl_idx = ck.astype(np.int32)
        # Rows behind the start line begin at slightly negative progress —
        # that's just what a starting grid is.
        laps = np.where(ck > m // 2, -1, 0).astype(np.int32)
        progress = (laps * track.total_length
                    + track.cum_s[ck]).astype(np.float32)
    return SimState(
        pos=pos,
        heading=heading,
        speed=np.zeros(pop, dtype=np.float32),
        alive=np.ones(pop, dtype=bool),
        crashed=np.zeros(pop, dtype=bool),
        cl_idx=cl_idx,
        laps=laps,
        progress=progress.copy() if heat_size > 0 else progress,
        stall_ref=progress.copy(),
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


def _sense_other_cars(state: SimState, heat_size: int, rel_angles: np.ndarray,
                      lengths: np.ndarray, car_radius: float) -> np.ndarray:
    """(P, R) normalized ray distance to the nearest same-heat car.

    Ray-circle intersection, vectorized per heat: with heats of H the tensor
    is (heats, H, R, H) — tiny. Crashed cars stay on track as wreckage and
    still read on the rays (and still hurt to touch): a race line through a
    pile-up is part of the game.
    """
    pop = state.pos.shape[0]
    nh, h = pop // heat_size, heat_size
    r_count = rel_angles.shape[0]
    pos = state.pos.reshape(nh, h, 2)
    ang = (state.heading[:, None] + rel_angles[None, :]).reshape(nh, h, r_count)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=-1)        # (nh, H, R, 2)

    to_c = pos[:, None, None, :, :] - pos[:, :, None, None, :]  # (nh,H,1,H,2)
    t = (to_c * d[:, :, :, None, :]).sum(-1)                 # (nh, H, R, H)
    center_d2 = (to_c ** 2).sum(-1)                          # (nh, H, 1, H)
    perp2 = center_d2 - t ** 2
    r2 = (2.0 * car_radius) ** 2  # circle-vs-circle: sum of both radii
    hit = (t > 0.0) & (perp2 < r2)
    t_hit = np.where(hit, t - np.sqrt(np.maximum(r2 - perp2, 0.0)), np.inf)
    # A car never senses itself.
    eye = np.eye(h, dtype=bool)[None, :, None, :]
    t_hit = np.where(eye, np.inf, t_hit)
    nearest = t_hit.min(axis=3).reshape(pop, r_count)        # (P, R)
    return np.minimum(nearest / lengths[None, :], 1.0).astype(np.float32)


def _car_contacts(pos: np.ndarray, heat_size: int, car_radius: float) -> np.ndarray:
    """(P,) bool: car center within 2*car_radius of any same-heat car."""
    pop = pos.shape[0]
    nh, h = pop // heat_size, heat_size
    p = pos.reshape(nh, h, 2)
    d2 = ((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1)  # (nh, H, H)
    d2[:, np.arange(h), np.arange(h)] = np.inf                 # not yourself
    return (d2 < (2.0 * car_radius) ** 2).any(axis=2).reshape(pop)


def _assemble_obs(dists: np.ndarray, speed_n: np.ndarray,
                  prev_dists: np.ndarray | None,
                  radar: np.ndarray | None, sensor_cfg) -> np.ndarray:
    """Build observation rows from ray distances + normalized speed.

    Plain:    [rays, speed].
    Delta:    [rays, gain*(rays - rays_stride_ago), speed] — appends the
              closure rate, which is what distance-alone can't convey
              (time-to-collision needs velocity, not just position).
    Capacity: [rays, rays, speed] — the delta-arm's control: same width and
              genome size, zero new information.
    Radar:    appends the 3-channel nearest-obstacle readout before speed.
    """
    if sensor_cfg.delta_rays:
        delta = (sensor_cfg.delta_gain * (dists - prev_dists)).astype(np.float32)
        parts = [dists, delta]
    elif sensor_cfg.capacity_control:
        parts = [dists, dists]
    else:
        parts = [dists]
    if radar is not None:
        parts.append(radar)
    parts.append(speed_n)
    return np.concatenate(parts, axis=1)


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
    if config.sim.heat_size > 0:
        # Other cars read as walls on the same rays: no new input channel,
        # so a solo-trained champion can race unmodified.
        car_dists = _sense_other_cars(state, config.sim.heat_size, rel_angles,
                                      lengths, config.car.car_radius)
        dists = np.minimum(dists, car_dists[rows])
    speed_n = (state.speed[rows] / config.car.v_max).astype(np.float32)[:, None]

    prev = None
    if sc.delta_rays:
        if state.ray_hist is None:
            raise RuntimeError(
                "delta_rays config requires init_ray_history(state, ...) before "
                "stepping — run_episode() does this automatically")
        # Copy, not a view: for idx=None `ray_hist[hist_pos][:]` is a view of
        # the slot we overwrite on the next line, which would zero every
        # closure rate in the full-population path (and silently diverge from
        # the alive-subset path, where fancy indexing already copies).
        prev = np.array(state.ray_hist[state.hist_pos][rows])  # rays `stride` steps ago
        state.ray_hist[state.hist_pos][rows] = dists           # overwrite that slot
        state.hist_pos = (state.hist_pos + 1) % sc.delta_stride

    radar = None
    if sc.obstacle_radar:
        radar = radar_nearest(state.pos[rows], state.heading[rows],
                              track.obstacle_pos, sc)

    row_obs = _assemble_obs(dists, speed_n, prev, radar, sc)
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
            if spec.recurrent:
                controls, state.h = forward_recurrent(genomes, obs, state.h, spec)
            else:
                controls = forward(genomes, obs, spec)
        else:
            obs = _sense_and_assemble(state, track, config, rays, idx=alive_idx)
            controls = np.zeros((pop, 2), dtype=np.float32)
            if spec.recurrent:
                # Gather/compute/scatter only living rows: a dead car's h is
                # frozen (never read again), and living cars' rows evolve
                # identically to the full-width path — batch-independence.
                c, h_new = forward_recurrent(genomes[alive_idx],
                                             obs[alive_idx],
                                             state.h[alive_idx], spec)
                controls[alive_idx] = c
                state.h[alive_idx] = h_new
            else:
                controls[alive_idx] = forward(genomes[alive_idx], obs[alive_idx], spec)
    else:
        obs = build_obs(state, track, config, rays)

    grip = None
    if track.surface is not None:  # local grip under each car, one gather
        gx = np.clip(np.round(state.pos[:, 0]).astype(np.int32), 0, g - 1)
        gy = np.clip(np.round(state.pos[:, 1]).astype(np.int32), 0, g - 1)
        grip = track.surface[gy, gx]
    step_cars(state.pos, state.heading, state.speed, state.alive, controls,
              config.car, grip)

    # Collision: one gather in the configuration-space grid.
    xi = np.clip(np.round(state.pos[:, 0]).astype(np.int32), 0, g - 1)
    yi = np.clip(np.round(state.pos[:, 1]).astype(np.int32), 0, g - 1)
    crashed_now = track.occ_coll[yi, xi] & state.alive  # grid indexes [y, x]
    if sim.heat_size > 0:
        # Car-on-car contact crashes the LIVING party; wreckage stays put
        # (already dead) and remains a hazard for everyone behind it.
        crashed_now |= (_car_contacts(state.pos, sim.heat_size,
                                      config.car.car_radius) & state.alive)
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

    state = init_state(genomes.shape[0], track, config.sim.heat_size)
    init_ray_history(state, track, config, rays)
    if spec.recurrent:
        state.h = np.zeros((genomes.shape[0], spec.hidden), dtype=np.float32)
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
