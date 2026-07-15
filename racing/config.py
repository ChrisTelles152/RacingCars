"""Every hyperparameter of the simulator and the GA, in one place.

Each subsystem gets its own small frozen dataclass; ``Config`` composes them
all plus the master seed. Frozen means accidental mutation mid-run is an
error — a run's behavior is fully determined by (Config, master seed).

Why this matters for learning ML: almost every question you'll ask
("what if mutation were stronger?", "do narrower tracks need more rays?")
is answered by changing one number here and re-running with the same seed.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrackConfig:
    # World: square grid, 1 pixel = 1 world unit. Tracks are closed loops
    # around the center. All rasterized lookups index grid[y, x].
    world_size: int = 1024
    # Keep base_radius * (1 + radial_jitter) + half_width < world_size / 2
    # or tracks fall off the grid (validity check also enforces this).
    base_radius: float = 320.0

    # M: the centerline is resampled to this many *equally spaced* points.
    # These double as progress checkpoints; spacing = track_length / M (~6 px).
    n_checkpoints: int = 400
    samples_per_segment: int = 40  # dense spline samples before resampling

    # Difficulty knobs. Each value is interpolated easy -> hard by d in [0, 1].
    # Hard-end values are calibrated so d=1.0 generation reliably succeeds
    # within max_attempts (measured 0/150 seeds failing; at 14 points and
    # 0.40 jitter it was ~20% — rejection sampling needs viable candidates).
    control_points_easy: int = 8      # fewer points = gentler, rounder track
    control_points_hard: int = 12
    radial_jitter_easy: float = 0.12  # fraction of base_radius
    radial_jitter_hard: float = 0.34
    half_width_easy: float = 50.0     # px from centerline to wall
    half_width_hard: float = 22.0
    angle_jitter: float = 0.25        # fraction of even angular spacing

    # Curriculum overshoot: training difficulty may exceed 1.0 so that the
    # d=1.0 evaluation point sits INSIDE the training distribution instead of
    # at its extrapolation edge (models are reliable where they interpolate).
    # "Extreme" values are the knob settings at max_difficulty.
    max_difficulty: float = 1.3
    half_width_extreme: float = 18.0        # car is 12 px wide — still fits
    control_points_extreme: int = 13
    radial_jitter_extreme: float = 0.38

    # Validity (rejection sampling): a candidate track is rejected when a
    # corner is too tight for its width, or two sections pinch together.
    min_radius_margin: float = 1.6  # min curvature radius > margin * half_width
    pinch_margin: float = 2.4       # non-adjacent points must be farther apart
    pinch_skip: int = 30            # index distance that counts as non-adjacent
    max_attempts: int = 50

    # --- variable corridor width (Sprint 3; default OFF) ---
    # Fractional width modulation via a loop-periodic Fourier profile:
    # w(s) = half_width * (1 + amp(d) * f(s)), amp(d) = width_profile_amp *
    # min(d, 1). Wide sections pinching into narrow ones is a strictly richer
    # corner vocabulary than any constant-width knob — and a policy trained
    # only on constant width implicitly learns "walls are equidistant".
    # Default 0.0 keeps every existing track (and frozen suite) bit-identical.
    width_profile_amp: float = 0.0
    width_profile_terms: int = 3

    # --- straight-into-hairpin traps (Sprint 3; default OFF) ---
    # With probability trap_prob * ramp(d) (ramp reaches 1 at d=0.9), rewrite
    # a few control points into a long straight feeding a near-limit corner —
    # the exact pattern that kills champions, promoted from rare accident to
    # curriculum regular (hard-example mining in the task generator). The
    # validity checks still veto anything undrivable, so trap sharpness
    # settles at the hardest *accepted* geometry.
    trap_prob: float = 0.0


@dataclass(frozen=True)
class CarConfig:
    # Kinematic model — no drift or slip; the point is ML, not vehicle dynamics.
    dt: float = 1.0 / 30.0
    v_max: float = 300.0        # px/s
    accel: float = 250.0        # px/s^2 at full throttle
    drag: float = 0.5           # 1/s, opposes speed
    steer_rate: float = 3.5     # rad/s at full lock and full speed
    # Steering authority scales with speed so cars can't spin in place:
    # authority = floor + (1 - floor) * speed / v_max
    steer_speed_floor: float = 0.3
    car_radius: float = 6.0     # collision circle radius


@dataclass(frozen=True)
class SensorConfig:
    # Ray-cast distance sensors, angles relative to the car's heading.
    # The fan is dense and LONG toward the front: braking from v_max to
    # tight-corner speed consumes ~125 px, so corner geometry must be
    # readable well beyond that — a policy can never out-drive its sensors
    # (perception horizon >= stopping distance, the classic robotics rule).
    ray_angles_deg: tuple[float, ...] = (
        -90.0, -45.0, -22.0, -10.0, -4.0, 0.0, 4.0, 10.0, 22.0, 45.0, 90.0)
    # Per-ray range (px), same order as the angles: long ahead, SHORT sides.
    # None -> every ray uses the scalar ray_length (old checkpoints rely on
    # this fallback, so their 7-ray configs keep loading unchanged).
    # The short side rays are the Sprint-2 "precision" win (was 160/160/220):
    # a shorter ray marches the same 36 samples over less distance, so its
    # quantization drops from ~4.4 px to ~1.9 px — enough to thread the tight
    # corridors the heatmap flagged. Halved the max-difficulty crash rate at
    # zero genome/compute cost. `--variant baseline` restores the old ranges.
    ray_lengths: tuple[float, ...] | None = (
        70.0, 90.0, 140.0, 300.0, 300.0, 300.0, 300.0, 300.0, 140.0, 90.0, 70.0)
    ray_length: float = 300.0   # px, fallback range when ray_lengths is None
    n_samples: int = 36         # S: samples marched along each ray (grid lookup)

    # --- observation augmentation (Sprint 2 experiments; default off) ---
    # Delta-rays: append each ray's CLOSURE RATE (this step's reading minus
    # the reading delta_stride steps ago, times delta_gain). Distance alone
    # makes the world partially observed — "wall 40px ahead" means something
    # different at 300px/s than at 80px/s; the closure rate restores the
    # missing velocity, i.e. time-to-collision. Doubles the ray inputs.
    delta_rays: bool = False
    delta_stride: int = 4       # steps between the two snapshots (>1: one-step
                                # deltas sit at the sensor quantization floor)
    delta_gain: float = 8.0     # scale so typical closure rates land in tanh's
                                # responsive band rather than near zero
    # Capacity control for the delta-ray A/B: append DUPLICATED current rays
    # instead of closure rates — identical input width and genome size, zero
    # new information. Delta-rays must beat THIS to prove the info helped
    # rather than just the extra parameters. Mutually exclusive with delta_rays.
    capacity_control: bool = False


@dataclass(frozen=True)
class BrainConfig:
    # Fixed topology MLP: (n_rays + 1 speed input) -> hidden -> 2, tanh both
    # layers. Deliberately tiny: small nets mutate well and generalize better.
    hidden: int = 16
    # Elman recurrence (default off): the hidden layer also receives its own
    # previous activation through an evolved W_rec — h_t = tanh(W_in x_t +
    # W_rec h_{t-1} + b). A reactive policy is capped when state is hidden
    # (POMDP); memory lets the net integrate history. Neuroevolution trains
    # recurrent nets exactly like feedforward ones — no backprop through
    # time, the genome just grows by hidden^2 weights.
    recurrent: bool = False


@dataclass(frozen=True)
class EvoConfig:
    population: int = 512       # P: vectorization sweet spot on Apple CPUs
    elite: int = 8              # copied verbatim each generation
    tournament_k: int = 3       # selection pressure knob
    # Fraction of children built by uniform per-gene crossover of two parents
    # (the rest are mutated copies of one parent). Set to 0.0 to run the
    # classic "does crossover even help for NN weights?" experiment.
    crossover_rate: float = 0.5
    mutation_prob: float = 0.10   # per-gene chance of being perturbed
    sigma_init: float = 0.20      # Gaussian mutation strength...
    sigma_decay: float = 0.995    # ...decaying per generation (coarse -> fine)
    sigma_floor: float = 0.02
    heavy_tail_prob: float = 0.10   # some children mutate much harder:
    heavy_tail_scale: float = 5.0   # a cheap escape hatch from local optima
    init_scale: float = 1.0         # multiplier on N(0, 1/sqrt(fan_in)) init
    # Self-adaptive mutation (Rechenberg/Schwefel): each genome carries its
    # OWN sigma, inherited from its parent and perturbed log-normally —
    # selection tunes the step size instead of a hand-written clock schedule
    # (which is nearly frozen by the time the curriculum gets hard, whether
    # or not the population still needs exploration there).
    self_adaptive_sigma: bool = False
    sigma_tau: float = 0.2          # log-normal perturbation strength
    sigma_min: float = 0.005
    sigma_max: float = 0.5
    # Island model (default off): partition the population into `islands`
    # sub-populations that evolve independently, with the best few genomes
    # migrating around a ring every migrate_every generations. Diversity by
    # STRUCTURE — separated gene pools drift apart (allopatric divergence),
    # so one driving style cannot take over the whole population.
    islands: int = 1
    migrate_every: int = 25
    migrants: int = 2


@dataclass(frozen=True)
class SimConfig:
    max_steps: int = 2000         # ~66 sim-seconds at dt = 1/30
    # Progress lookup searches only a window of checkpoints around each car's
    # last known index — cheap, and progress physically cannot teleport across
    # a nearby fold of track. v_max * dt must stay well inside the window.
    progress_window_back: int = 4
    progress_window_fwd: int = 20
    # Stagnation kill: a car gaining < min_progress px over the trailing
    # window is parked/circling — freeze it so the episode can end early.
    stall_check_every: int = 120  # steps (4 sim-seconds)
    stall_min_progress: float = 2.0  # px
    crash_penalty: float = 0.02   # fitness penalty (in lap fractions)


@dataclass(frozen=True)
class TrainConfig:
    generations: int = 300
    # K fresh random tracks per generation. New tracks every generation is
    # the anti-overfitting mechanism: there is nothing to memorize.
    tracks_per_generation: int = 3
    # How per-track fitness aggregates into one score ("the aggregation
    # function is part of the reward design"): 'mean' trades rare crashes for
    # average speed; 'min' scores you by your worst track; 'cvar' is the mean
    # of the worst ceil(cvar_frac * K) tracks — risk-sensitive middle ground.
    # Default cvar: crashing on the occasional tightest corner is exactly a
    # tail event that mean-fitness never punishes.
    fitness_agg: str = "cvar"       # 'mean' | 'min' | 'cvar'
    cvar_frac: float = 0.5          # with K=3: mean of the worst 2 tracks
    # Domain randomization over DYNAMICS (default off): each training episode
    # scales accel/drag/steer_rate/v_max by U(1-band, 1+band). A policy that
    # cannot rely on exact physics learns margins instead of memorized braking
    # points — the sim2real workhorse. Evaluation always runs nominal physics.
    physics_rand: float = 0.0
    # Curriculum: difficulty d ratchets up as the population gets good.
    # Per-track difficulty is sampled from the trailing band [d - band, d]
    # so easy tracks stay in the mix (no catastrophic forgetting).
    curriculum_band: float = 0.2
    promote_threshold: float = 0.7  # median lap-fraction that counts as "good"
    promote_streak: int = 3         # consecutive good generations to promote
    promote_step: float = 0.1
    demote_after: int = 20          # consecutive below-bar gens -> ease off
    demote_step: float = 0.05
    # Held-out validation: fixed seeds NEVER used in training. The ladder is
    # 50 tracks, weighted toward the hard end, because champion selection
    # keys on the WORST validation track: with only 2 hard tracks (the
    # original ladder) a genome that crashes on half of all hard tracks
    # still aces both with probability 25% — measured to lock lucky fragile
    # champions into the checkpoint ratchet in 3 of 4 runs. With 15 hard
    # tracks that pass probability is ~0.003%.
    val_every: int = 10
    val_seeds: tuple[int, ...] = tuple(range(10_000, 10_050))
    val_difficulties: tuple[float, ...] = (
        (0.3,) * 5 + (0.5,) * 5 + (0.7,) * 10 + (0.9,) * 15 + (1.0,) * 15)
    # Robust champion selection: argmax over K noisy tracks is a lottery (the
    # winner's curse), so each validation round races the top-N train genomes
    # on the validation set and keeps the one with the best WORST-case score.
    champion_candidates: int = 5
    # Held-out TEST bank (evaluate.py): seeds and per-difficulty count for the
    # final honest number. Used for selection exactly zero times — validation
    # drives checkpointing 30+ times per run, so it is slowly overfit too.
    test_seed_base: int = 20_000
    test_per_difficulty: int = 25
    test_difficulties: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.0)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    track: TrackConfig = field(default_factory=TrackConfig)
    car: CarConfig = field(default_factory=CarConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    evo: EvoConfig = field(default_factory=EvoConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Config":
        raw = json.loads(text)
        parts = {
            "track": TrackConfig, "car": CarConfig, "sensor": SensorConfig,
            "brain": BrainConfig, "evo": EvoConfig, "sim": SimConfig,
            "train": TrainConfig,
        }
        kwargs: dict = {"seed": raw["seed"]}
        for name, cls in parts.items():
            sub = dict(raw[name])
            # Checkpoints saved before per-ray ranges existed have no
            # ray_lengths key; they must get the uniform-range fallback, not
            # today's default tuple (which matches today's angle fan only).
            if cls is SensorConfig and "ray_lengths" not in sub:
                sub["ray_lengths"] = None
            # JSON turns tuples into lists; restore tuples for frozen fields.
            for f in dataclasses.fields(cls):
                if f.type.startswith("tuple") and isinstance(sub.get(f.name), list):
                    sub[f.name] = tuple(sub[f.name])
            kwargs[name] = cls(**sub)
        return Config(**kwargs)
