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
    base_radius: float = 380.0

    # M: the centerline is resampled to this many *equally spaced* points.
    # These double as progress checkpoints; spacing = track_length / M (~6 px).
    n_checkpoints: int = 400
    samples_per_segment: int = 40  # dense spline samples before resampling

    # Difficulty knobs. Each value is interpolated easy -> hard by d in [0, 1].
    control_points_easy: int = 8      # fewer points = gentler, rounder track
    control_points_hard: int = 14
    radial_jitter_easy: float = 0.12  # fraction of base_radius
    radial_jitter_hard: float = 0.40
    half_width_easy: float = 50.0     # px from centerline to wall
    half_width_hard: float = 22.0
    angle_jitter: float = 0.25        # fraction of even angular spacing

    # Validity (rejection sampling): a candidate track is rejected when a
    # corner is too tight for its width, or two sections pinch together.
    min_radius_margin: float = 1.6  # min curvature radius > margin * half_width
    pinch_margin: float = 2.4       # non-adjacent points must be farther apart
    pinch_skip: int = 30            # index distance that counts as non-adjacent
    max_attempts: int = 50


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
    ray_angles_deg: tuple[float, ...] = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)
    ray_length: float = 160.0   # px, max sensing range
    n_samples: int = 24         # S: samples marched along each ray (grid lookup)


@dataclass(frozen=True)
class BrainConfig:
    # Fixed topology MLP: (n_rays + 1 speed input) -> hidden -> 2, tanh both
    # layers. Deliberately tiny: small nets mutate well and generalize better.
    hidden: int = 16


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
    # K fresh random tracks per generation; fitness = mean over the K tracks.
    # New tracks every generation is the anti-overfitting mechanism: there is
    # nothing to memorize.
    tracks_per_generation: int = 3
    # Curriculum: difficulty d ratchets up as the population gets good.
    # Per-track difficulty is sampled from the trailing band [d - band, d]
    # so easy tracks stay in the mix (no catastrophic forgetting).
    curriculum_band: float = 0.2
    promote_threshold: float = 0.7  # median lap-fraction that counts as "good"
    promote_streak: int = 3         # consecutive good generations to promote
    promote_step: float = 0.1
    demote_after: int = 20          # generations stuck -> ease off
    demote_step: float = 0.05
    # Held-out validation: fixed seeds NEVER used in training. The train-vs-
    # validation gap is the honest overfitting signal.
    val_every: int = 10
    val_seeds: tuple[int, ...] = tuple(range(10_000, 10_010))
    val_difficulties: tuple[float, ...] = (0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 1.0, 1.0)


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
            # JSON turns tuples into lists; restore tuples for frozen fields.
            for f in dataclasses.fields(cls):
                if f.type.startswith("tuple") and isinstance(sub.get(f.name), list):
                    sub[f.name] = tuple(sub[f.name])
            kwargs[name] = cls(**sub)
        return Config(**kwargs)
