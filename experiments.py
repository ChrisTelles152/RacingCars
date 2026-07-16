"""Named config variants for A/B experiments (Sprint 2+).

Each variant is a pure function Config -> Config, so every experiment arm is
built from the SAME base config and the same master seeds — the paired /
common-random-numbers design that compare.py relies on. Keeping the variants
in one registry (rather than scattered CLI flags) means an experiment is
fully described by a name, and the name lands in the run directory and the
checkpoint, so results stay traceable.

Run an arm with:  python3 train.py --variant <name> --seed <s>
"""

from __future__ import annotations

import dataclasses

from racing.config import Config

# The short side rays that WON Sprint 2 are now the config default (the
# flagship), so "precision" is the identity variant. To reproduce the
# pre-Sprint-2 flagship for comparison, the "baseline" variant restores the
# long side rays. (Sprint-1 heatmap diagnosis: residual d=1.0 failures were
# on the corridor-narrowness axis; sharpening the lateral rays from ~4.4 px
# to ~1.9 px quantization halved the max-difficulty crash rate.)
_BASELINE_LENGTHS = (160.0, 160.0, 220.0, 300.0, 300.0, 300.0, 300.0, 300.0,
                     220.0, 160.0, 160.0)

# Follow-up (obstacle blind-spot fix): the flagship crashes on 97% of
# mid-corridor cone tracks because 12 px cones slip between the flagship's
# 4-6 deg forward ray spacing past ~120 px — the failure is ANGULAR aliasing,
# the exact analog of the DISTANCE aliasing that "precision" fixed. So the fix
# is angular resolution: a denser forward fan (17 rays, ~2-3 deg spacing over
# +-16 deg), which catches a cone out to ~200 px instead of ~120. Short side
# rays are kept (the precision win). Genome grows 242 -> 338.
_DENSE_ANGLES = (-90.0, -45.0, -28.0, -16.0, -11.0, -7.0, -4.0, -2.0, 0.0,
                 2.0, 4.0, 7.0, 11.0, 16.0, 28.0, 45.0, 90.0)
_DENSE_LENGTHS = (70.0, 100.0, 160.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0,
                  300.0, 300.0, 300.0, 300.0, 300.0, 160.0, 100.0, 70.0)


def _sensor(config: Config, **kw) -> Config:
    return dataclasses.replace(
        config, sensor=dataclasses.replace(config.sensor, **kw))


def _dense(config: Config) -> Config:
    return _sensor(config, ray_angles_deg=_DENSE_ANGLES,
                   ray_lengths=_DENSE_LENGTHS)


VARIANTS: dict[str, "callable"] = {
    # The current flagship (config default: short/precise side rays).
    "precision": lambda c: c,
    # Pre-Sprint-2 flagship: long side rays (~4.4 px lateral quantization).
    "baseline": lambda c: _sensor(c, ray_lengths=_BASELINE_LENGTHS),
    # KILLED in Sprint 2: closure-rate inputs (delta) were WORSE than their
    # own capacity control on hard tracks — the velocity information hurt
    # where position precision was needed. Kept for reproducibility/teaching.
    "delta": lambda c: _sensor(c, delta_rays=True),
    "deltacap": lambda c: _sensor(c, capacity_control=True),
    # Self-adaptive per-genome mutation strength (Sprint-2 leftover, run in
    # Sprint 3): evolution tunes its own step size instead of the decay clock.
    "sigma": lambda c: dataclasses.replace(
        c, evo=dataclasses.replace(c.evo, self_adaptive_sigma=True)),
    # Sprint 3: variable corridor width (pinches) in training tracks.
    "width": lambda c: _track(c, width_profile_amp=0.3),
    # Sprint 3: straight-into-hairpin traps in training tracks.
    "traps": lambda c: _track(c, trap_prob=0.7),
    # Sprint 3: domain randomization over dynamics during training.
    "physrand": lambda c: dataclasses.replace(
        c, train=dataclasses.replace(c.train, physics_rand=0.15)),
    # Deferred item: Elman recurrent hidden layer (memory). Genome grows by
    # hidden^2 = 256; if this WINS its A/B, a capacity control must confirm
    # the memory (not the parameters) did it — the delta-ray lesson.
    "recurrent": lambda c: dataclasses.replace(
        c, brain=dataclasses.replace(c.brain, recurrent=True)),
    # Deferred item: island model (diversity by structure).
    "islands": lambda c: dataclasses.replace(
        c, evo=dataclasses.replace(c.evo, islands=4)),
    # Deferred item: low-grip surface zones (blind — no grip sensor).
    "lowgrip": lambda c: _track(c, grip_zones=2),
    # Deferred item: chicane cones stamped into the corridor.
    "obstacles": lambda c: _track(c, obstacles=3),
    # Deferred item (endpoint of the environment axis): heats of 8 that
    # sense and collide with each other — difficulty from other drivers.
    "multicar": lambda c: dataclasses.replace(
        c, sim=dataclasses.replace(c.sim, heat_size=8)),
    # Multicar ON TOP of the shipped flagship (precision defaults + width):
    # the arm that answers "does racing traffic teach anything solo driving
    # didn't?" — gated on solo non-regression plus race performance.
    "flag_multicar": lambda c: dataclasses.replace(
        _track(c, width_profile_amp=0.3),
        sim=dataclasses.replace(c.sim, heat_size=8)),
    # Obstacle blind-spot fix. Two arms probe whether the 97%-crash failure
    # is PERCEPTION (can't see cones) or POLICY (sees but doesn't avoid):
    #  densefan     = flagship + dense forward rays, NO obstacle training.
    #                 If cones become visible, the existing wall-avoidance may
    #                 dodge them with no new training — the elegant outcome.
    #  densefan_obs = dense rays + obstacle training. If perception ALONE
    #                 isn't enough, exposure on top should finish the job —
    #                 and better sight may avoid the blanket-caution tax that
    #                 sank plain obstacle training (-1.7 laps everywhere).
    "densefan": lambda c: _track(_dense(c), width_profile_amp=0.3),
    "densefan_obs": lambda c: _track(_dense(c), width_profile_amp=0.3,
                                     obstacles=3),
    # Obstacle fix, round 2: a dedicated RADAR channel (nearest cone's
    # distance + bearing, alignment-independent) on the ordinary flagship
    # fan, + obstacle training. Tests whether the right MODALITY beats more
    # angular resolution: dense-fan+training plateaued at 64% crash because
    # rays only report cones that happen to line up; radar always reports
    # the true bearing. Genome 242 -> 290 (vs densefan_obs's 338 — if radar
    # wins with FEWER params, the information/parameters question answers
    # itself and no capacity control is needed).
    "radar": lambda c: _track(_sensor(c, obstacle_radar=True),
                              width_profile_amp=0.3, obstacles=3),
}


def _track(config: Config, **kw) -> Config:
    return dataclasses.replace(
        config, track=dataclasses.replace(config.track, **kw))


def apply_variant(config: Config, name: str) -> Config:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; have {sorted(VARIANTS)}")
    return VARIANTS[name](config)
