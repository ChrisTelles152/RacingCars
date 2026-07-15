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


def _sensor(config: Config, **kw) -> Config:
    return dataclasses.replace(
        config, sensor=dataclasses.replace(config.sensor, **kw))


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
}


def _track(config: Config, **kw) -> Config:
    return dataclasses.replace(
        config, track=dataclasses.replace(config.track, **kw))


def apply_variant(config: Config, name: str) -> Config:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; have {sorted(VARIANTS)}")
    return VARIANTS[name](config)
