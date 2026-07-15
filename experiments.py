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

# Sprint-1 heatmap diagnosis: the residual d=1.0 failures are on the
# corridor-NARROWNESS axis, so the lateral rays' coarse ~4.4 px quantization
# (160 px range / 36 samples) against a ~12 px corridor margin is a prime
# suspect. Shortening the side rays sharpens them to ~1.9 px at zero compute
# or genome cost (same ray count, same sample count, shorter range = finer
# steps). Forward rays stay long for braking sight.
_PRECISION_LENGTHS = (70.0, 90.0, 140.0, 300.0, 300.0, 300.0, 300.0, 300.0,
                      140.0, 90.0, 70.0)


def _sensor(config: Config, **kw) -> Config:
    return dataclasses.replace(
        config, sensor=dataclasses.replace(config.sensor, **kw))


VARIANTS: dict[str, "callable"] = {
    "baseline": lambda c: c,
    # Sharper lateral perception (config-only, clean 2-arm A/B).
    "precision": lambda c: _sensor(c, ray_lengths=_PRECISION_LENGTHS),
    # Closure-rate inputs (time-to-collision) + its capacity control.
    "delta": lambda c: _sensor(c, delta_rays=True),
    "deltacap": lambda c: _sensor(c, capacity_control=True),
    # Combination arm, filled in only if both wins compose.
    "precision_delta": lambda c: _sensor(
        c, ray_lengths=_PRECISION_LENGTHS, delta_rays=True),
}


def apply_variant(config: Config, name: str) -> Config:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; have {sorted(VARIANTS)}")
    return VARIANTS[name](config)
