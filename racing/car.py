"""Vectorized kinematic car physics for the whole population at once.

The state is a "structure of arrays": pos (P, 2), heading (P,), speed (P,) —
one numpy array per property, not one Python object per car. Advancing the
entire population is then a handful of array operations regardless of P.

The model is deliberately simple (no drift, no slip — the ML is the point):

    speed   += (throttle * accel - drag * speed) * dt,   clipped to [0, v_max]
    heading += steer * steer_rate * authority(speed) * dt
    pos     += speed * (cos(heading), sin(heading)) * dt

Two design choices double as anti-cheat rules baked into physics:
- speed is clipped at 0 — **no reverse gear**, so "drive backwards over the
  start line" can never be a fitness strategy;
- steering authority scales with speed, so a car cannot spin in place.

Dead cars are frozen by multiplying with the alive mask instead of being
removed — at this population size, masked math is cheaper than bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .config import CarConfig


def step_cars(
    pos: np.ndarray,      # (P, 2) float32, updated in place
    heading: np.ndarray,  # (P,)   float32, radians, updated in place
    speed: np.ndarray,    # (P,)   float32, updated in place
    alive: np.ndarray,    # (P,)   bool
    controls: np.ndarray, # (P, 2) float32 in [-1, 1]: [steer, throttle]
    cfg: CarConfig,
) -> None:
    """Advance every alive car by one fixed timestep dt. Dead cars stay put."""
    live = alive.astype(np.float32)
    steer = controls[:, 0]
    throttle = controls[:, 1]  # negative throttle = brake (speed clip stops reverse)

    authority = cfg.steer_speed_floor + (1.0 - cfg.steer_speed_floor) * speed / cfg.v_max
    heading += steer * cfg.steer_rate * authority * cfg.dt * live

    speed += (throttle * cfg.accel - cfg.drag * speed) * cfg.dt * live
    np.clip(speed, 0.0, cfg.v_max, out=speed)
    speed *= live  # freeze the dead: zero speed means zero displacement below

    pos[:, 0] += np.cos(heading) * speed * cfg.dt
    pos[:, 1] += np.sin(heading) * speed * cfg.dt
