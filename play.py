#!/usr/bin/env python3
"""Drive a procedurally generated track yourself — arrow keys.

This is the physics/sensor test bench: you experience exactly the same
world the evolved cars do (same step function, same collision rules, same
stagnation kill — park too long and you're "dead" too).

Keys:
  arrows  steer (left/right) and throttle/brake (up/down)
  S       toggle sensor rays
  R       reset to the start line
  N       new random track (same difficulty)
  ESC     quit

Usage:
  python3 play.py [--track-seed 7] [--difficulty 0.3]
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from racing.brain import make_spec
from racing.config import Config
from racing.sensors import ray_geometry
from racing.simulation import init_state, step
from racing.track import make_track
from viewer import Viewer


def nonneg_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-seed", type=nonneg_int, default=7)
    ap.add_argument("--difficulty", type=float, default=0.3)
    args = ap.parse_args()

    config = Config()
    spec = make_spec(config.brain, config.sensor)
    rays = ray_geometry(config.sensor)

    seed = args.track_seed
    track = make_track(seed, args.difficulty, config.track, config.car.car_radius)
    viewer = Viewer(config.track.world_size, title="RacingCars — you drive")
    state = init_state(1, track)
    show_rays = True

    running = True
    while running:
        running, keys = viewer.poll()
        for key in keys:
            if key == pygame.K_r:
                state = init_state(1, track)
            elif key == pygame.K_n:
                seed += 1
                track = make_track(seed, args.difficulty, config.track, config.car.car_radius)
                state = init_state(1, track)
            elif key == pygame.K_s:
                show_rays = not show_rays

        pressed = pygame.key.get_pressed()
        steer = float(pressed[pygame.K_RIGHT]) - float(pressed[pygame.K_LEFT])
        throttle = float(pressed[pygame.K_UP]) - float(pressed[pygame.K_DOWN])
        controls = np.array([[steer, throttle]], dtype=np.float32)

        if state.alive[0]:
            step(state, track, None, spec, config, rays, controls=controls)

        viewer.draw_track(track)
        if show_rays and state.alive[0]:
            viewer.draw_rays(track, state.pos, state.heading, config.sensor, [0])
        viewer.draw_cars(state.pos, state.heading, state.alive)

        laps = state.progress[0] / track.total_length
        status = []
        if state.crashed[0]:
            status = ["CRASHED — press R to reset"]
        elif not state.alive[0]:
            status = ["STALLED OUT (the evolution kill rule got you) — press R"]
        viewer.draw_hud([
            f"track seed {seed}  difficulty {args.difficulty:.2f}",
            f"progress {laps:5.2f} laps   speed {state.speed[0]:5.1f} px/s",
            "arrows drive | S rays | R reset | N new track | ESC quit",
            *status,
        ])
        viewer.flip(fps=30)

    pygame.quit()


if __name__ == "__main__":
    main()
