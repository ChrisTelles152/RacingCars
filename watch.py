#!/usr/bin/env python3
"""Watch a saved champion drive — ideally a track it has never seen.

The champion's checkpoint embeds the exact config it was trained with, so it
always drives the same physics it evolved in.

Keys while watching:
  1 / 2 / 4 / 8   simulation speed (steps per rendered frame)
  S               toggle sensor rays
  N               skip to a new random track
  R               restart on the same track
  ESC             quit

Usage:
  python3 watch.py --champion runs/demo/champion_best_val.npz --track-seed 999
  python3 watch.py --champion ... --difficulty 0.8      # random hard tracks
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from racing.persistence import load_genome
from racing.simulation import run_episode
from racing.track import make_track
from viewer import Viewer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", required=True, help=".npz checkpoint from train.py")
    ap.add_argument("--track-seed", type=int, default=None,
                    help="fixed track seed (default: fresh random each time)")
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    genome, config, meta = load_genome(args.champion)
    genomes = genome[None, :]
    if meta:
        print("champion metadata:", {k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in meta.items()})

    viewer = Viewer(config.track.world_size, title="RacingCars — champion")
    rng = np.random.default_rng()
    show_rays = True
    speed_mult = 2

    def next_seed() -> int:
        return args.track_seed if args.track_seed is not None else int(rng.integers(1 << 20, 1 << 31))

    seed = next_seed()
    running = True
    while running:
        track = make_track(seed, args.difficulty, config.track, config.car.car_radius)
        outcome = {"restart": False}

        def on_step(state) -> bool | None:
            nonlocal running, show_rays, speed_mult, seed
            alive_now, keys = viewer.poll()
            if not alive_now:
                running = False
                return False
            for key in keys:
                if key == pygame.K_s:
                    show_rays = not show_rays
                elif key in (pygame.K_1, pygame.K_2, pygame.K_4, pygame.K_8):
                    speed_mult = int(pygame.key.name(key))
                elif key == pygame.K_n:
                    seed = seed + 1 if args.track_seed is not None else next_seed()
                    outcome["restart"] = True
                    return False
                elif key == pygame.K_r:
                    outcome["restart"] = True
                    return False
            if state.t % speed_mult:
                return True  # skip rendering this step, keep simulating

            viewer.draw_track(track)
            if show_rays and state.alive[0]:
                viewer.draw_rays(track, state.pos, state.heading, config.sensor, [0])
            viewer.draw_cars(state.pos, state.heading, state.alive)
            laps = state.progress[0] / track.total_length
            viewer.draw_hud([
                f"track seed {seed}  difficulty {args.difficulty:.2f}  (unseen by training)",
                f"t {state.t:4d}   progress {laps:5.2f} laps   speed {state.speed[0]:5.1f} px/s",
                f"speed x{speed_mult}  |  1/2/4/8 speed | S rays | N new track | R restart | ESC quit",
            ])
            viewer.flip(fps=args.fps)
            return True

        result = run_episode(genomes, track, config, on_step=on_step)
        if not running:
            break
        if outcome["restart"]:
            continue

        # Episode over: show the final verdict until the user picks an action.
        laps = float(result.laps[0])
        verdict = "CRASHED" if result.crashed[0] else "finished"
        waiting = True
        while running and waiting:
            running, keys = viewer.poll()
            for key in keys:
                if key == pygame.K_n:
                    seed = seed + 1 if args.track_seed is not None else next_seed()
                    waiting = False
                elif key == pygame.K_r:
                    waiting = False
            viewer.draw_hud([f"{verdict}: {laps:.2f} laps in {result.steps_run} steps"
                             f"   —   N new track | R replay | ESC quit"])
            viewer.flip(fps=30)

    pygame.quit()


if __name__ == "__main__":
    main()
