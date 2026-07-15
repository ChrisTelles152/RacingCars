#!/usr/bin/env python3
"""Exhibition race: put N champions on one track, contact rules ON.

Because rivals read as walls on the ordinary sensor rays, solo-trained
champions race unmodified — no retraining, no new inputs. Watching them
negotiate traffic they never trained for is itself a generalization test.

Keys: 1/2/4/8 sim speed, S rays for the leader, R restart, N new track, ESC.

Usage:
  python3 race.py --champions runs/a/champion_best_val.npz runs/b/... \
      [--track-seed 999] [--difficulty 0.9]
  python3 race.py --champions ... --gif out.gif   # headless GIF instead
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from racing.persistence import load_genome
from racing.simulation import run_episode
from racing.track import make_track


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champions", nargs="+", required=True)
    ap.add_argument("--track-seed", type=int, default=None)
    ap.add_argument("--difficulty", type=float, default=0.9)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif", default=None, help="render headless to a GIF")
    args = ap.parse_args()

    loaded = [load_genome(p) for p in args.champions]
    base_config = loaded[0][1]
    # Same world required; the master seed is training history, not physics.
    def world(cfg):
        return dataclasses.replace(cfg, seed=0).to_json()
    for path, (_, cfg, _) in zip(args.champions, loaded):
        if world(cfg) != world(base_config):
            raise SystemExit(f"{path}: config differs — racers must share a world")
    genomes = np.stack([g for g, _, _ in loaded]).astype(np.float32)
    n = len(genomes)
    config = dataclasses.replace(
        base_config, sim=dataclasses.replace(base_config.sim, heat_size=n))

    rng = np.random.default_rng()
    seed = (args.track_seed if args.track_seed is not None
            else int(rng.integers(1 << 20, 1 << 31)))
    track = make_track(seed, args.difficulty, config.track,
                       config.car.car_radius)

    if args.gif:
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # after the driver env var
    from PIL import Image
    from viewer import Viewer

    viewer = Viewer(config.track.world_size,
                    title=f"RacingCars — {n}-car heat")
    frames: list = []
    speed_mult = 2 if not args.gif else 8

    def on_step(state):
        nonlocal speed_mult
        if not args.gif:
            alive_now, keys = viewer.poll()
            if not alive_now:
                return False
            for key in keys:
                if key in (pygame.K_1, pygame.K_2, pygame.K_4, pygame.K_8):
                    speed_mult = int(pygame.key.name(key))
        if state.t % speed_mult:
            return True
        viewer.draw_track(track)
        leader = int(np.argmax(state.progress))
        viewer.draw_cars(state.pos, state.heading, state.alive, best_idx=leader)
        order = np.argsort(-state.progress)
        standings = "  ".join(f"P{rank+1}:car{idx}" for rank, idx
                              in enumerate(order[:4]))
        viewer.draw_hud([
            f"{n}-car heat  seed {seed}  d={args.difficulty:.2f}  t {state.t}",
            f"standings: {standings}   alive {int(state.alive.sum())}/{n}",
        ])
        viewer.flip(fps=None if args.gif else args.fps)
        if args.gif:
            frames.append(Image.fromarray(
                pygame.surfarray.array3d(viewer.screen).transpose(1, 0, 2)))
        return True

    result = run_episode(genomes, track, config, on_step=on_step)
    order = np.argsort(-result.progress)
    print("\nFINAL STANDINGS")
    for rank, idx in enumerate(order):
        status = "CRASHED" if result.crashed[idx] else "finished"
        print(f"  P{rank+1}  car {idx} ({args.champions[idx]}): "
              f"{result.laps[idx]:.2f} laps, {status}")
    if args.gif and frames:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=40, loop=0)
        print(f"wrote {args.gif}")
    pygame.quit()


if __name__ == "__main__":
    main()
