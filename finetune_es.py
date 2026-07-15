#!/usr/bin/env python3
"""Fine-tune a trained champion with Evolution Strategies (racing/es.py).

Where the GA explores broadly with a population, ES polishes locally: start
from the champion, estimate the fitness gradient by sampling small
perturbations, and climb it. Fine-tuning happens on FRESH hard tracks
(difficulty 0.9-1.3), concentrating effort exactly where the curriculum's
trailing band trains least.

The output is NOT auto-shipped: compare it against the original on the
decision suite (compare.py or evaluate.py) — the pre-registered guardrail is
no regression > 0.1 mean laps anywhere.

Usage:
  python3 finetune_es.py --champion runs/flag-101/champion_best_val.npz
  # -> writes <champion dir>/champion_es.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from racing.es import es_step
from racing.persistence import load_genome, save_genome
from racing.simulation import run_episode
from racing.track import make_track


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--pairs", type=int, default=64,
                    help="mirrored sample pairs per iteration (2x evaluations)")
    ap.add_argument("--sigma", type=float, default=0.03,
                    help="perturbation scale — small: polish, don't wander")
    ap.add_argument("--alpha", type=float, default=0.02, help="learning rate")
    ap.add_argument("--tracks-per-iter", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    genome, config, meta = load_genome(args.champion)
    theta = genome.astype(np.float32)
    rng = np.random.default_rng(args.seed)

    print(f"ES fine-tune of {args.champion} (genome {theta.size}) — "
          f"{args.iters} iters x {2 * args.pairs} evals x "
          f"{args.tracks_per_iter} fresh hard tracks")

    for it in range(args.iters):
        # Fresh hard tracks each iteration: the objective stays "drive well
        # on unseen hard tracks", never "memorize these tracks".
        tracks = []
        while len(tracks) < args.tracks_per_iter:
            seed = int(rng.integers(1 << 20, 1 << 31))
            d = float(rng.uniform(0.9, config.track.max_difficulty))
            t = make_track(seed, d, config.track, config.car.car_radius)
            if abs(t.difficulty - min(d, config.track.max_difficulty)) < 1e-9:
                tracks.append(t)

        def eval_batch(candidates: np.ndarray) -> np.ndarray:
            per_track = np.stack([run_episode(candidates, t, config).fitness
                                  for t in tracks])
            return per_track.mean(axis=0)

        t0 = time.perf_counter()
        theta, best, mean = es_step(theta, args.sigma, args.alpha,
                                    args.pairs, eval_batch, rng)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  candidate best {best:6.3f} "
                  f"mean {mean:6.3f}  ({time.perf_counter() - t0:.1f}s/iter)")

    out = os.path.join(os.path.dirname(args.champion), "champion_es.npz")
    save_genome(out, theta, config,
                {**meta, "es_iters": args.iters, "es_sigma": args.sigma,
                 "es_alpha": args.alpha, "es_seed": args.seed})
    print(f"wrote {out}\nnow gate it:\n  python3 evaluate.py --suite decision "
          f"{args.champion} {out}")


if __name__ == "__main__":
    main()
