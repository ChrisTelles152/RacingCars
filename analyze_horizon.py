#!/usr/bin/env python3
"""Did doubling the training horizon make radar discovery RELIABLE?

Because every RNG stream is seeded per generation, a 600-generation run
replays its own first 300 generations exactly — so one run answers both
"what would we have shipped at gen 300?" and "...at gen 600?" as a perfectly
paired comparison, with no separate control arm.

The counterfactual champion for each horizon is chosen by the run's OWN
selection rule (best validation worst-case, from the archived metadata) —
never by peeking at the cone probe. Selecting on the reported metric would
manufacture the result this script exists to measure.

Reliability is a variance question, so the headline is the SOLVE RATE
(fraction of seeds under the threshold), not the mean.

Usage:
  python3 analyze_horizon.py --prefix radarlong --seeds 101 102 103 104 105
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from evaluate import SUITE_TRACK
from racing.persistence import load_genome
from racing.simulation import run_episode
from racing.track import make_track
from train import champion_key

SOLVED = 0.10          # cone-crash rate that counts as "solved"
PROBE_SEEDS = range(33600, 33625)   # 25 cone tracks, disjoint from every suite


def probe_tracks(config):
    return [make_track(s, 1.0, SUITE_TRACK["obstacle"], config.car.car_radius)
            for s in PROBE_SEEDS]


def cone_crash_rate(genome, config, tracks) -> float:
    crashes = sum(bool(run_episode(genome[None, :], t, config).crashed[0])
                  for t in tracks)
    return crashes / len(tracks)


def pick_by_validation(paths, max_gen):
    """The champion this run WOULD have shipped had it stopped at max_gen."""
    best, best_key = None, None
    for p in paths:
        _, _, meta = load_genome(p)
        if meta.get("generation", 0) > max_gen:
            continue
        if "val_min" not in meta:
            continue
        key = champion_key(meta["val_min"], meta["val_mean"])
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="radarlong")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[101, 102, 103, 104, 105])
    ap.add_argument("--horizons", type=int, nargs="+", default=[300, 600])
    args = ap.parse_args()

    tracks = None
    results: dict[int, list[float]] = {h: [] for h in args.horizons}
    print(f"cone-crash rate of the champion each run would have shipped\n"
          f"(probe: {len(PROBE_SEEDS)} held-out cone tracks at d=1.0; "
          f"'solved' = below {SOLVED:.0%})\n")
    header = "  seed  " + "".join(f"{'gen ' + str(h):>12}" for h in args.horizons)
    print(header)
    for seed in args.seeds:
        run = f"runs/{args.prefix}-{seed}"
        paths = sorted(glob.glob(os.path.join(run, "champions", "gen_*.npz")))
        if not paths:
            print(f"  {seed}: no archived champions in {run}")
            continue
        row = f"  {seed:>4}  "
        for h in args.horizons:
            pick = pick_by_validation(paths, h)
            if pick is None:
                row += f"{'--':>12}"
                continue
            genome, config, _ = load_genome(pick)
            if tracks is None:
                tracks = probe_tracks(config)
            rate = cone_crash_rate(genome, config, tracks)
            results[h].append(rate)
            mark = "*" if rate < SOLVED else " "
            row += f"{rate:>11.0%}{mark}"
        print(row)

    print(f"\n{'horizon':>10}  {'solve rate':>11}  {'median':>8}  {'worst':>7}")
    for h in args.horizons:
        r = np.array(results[h])
        if not len(r):
            continue
        print(f"{'gen ' + str(h):>10}  "
              f"{(r < SOLVED).mean():>10.0%}  {np.median(r):>8.0%}  {r.max():>7.0%}"
              f"   ({(r < SOLVED).sum()}/{len(r)} seeds solved)")


if __name__ == "__main__":
    main()
