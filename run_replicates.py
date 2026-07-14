#!/usr/bin/env python3
"""Run N replicate training runs (one per master seed) as parallel processes.

One training run is one anecdote. Replicates on a fixed seed list give every
claim an error bar, and — because every experiment arm reuses the SAME seed
list — paired comparisons (compare.py) that cancel most run-to-run variance.

Runs are single-process on purpose: N plain runs fit the performance cores
side by side, whereas per-run multiprocessing on top would oversubscribe
them and give back its speedup.

Usage:
  python3 run_replicates.py --prefix baseline --seeds 101 102 103
  python3 run_replicates.py --prefix deltaray --seeds 101 102 103 \
      -- --generations 300          # anything after -- goes to train.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True,
                    help="run names become <prefix>-<seed>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("train_args", nargs="*",
                    help="extra args forwarded to train.py (after --)")
    args = ap.parse_args()

    procs = []
    for seed in args.seeds:
        name = f"{args.prefix}-{seed}"
        log = open(f"runs/{name}.log", "w")
        cmd = [sys.executable, "-u", "train.py", "--run-name", name,
               "--seed", str(seed), *args.train_args]
        print("launch:", " ".join(cmd))
        procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=log), log))

    t0 = time.time()
    failed = []
    for name, proc, log in procs:
        code = proc.wait()
        log.close()
        print(f"{name}: exit {code}  ({(time.time() - t0) / 60:.0f} min elapsed)")
        if code != 0:
            failed.append(name)
    if failed:
        raise SystemExit(f"FAILED runs: {failed} — see runs/<name>.log")

    champs = " ".join(f"runs/{args.prefix}-{s}/champion_best_val.npz"
                      for s in args.seeds)
    print(f"\nall runs done. score them with:\n"
          f"  python3 evaluate.py --suite decision {champs}")


if __name__ == "__main__":
    main()
