#!/usr/bin/env python3
"""Plot learning curves from a training run's metrics.csv.

Top panel: train fitness (best / median) with held-out validation markers —
if the train curve climbs while validation stalls, that's overfitting.
Bottom panel: crash rate and curriculum difficulty, which explain most
"why did fitness dip?" moments (harder tracks arrived).

Usage:
  python3 plot_curves.py runs/demo          # saves runs/demo/curves.png
  python3 plot_curves.py runs/demo --show   # also open a window
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
import numpy as np


def load(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no data yet in {path}")
    cols: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = [row[key] for row in rows]
        if key == "track_seeds":
            continue
        cols[key] = np.array([float(v) if v not in ("", None) else np.nan for v in vals])
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="e.g. runs/demo")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = load(os.path.join(args.run_dir, "metrics.csv"))
    gen = m["gen"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   height_ratios=[2, 1])
    ax1.plot(gen, m["best_fit"], label="best (train)", color="#2b8cbe", lw=1.5)
    ax1.plot(gen, m["median_fit"], label="median (train)", color="#a6bddb", lw=1.2)
    has_val = ~np.isnan(m["val_mean"])
    if has_val.any():
        ax1.plot(gen[has_val], m["val_mean"][has_val], "o-", color="#e34a33",
                 ms=4, lw=1.2, label="champion on held-out tracks")
    ax1.set_ylabel("fitness (laps)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_title(os.path.basename(args.run_dir.rstrip("/")))

    ax2.plot(gen, m["crash_rate"], color="#c05050", lw=1.2, label="crash rate")
    ax2.set_ylabel("crash rate", color="#c05050")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("generation")
    ax2.grid(alpha=0.25)
    ax3 = ax2.twinx()
    ax3.plot(gen, m["difficulty"], color="#777777", lw=1.2, ls="--",
             label="difficulty")
    ax3.set_ylabel("difficulty", color="#777777")
    ax3.set_ylim(0, 1.05)

    fig.tight_layout()
    out = os.path.join(args.run_dir, "curves.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
