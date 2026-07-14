#!/usr/bin/env python3
"""Evaluate champion checkpoints on the frozen held-out TEST bank.

Why this exists (the three-way split): training tracks drive evolution, and
the VALIDATION set drives champion checkpointing 30+ times per run — so the
reported validation score is a max over many draws and is biased upward
(selecting on a metric slowly overfits it). The TEST bank is used for
selection exactly zero times; numbers printed here are the honest ones.

The bank is deterministic: seeds test_seed_base + 1000*difficulty_index + i,
so every run and every checkpoint is graded on the identical tracks. Each
checkpoint drives the physics/sensors embedded in its own .npz.

Usage:
  python3 evaluate.py runs/first-train/champion_best_val.npz
  python3 evaluate.py runs/a/champion_best_val.npz runs/b/champion_best_val.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np

from racing.persistence import load_genome
from racing.simulation import run_episode
from racing.track import make_track

_track_cache: dict[str, list] = {}


def test_bank(config):
    """The frozen test tracks for a config (cached across checkpoints)."""
    tr = config.train
    key = json.dumps(dataclasses.asdict(config.track), sort_keys=True) + \
        f"|{tr.test_seed_base}|{tr.test_per_difficulty}|{tr.test_difficulties}"
    if key not in _track_cache:
        bank = []
        for di, d in enumerate(tr.test_difficulties):
            for i in range(tr.test_per_difficulty):
                seed = tr.test_seed_base + 1000 * di + i
                bank.append(make_track(seed, d, config.track, config.car.car_radius))
        _track_cache[key] = bank
    return _track_cache[key]


def evaluate_checkpoint(path: str) -> dict:
    genome, config, meta = load_genome(path)
    tracks = test_bank(config)
    by_d: dict[float, list[tuple[float, bool]]] = {}
    for track in tracks:
        r = run_episode(genome[None, :], track, config)
        by_d.setdefault(track.difficulty, []).append(
            (float(r.laps[0]), bool(r.crashed[0])))

    print(f"\n=== {path} ===")
    if meta:
        print("  meta:", {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in meta.items()})
    print(f"  {'difficulty':>10}  {'crash rate':>10}  {'mean laps':>9}  "
          f"{'min laps':>8}  (n per difficulty = {config.train.test_per_difficulty})")
    all_laps, all_crash = [], []
    for d in sorted(by_d):
        laps = np.array([l for l, _ in by_d[d]])
        crashes = np.array([c for _, c in by_d[d]])
        all_laps.extend(laps)
        all_crash.extend(crashes)
        print(f"  {d:>10.2f}  {crashes.mean():>9.0%}  {laps.mean():>9.2f}  "
              f"{laps.min():>8.2f}")
    overall = {"test_mean_laps": float(np.mean(all_laps)),
               "test_min_laps": float(np.min(all_laps)),
               "test_crash_rate": float(np.mean(all_crash))}
    print(f"  {'OVERALL':>10}  {overall['test_crash_rate']:>9.0%}  "
          f"{overall['test_mean_laps']:>9.2f}  {overall['test_min_laps']:>8.2f}")
    return overall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", help=".npz champion checkpoints")
    args = ap.parse_args()
    for path in args.checkpoints:
        evaluate_checkpoint(path)


if __name__ == "__main__":
    main()
