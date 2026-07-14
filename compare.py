#!/usr/bin/env python3
"""Paired, one-sided comparison of two experiment arms (the ship/kill gate).

Every A/B in this project runs both arms on the SAME master seeds (common
random numbers): replicate i of treatment and control saw identical training
track sequences, so per-seed differences cancel the run-to-run variance that
dominates here. The analysis is then a paired one-sided t-test on those
differences — "is the treatment better than its own paired control?"

Pre-registered protocol (ROADMAP.md): primary metric = mean laps on the
d>=0.9 tracks of the DECISION suite; alpha = 0.05, one-sided. Secondary
(reported, not gated): d=1.0 crash rate on the decision suite. Ship rule
also requires no regression > 0.2 mean laps at any difficulty band.

Usage (checkpoints paired by position, i.e. same seed order in both lists):
  python3 compare.py \
      --control  runs/base-101/champion_best_val.npz runs/base-102/... \
      --treatment runs/delta-101/champion_best_val.npz runs/delta-102/...
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluate import score_checkpoint

# One-sided critical t values at alpha = 0.05 by degrees of freedom.
# (Hard-coded so the project stays scipy-free; df = n_pairs - 1.)
T_CRIT_05 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
             6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}


def paired_t(diffs: np.ndarray) -> tuple[float, float, bool]:
    """(t statistic, critical value, significant?) for one-sided alpha=.05."""
    n = diffs.size
    if n < 2:
        raise SystemExit("need at least 2 pairs for a paired t-test")
    sd = diffs.std(ddof=1)
    if sd == 0.0:  # identical differences — degenerate but decidable
        return float("inf") if diffs.mean() > 0 else 0.0, T_CRIT_05[n - 1], diffs.mean() > 0
    t = diffs.mean() / (sd / np.sqrt(n))
    crit = T_CRIT_05[min(n - 1, 9)]
    return float(t), crit, t > crit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--treatment", nargs="+", required=True)
    ap.add_argument("--suite", default="decision")
    args = ap.parse_args()
    if len(args.control) != len(args.treatment):
        ap.error("control and treatment need the same number of checkpoints "
                 "(paired by position — same seed order)")

    ctrl = [score_checkpoint(p, args.suite) for p in args.control]
    treat = [score_checkpoint(p, args.suite) for p in args.treatment]

    prim_c = np.array([r["primary_hard_mean_laps"] for r in ctrl])
    prim_t = np.array([r["primary_hard_mean_laps"] for r in treat])
    diffs = prim_t - prim_c

    print("PRIMARY metric: mean laps on d>=0.9 decision tracks (paired)")
    for i, (a, b) in enumerate(zip(prim_c, prim_t)):
        print(f"  pair {i}: control {a:.3f}  treatment {b:.3f}  diff {b - a:+.3f}")
    t, crit, significant = paired_t(diffs)
    print(f"  mean diff {diffs.mean():+.3f} +- {diffs.std(ddof=1):.3f}   "
          f"t = {t:.2f} vs t_crit(one-sided .05, df={diffs.size - 1}) = {crit:.2f}")
    print(f"  => {'SIGNIFICANT improvement' if significant else 'not significant'}")

    print("\nSECONDARY: d=1.0 crash rate (decision suite, reported not gated)")
    cr_c = np.array([r["per_difficulty"][1.0]["crash_rate"] for r in ctrl])
    cr_t = np.array([r["per_difficulty"][1.0]["crash_rate"] for r in treat])
    print(f"  control   {cr_c.mean():.1%} +- {cr_c.std(ddof=1):.1%}")
    print(f"  treatment {cr_t.mean():.1%} +- {cr_t.std(ddof=1):.1%}")

    print("\nGUARDRAIL: per-difficulty mean-laps regression check (> 0.2 fails)")
    worst = 0.0
    for d in ctrl[0]["per_difficulty"]:
        ml_c = np.mean([r["per_difficulty"][d]["mean_laps"] for r in ctrl])
        ml_t = np.mean([r["per_difficulty"][d]["mean_laps"] for r in treat])
        drop = ml_c - ml_t
        worst = max(worst, drop)
        flag = "  <-- REGRESSION" if drop > 0.2 else ""
        print(f"  d={d:.2f}: control {ml_c:.2f}  treatment {ml_t:.2f}  "
              f"delta {ml_t - ml_c:+.2f}{flag}")

    verdict = significant and worst <= 0.2
    print(f"\nSHIP VERDICT: {'SHIP' if verdict else 'KILL (or gather more seeds)'}")


if __name__ == "__main__":
    main()
