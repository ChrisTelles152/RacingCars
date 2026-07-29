#!/usr/bin/env python3
"""Evaluate champion checkpoints on frozen held-out track suites.

Why this exists (the three-way split): training tracks drive evolution, and
the VALIDATION set drives champion checkpointing 30+ times per run — so the
reported validation score is a max over many draws and is biased upward
(selecting on a metric slowly overfits it). Suites here are for measurement.

Two suites, two purposes:
- **decision** (seeds 41000+/45000+, 300 tracks, 200 of them at d=1.0):
  used for ALL keep-or-kill experiment gating. Big enough that a d=1.0
  crash-rate difference of a few points is real signal (binomial SE ~1.9pp).
- **test** (seeds 20000+, 125 tracks): the frozen reporting bank, touched
  once per sprint for the honest headline number — never to choose between
  arms. Spending its integrity on decisions would slowly overfit it too.

Suite GEOMETRY is pinned to a canonical TrackConfig (the code defaults with
any distribution-widening knobs zeroed), NOT the checkpoint's embedded track
config — "seed-frozen" must mean "track-frozen" even after the generator
grows new features. The checkpoint still drives its OWN car/sensor/brain
physics, and collision grids are built for its car radius.

Usage:
  python3 evaluate.py runs/x/champion_best_val.npz                # test bank
  python3 evaluate.py --suite decision runs/x/champion_best_val.npz
  python3 evaluate.py --suite decision runs/a.npz runs/b.npz runs/c.npz
  python3 evaluate.py --heatmap runs/x/champion_best_val.npz      # 6x6 diagnosis
  python3 evaluate.py --suite decision --json out.json runs/*.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np

from racing.config import TrackConfig
from racing.persistence import load_genome
from racing.simulation import run_episode
from racing.track import make_track, make_track_axes

# Suite geometry is pinned here — never taken from a checkpoint.
CANONICAL_TRACK = TrackConfig()

# name -> list of (seed, difficulty). Seed-range registry lives in ROADMAP.md.
SUITE_SPECS: dict[str, list[tuple[int, float]]] = {
    "test": [(20_000 + 1000 * di + i, d)
             for di, d in enumerate((0.3, 0.5, 0.7, 0.9, 1.0))
             for i in range(25)],
    "decision": [(41_000 + 1000 * di + i, d)
                 for di, d in enumerate((0.3, 0.5, 0.7, 0.9))
                 for i in range(25)]
                + [(45_000 + i, 1.0) for i in range(200)],
    # Sprint-3 capability suites: same difficulty labels, richer generators.
    "width": [(30_000 + i, 0.9) for i in range(25)]
             + [(30_500 + i, 1.0) for i in range(25)],
    # Trap seeds are the first 25 in-range seeds that generate AT their
    # difficulty label (31008 backs off at d=0.9 and is skipped — vetted by
    # generability only, never by champion performance, so no leakage).
    "trap": [(s, 0.9) for s in (*range(31_000, 31_008), *range(31_009, 31_026))]
            + [(31_500 + i, 1.0) for i in range(25)],
    "grip": [(32_000 + i, 0.9) for i in range(25)]
            + [(32_500 + i, 1.0) for i in range(25)],
    "obstacle": [(33_000 + i, 0.9) for i in range(25)]
                + [(33_500 + i, 1.0) for i in range(25)],
    # 100 tracks at max difficulty (SE ~3pp). The 25-track suite above
    # cannot resolve single-digit crash rates and flipped this project's
    # conclusions TWICE — once making a champion look better than it was
    # (4% for a true ~7%), once making a whole arm look like a winner
    # (8% for a true 13%). Use this one for any sub-20% comparison.
    "obstacle_big": [(34_000 + i, 1.0) for i in range(100)],
}

# Per-suite generator config (the capability being measured lives here).
SUITE_TRACK: dict[str, TrackConfig] = {
    "test": CANONICAL_TRACK,
    "decision": CANONICAL_TRACK,
    "width": dataclasses.replace(CANONICAL_TRACK, width_profile_amp=0.3),
    "trap": dataclasses.replace(CANONICAL_TRACK, trap_prob=1.0),
    "grip": dataclasses.replace(CANONICAL_TRACK, grip_zones=2),
    "obstacle": dataclasses.replace(CANONICAL_TRACK, obstacles=3),
    "obstacle_big": dataclasses.replace(CANONICAL_TRACK, obstacles=3),
}

HEATMAP_SEED_BASE = 25_000   # heatmap cells draw from this reserved range
HEATMAP_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
HEATMAP_TRACKS_PER_CELL = 10

_track_cache: dict = {}


def build_suite(name: str, car_radius: float) -> list:
    """The frozen tracks of a suite, for a given car radius (cached).

    Every suite track must generate at exactly its labeled difficulty: the
    generator silently backs off difficulty when a seed can't satisfy the
    validity checks, and a benchmark quietly easier than its label would
    corrupt every number derived from it — so a backoff here is an error.
    """
    key = (name, float(car_radius))
    if key not in _track_cache:
        tracks = []
        for seed, d in SUITE_SPECS[name]:
            t = make_track(seed, d, SUITE_TRACK[name], car_radius)
            if abs(t.difficulty - d) > 1e-9:
                raise RuntimeError(
                    f"suite {name!r} seed {seed} generated at difficulty "
                    f"{t.difficulty:.2f} instead of {d:.2f} — the suite "
                    f"definition needs recalibrating, not silent acceptance")
            tracks.append(t)
        _track_cache[key] = tracks
    return _track_cache[key]


def score_checkpoint(path: str, suite: str) -> dict:
    """Raw per-difficulty results for one checkpoint on one suite.

    Suites always score SOLO driving — a multicar-trained checkpoint keeps
    its physics/sensors but races nobody here (heat_size forced to 0).
    """
    genome, config, meta = load_genome(path)
    config = dataclasses.replace(
        config, sim=dataclasses.replace(config.sim, heat_size=0))
    by_d: dict[float, dict[str, list]] = {}
    for track, (seed, d) in zip(build_suite(suite, config.car.car_radius),
                                SUITE_SPECS[suite]):
        r = run_episode(genome[None, :], track, config)
        cell = by_d.setdefault(d, {"laps": [], "crashed": []})
        cell["laps"].append(float(r.laps[0]))
        cell["crashed"].append(bool(r.crashed[0]))

    per_difficulty = {}
    all_laps: list[float] = []
    all_crashed: list[bool] = []
    for d in sorted(by_d):
        laps = np.array(by_d[d]["laps"])
        crashed = np.array(by_d[d]["crashed"])
        per_difficulty[d] = {
            "n": int(laps.size),
            "mean_laps": float(laps.mean()),
            "min_laps": float(laps.min()),
            "crash_rate": float(crashed.mean()),
            "laps": [round(float(l), 4) for l in laps],
        }
        all_laps.extend(laps)
        all_crashed.extend(crashed)
    # The pre-registered primary experiment metric: mean laps on hard tracks.
    hard = np.concatenate([np.array(per_difficulty[d]["laps"])
                           for d in per_difficulty if d >= 0.9])
    return {
        "checkpoint": path,
        "suite": suite,
        "meta": {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in meta.items()},
        "per_difficulty": per_difficulty,
        "overall_mean_laps": float(np.mean(all_laps)),
        "overall_crash_rate": float(np.mean(all_crashed)),
        "primary_hard_mean_laps": float(hard.mean()),
    }


def print_report(result: dict) -> None:
    print(f"\n=== {result['checkpoint']}  [{result['suite']} suite] ===")
    if result["meta"]:
        print("  meta:", result["meta"])
    print(f"  {'difficulty':>10}  {'n':>4}  {'crash rate':>10}  "
          f"{'mean laps':>9}  {'min laps':>8}")
    for d, cell in result["per_difficulty"].items():
        print(f"  {d:>10.2f}  {cell['n']:>4}  {cell['crash_rate']:>9.1%}  "
              f"{cell['mean_laps']:>9.2f}  {cell['min_laps']:>8.2f}")
    print(f"  {'OVERALL':>10}  {'':>4}  {result['overall_crash_rate']:>9.1%}  "
          f"{result['overall_mean_laps']:>9.2f}")
    print(f"  primary metric (mean laps, d>=0.9): "
          f"{result['primary_hard_mean_laps']:.3f}")


def print_group_summary(results: list[dict]) -> None:
    print(f"\n=== GROUP SUMMARY over {len(results)} checkpoints ===")
    prim = np.array([r["primary_hard_mean_laps"] for r in results])
    print(f"  primary (mean laps d>=0.9): {prim.mean():.3f} +- {prim.std(ddof=1):.3f}")
    for d in results[0]["per_difficulty"]:
        cr = np.array([r["per_difficulty"][d]["crash_rate"] for r in results])
        ml = np.array([r["per_difficulty"][d]["mean_laps"] for r in results])
        print(f"  d={d:.2f}: crash {cr.mean():.1%} +- {cr.std(ddof=1):.1%}   "
              f"laps {ml.mean():.2f} +- {ml.std(ddof=1):.2f}")


def run_heatmap(path: str, out_png: str | None = None,
                levels: tuple[float, ...] = HEATMAP_LEVELS,
                per_cell: int = HEATMAP_TRACKS_PER_CELL) -> None:
    """Failure diagnosis: crash rate over decoupled (width, curvature) axes.

    Answers WHERE the champion's crashes come from — narrow corridors, sharp
    corners, or only their conjunction — which decides what to build next.
    Cells sample the *generable* subset of their label (validity checks
    couple the axes by design: wide corridors need room for their corners).
    """
    genome, config, meta = load_genome(path)
    n = len(levels)
    crash = np.full((n, n), np.nan)
    for wi, dw in enumerate(levels):
        for ci, dc in enumerate(levels):
            crashes, done, seed = 0, 0, HEATMAP_SEED_BASE + (wi * n + ci) * 1000
            while done < per_cell:
                seed += 1
                try:
                    track = make_track_axes(seed, dw, dc, CANONICAL_TRACK,
                                            config.car.car_radius)
                except RuntimeError:
                    continue  # ungenerable seed at this cell; try the next
                r = run_episode(genome[None, :], track, config)
                crashes += bool(r.crashed[0])
                done += 1
            crash[wi, ci] = crashes / per_cell
        print(f"  width row d_width={dw:.2f} done")

    print(f"\ncrash-rate heatmap for {path}  ({per_cell} tracks/cell)")
    print("  rows = d_width (corridor narrowness), cols = d_curve (corner sharpness)")
    header = "             " + "  ".join(f"c={c:.2f}" for c in levels)
    print(header)
    for wi, dw in enumerate(levels):
        cells = "  ".join(f"{crash[wi, ci]:>6.0%}" for ci in range(n))
        print(f"  w={dw:.2f}   {cells}")

    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        vmax = max(0.25, float(np.nanmax(crash)))  # don't wash out small rates
        im = ax.imshow(crash, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(n), [f"{c:.2f}" for c in levels])
        ax.set_yticks(range(n), [f"{w:.2f}" for w in levels])
        ax.set_xlabel("d_curve (corner sharpness)")
        ax.set_ylabel("d_width (corridor narrowness)")
        ax.set_title("champion crash rate by difficulty axis")
        for wi in range(n):
            for ci in range(n):
                ax.text(ci, wi, f"{crash[wi, ci]:.0%}", ha="center",
                        va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="crash rate")
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        print(f"saved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", help=".npz champion checkpoints")
    ap.add_argument("--suite", choices=sorted(SUITE_SPECS), default="test")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="also dump raw results as JSON")
    ap.add_argument("--heatmap", action="store_true",
                    help="width-vs-curvature crash diagnosis instead of suites")
    ap.add_argument("--heatmap-levels", default=None,
                    help="comma-separated axis levels (default 0..1 in 0.2 steps); "
                         "zoom in when the champion is too good for the full grid")
    ap.add_argument("--heatmap-per-cell", type=int, default=HEATMAP_TRACKS_PER_CELL,
                    help="tracks per cell — size to the effect: detecting an 8%% "
                         "rate needs ~50+, not 10")
    args = ap.parse_args()

    if args.heatmap:
        levels = (tuple(float(x) for x in args.heatmap_levels.split(","))
                  if args.heatmap_levels else HEATMAP_LEVELS)
        for path in args.checkpoints:
            run_heatmap(path, out_png=path.replace(".npz", "_heatmap.png"),
                        levels=levels, per_cell=args.heatmap_per_cell)
        return

    results = [score_checkpoint(path, args.suite) for path in args.checkpoints]
    for result in results:
        print_report(result)
    if len(results) > 1:
        print_group_summary(results)
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_path}")


if __name__ == "__main__":
    main()
