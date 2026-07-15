#!/usr/bin/env python3
"""Ensemble driving: one car controlled by the AVERAGE of several champions.

If replicate champions fail on different tracks (idiosyncratic errors), the
average of their control outputs can beat every individual member — the
classic "ensembling decorrelates errors" result, in twenty lines. If they
fail on the SAME tracks (systematic errors), it won't help; either way the
measurement is cheap and the decision suite gives the verdict.

Usage:
  python3 ensemble.py --members runs/flag-101/champion_best_val.npz \
      runs/flag-102/champion_best_val.npz runs/flag-103/champion_best_val.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluate import SUITE_SPECS, build_suite
from racing.brain import forward, make_spec
from racing.persistence import load_genome
from racing.sensors import ray_geometry
from racing.simulation import build_obs, init_ray_history, init_state, step


def run_ensemble_episode(member_genomes: np.ndarray, track, config):
    """One car; each step every member votes and the car takes the mean."""
    spec = make_spec(config.brain, config.sensor)
    rays = ray_geometry(config.sensor)
    state = init_state(1, track)
    init_ray_history(state, track, config, rays)
    while state.t < config.sim.max_steps and state.alive.any():
        obs = build_obs(state, track, config, rays)          # (1, n_in)
        votes = forward(member_genomes,
                        np.repeat(obs, len(member_genomes), axis=0), spec)
        controls = votes.mean(axis=0, keepdims=True).astype(np.float32)
        step(state, track, None, spec, config, rays, controls=controls)
    laps = float(state.progress[0]) / track.total_length
    return laps, bool(state.crashed[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--suite", default="decision")
    args = ap.parse_args()

    import dataclasses
    loaded = [load_genome(p) for p in args.members]
    config = loaded[0][1]

    def world(cfg):  # the master seed is training history, not physics
        return dataclasses.replace(cfg, seed=0).to_json()
    for path, (_, cfg, _) in zip(args.members, loaded):
        if world(cfg) != world(config):
            raise SystemExit(f"{path} has a different config than the first "
                             f"member — ensemble members must share a world")
    members = np.stack([g for g, _, _ in loaded])
    print(f"ensemble of {len(members)} champions on the {args.suite} suite")

    by_d: dict[float, list[tuple[float, bool]]] = {}
    for track, (seed, d) in zip(build_suite(args.suite, config.car.car_radius),
                                SUITE_SPECS[args.suite]):
        laps, crashed = run_ensemble_episode(members, track, config)
        by_d.setdefault(d, []).append((laps, crashed))

    all_laps, all_crash = [], []
    print(f"  {'difficulty':>10}  {'crash rate':>10}  {'mean laps':>9}  {'min laps':>8}")
    for d in sorted(by_d):
        laps = np.array([l for l, _ in by_d[d]])
        crash = np.array([c for _, c in by_d[d]])
        all_laps.extend(laps)
        all_crash.extend(crash)
        print(f"  {d:>10.2f}  {crash.mean():>9.1%}  {laps.mean():>9.2f}  {laps.min():>8.2f}")
    hard = np.concatenate([np.array([l for l, _ in by_d[d]])
                           for d in by_d if d >= 0.9])
    print(f"  OVERALL crash {np.mean(all_crash):.1%}  mean laps {np.mean(all_laps):.2f}")
    print(f"  primary metric (mean laps, d>=0.9): {hard.mean():.3f}")


if __name__ == "__main__":
    main()
