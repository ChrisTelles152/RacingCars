#!/usr/bin/env python3
"""Train a population of neural-net drivers with a genetic algorithm.

The training loop, one generation at a time:

  1. draw K FRESH random tracks (new seeds every generation — there is never
     a track to memorize; generalizing IS the objective)
  2. score every genome on all K tracks; fitness = mean lap fraction
  3. curriculum: when the *median* car handles the current difficulty,
     make the tracks narrower and twistier
  4. every N generations: test the champion on 10 fixed held-out validation
     tracks it never trains on — the honest generalization score — and
     checkpoint it whenever that score improves
  5. breed the next generation (elitism / tournaments / crossover / mutation)

Reproducibility: one master seed spawns independent RNG streams for
population init, mutation, and track sampling (numpy SeedSequence), and the
simulation itself is RNG-free — the same command reproduces the same run.

Usage:
  python3 train.py --run-name demo --generations 300 --seed 42
  python3 train.py --run-name onetrack --fixed-track-seed 7   # sanity mode
Outputs in runs/<run-name>/: config.json, metrics.csv,
  champion_best_val.npz (best validation score), champion_latest.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import numpy as np

from racing.brain import init_population, make_spec
from racing.config import Config
from racing.evolution import next_generation, sigma_at
from racing.persistence import MetricsLogger, save_genome
from racing.simulation import run_episode
from racing.track import Track, make_track


def aggregate_fitness(per_track: np.ndarray, agg: str, cvar_frac: float) -> np.ndarray:
    """Collapse per-track fitness (K, P) into one score per genome (P,).

    The choice IS part of the reward design: 'mean' happily trades a rare
    crash for average speed; 'min' scores each genome by its worst track
    (maximally risk-averse, noisy at small K); 'cvar' averages the worst
    ceil(cvar_frac * K) tracks — pressure against tail-event crashes while
    keeping some of mean's stability.
    """
    if agg == "mean":
        return per_track.mean(axis=0)
    if agg == "min":
        return per_track.min(axis=0)
    if agg == "cvar":
        worst = max(1, int(np.ceil(cvar_frac * per_track.shape[0])))
        return np.sort(per_track, axis=0)[:worst].mean(axis=0)
    raise ValueError(f"unknown fitness_agg {agg!r} (want mean|min|cvar)")


# Worst-case (min) validation score only outranks mean once it improves by a
# meaningful margin. Early in training every candidate crashes on the hardest
# validation tracks, making raw min pure quantization noise (~0.01-0.04 lap
# fractions) — a strict (min, mean) ordering would let that noise override
# multi-lap differences in mean. Bucketing min in quarter-lap steps makes the
# ordering "min first, but only in increments that mean something".
CHAMPION_MIN_BUCKET = 0.25


def champion_key(val_min: float, val_mean: float) -> tuple[float, float]:
    """Sort key for champion candidates and the best-checkpoint ratchet."""
    return (float(np.floor(val_min / CHAMPION_MIN_BUCKET)), val_mean)


def evaluate(genomes: np.ndarray, tracks: list[Track], config: Config):
    """Aggregated fitness over tracks (P,), plus stats for logging."""
    results = [run_episode(genomes, t, config) for t in tracks]
    per_track = np.stack([r.fitness for r in results])
    fitness = aggregate_fitness(per_track, config.train.fitness_agg,
                                config.train.cvar_frac)
    steps = sum(r.steps_run for r in results)
    alive_rate = float(np.mean([(r.steps_alive == r.steps_run).mean() for r in results]))
    crash_rate = float(np.mean([r.crashed.mean() for r in results]))
    best_laps = float(np.max([r.laps.max() for r in results]))
    return fitness.astype(np.float32), steps, alive_rate, crash_rate, best_laps


def nonneg_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--seed", type=nonneg_int, default=None)
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--tracks-per-gen", type=int, default=None)
    ap.add_argument("--start-difficulty", type=float, default=0.0)
    ap.add_argument("--pin-difficulty", action="store_true",
                    help="freeze difficulty at --start-difficulty (disables "
                         "the adaptive curriculum — for controlled experiments)")
    ap.add_argument("--fixed-track-seed", type=nonneg_int, default=None,
                    help="train on ONE fixed track, no curriculum/validation "
                         "(the 'can evolution learn at all?' sanity mode)")
    ap.add_argument("--fixed-difficulty", type=float, default=0.3,
                    help="difficulty used with --fixed-track-seed")
    ap.add_argument("--fitness-agg", choices=["mean", "min", "cvar"], default=None,
                    help="override how per-track fitness aggregates (config: cvar)")
    args = ap.parse_args()

    config = Config()
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)
    if args.population is not None:
        config = dataclasses.replace(
            config, evo=dataclasses.replace(config.evo, population=args.population))
    if args.tracks_per_gen is not None:
        config = dataclasses.replace(
            config, train=dataclasses.replace(config.train,
                                              tracks_per_generation=args.tracks_per_gen))
    if args.generations is not None:
        config = dataclasses.replace(
            config, train=dataclasses.replace(config.train, generations=args.generations))
    if args.fitness_agg is not None:
        config = dataclasses.replace(
            config, train=dataclasses.replace(config.train, fitness_agg=args.fitness_agg))
    tr = config.train
    if config.evo.population <= config.evo.elite:
        ap.error(f"population ({config.evo.population}) must exceed the elite "
                 f"count ({config.evo.elite}) or there is no room for children")

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    metrics = MetricsLogger(os.path.join(run_dir, "metrics.csv"))

    # Independent RNG streams from one master seed: results stay reproducible
    # and changing e.g. mutation draws can never silently reshuffle the tracks.
    init_ss, mut_ss, track_ss = np.random.SeedSequence(config.seed).spawn(3)
    init_rng = np.random.default_rng(init_ss)
    mut_rng = np.random.default_rng(mut_ss)
    track_rng = np.random.default_rng(track_ss)

    spec = make_spec(config.brain, config.sensor)
    genomes = init_population(config.evo.population, spec, init_rng,
                              scale=config.evo.init_scale)
    print(f"population {config.evo.population} x genome {spec.genome_size} "
          f"({spec.n_in}-{spec.hidden}-{spec.n_out} MLP)")

    fixed_mode = args.fixed_track_seed is not None
    fixed_track = None
    val_tracks: list[Track] = []
    if fixed_mode:
        fixed_track = make_track(args.fixed_track_seed, args.fixed_difficulty,
                                 config.track, config.car.car_radius)
        print(f"fixed-track mode: seed {args.fixed_track_seed} "
              f"difficulty {args.fixed_difficulty}")
    else:
        val_tracks = [make_track(s, d, config.track, config.car.car_radius)
                      for s, d in zip(tr.val_seeds, tr.val_difficulties)]
        for vt, want in zip(val_tracks, tr.val_difficulties):
            if abs(vt.difficulty - want) > 1e-9:  # backoff would skew the metric
                print(f"WARNING: validation seed {vt.seed} generated at "
                      f"difficulty {vt.difficulty:.2f} instead of {want:.2f}")
        print(f"validation set: {len(val_tracks)} held-out tracks "
              f"(difficulties {min(tr.val_difficulties)}-{max(tr.val_difficulties)})")

    difficulty = args.start_difficulty
    streak = 0          # consecutive generations with a competent median
    gens_below_bar = 0  # consecutive generations below the promote threshold
    best_val = (-np.inf, -np.inf)       # champion_key of the saved checkpoint
    best_val_stats = (-np.inf, -np.inf)  # its raw (val_min, val_mean), for logs

    for gen in range(tr.generations):
        t0 = time.perf_counter()
        sigma = sigma_at(gen, config.evo)

        if fixed_mode:
            tracks = [fixed_track]
            seeds = [args.fixed_track_seed]
        else:
            # Fresh seeds each generation, from a range disjoint from the
            # validation seeds by construction.
            seeds = [int(s) for s in
                     track_rng.integers(1 << 20, 1 << 31, tr.tracks_per_generation)]
            lo = max(0.0, difficulty - tr.curriculum_band)
            diffs = track_rng.uniform(lo, max(difficulty, 1e-9), tr.tracks_per_generation)
            tracks = [make_track(s, d, config.track, config.car.car_radius)
                      for s, d in zip(seeds, diffs)]

        fitness, steps, alive_rate, crash_rate, best_laps = evaluate(genomes, tracks, config)
        median_fit = float(np.median(fitness))

        # Curriculum: promote when the median car is competent for a streak
        # of generations; ease off only when it has been BELOW the bar for a
        # long stretch. (Demotion keys on incompetence, not on "no promotion
        # happened" — a thriving population at max difficulty must not be
        # demoted just because there is nothing left to promote to.)
        if not fixed_mode and not args.pin_difficulty:
            if median_fit >= tr.promote_threshold:
                streak += 1
                gens_below_bar = 0
            else:
                streak = 0
                gens_below_bar += 1
            # Promotion runs past 1.0 up to max_difficulty (curriculum
            # overshoot): d=1.0 evaluation should be interpolation, not the
            # edge of the training distribution.
            d_cap = config.track.max_difficulty
            if streak >= tr.promote_streak and difficulty < d_cap:
                difficulty = min(d_cap, difficulty + tr.promote_step)
                streak = 0
                print(f"  >> curriculum promoted to difficulty {difficulty:.2f}")
            elif gens_below_bar >= tr.demote_after and difficulty > 0.0:
                difficulty = max(0.0, difficulty - tr.demote_step)
                gens_below_bar = 0
                print(f"  << population stuck, easing to difficulty {difficulty:.2f}")

        # Held-out validation with ROBUST champion selection: the argmax over
        # K noisy tracks is a lottery ticket (winner's curse — argmax of a
        # noisy estimate is biased up), so race the top-N train genomes on
        # the whole validation set and keep the best WORST-case performer.
        val_mean = val_min = val_gap = ""
        champ_idx = int(fitness.argmax())
        if not fixed_mode and (gen % tr.val_every == 0 or gen == tr.generations - 1):
            n_cand = min(tr.champion_candidates, len(fitness))
            cand_idx = np.argsort(-fitness)[:n_cand]
            cand = genomes[cand_idx]  # (N, G): one batched episode per val track
            val_fits = np.stack([run_episode(cand, vt, config).fitness
                                 for vt in val_tracks])          # (V, N)
            cand_min = val_fits.min(axis=0)
            cand_mean = val_fits.mean(axis=0)
            pick = max(range(n_cand),
                       key=lambda i: champion_key(cand_min[i], cand_mean[i]))
            champ_idx = int(cand_idx[pick])
            val_mean = float(cand_mean[pick])
            val_min = float(cand_min[pick])
            val_gap = float(fitness[champ_idx] - val_mean)
            meta = {"generation": gen, "train_fitness": float(fitness[champ_idx]),
                    "val_mean": val_mean, "val_min": val_min, "difficulty": difficulty}
            save_genome(os.path.join(run_dir, "champion_latest.npz"),
                        genomes[champ_idx], config, meta)
            if champion_key(val_min, val_mean) > best_val:
                best_val = champion_key(val_min, val_mean)
                best_val_stats = (val_min, val_mean)
                save_genome(os.path.join(run_dir, "champion_best_val.npz"),
                            genomes[champ_idx], config, meta)
            print(f"  == validation: picked train-rank {pick + 1}/{n_cand}: "
                  f"min {val_min:.3f} mean {val_mean:.3f} laps "
                  f"(best so far min {best_val_stats[0]:.3f} "
                  f"mean {best_val_stats[1]:.3f})")
        elif fixed_mode and (gen % 10 == 0 or gen == tr.generations - 1):
            save_genome(os.path.join(run_dir, "champion_latest.npz"),
                        genomes[champ_idx], config,
                        {"generation": gen, "train_fitness": float(fitness[champ_idx])})

        wall = time.perf_counter() - t0
        # Log the difficulty the tracks were ACTUALLY generated at (the
        # generator backs off when a seed can't satisfy the validity checks),
        # not merely what the curriculum requested — metrics stay honest.
        realized_d = float(np.mean([t.difficulty for t in tracks]))
        metrics.log(
            gen=gen, difficulty=round(realized_d, 3),
            track_seeds=";".join(map(str, seeds)), sigma=round(sigma, 4),
            best_fit=round(float(fitness.max()), 4),
            mean_fit=round(float(fitness.mean()), 4),
            median_fit=round(median_fit, 4),
            p90_fit=round(float(np.percentile(fitness, 90)), 4),
            best_laps=round(best_laps, 3),
            alive_rate_end=round(alive_rate, 3), crash_rate=round(crash_rate, 3),
            steps_simulated=steps, wall_s=round(wall, 2),
            val_mean=val_mean and round(val_mean, 4),
            val_min=val_min and round(val_min, 4),
            val_gap=val_gap and round(val_gap, 4),
        )
        print(f"gen {gen:4d}  d={difficulty:.2f}  best {fitness.max():6.3f}  "
              f"median {median_fit:6.3f}  crash {crash_rate*100:3.0f}%  "
              f"sigma {sigma:.3f}  {wall:5.1f}s")

        genomes = next_generation(genomes, fitness, config.evo, mut_rng, sigma)

    print(f"done. outputs in {run_dir}/")
    if not fixed_mode:
        print(f"watch the champion on an unseen track:\n"
              f"  python3 watch.py --champion {run_dir}/champion_best_val.npz")


if __name__ == "__main__":
    main()
