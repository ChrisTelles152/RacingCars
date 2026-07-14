# RacingCars — Neuroevolution Racing Simulator

Teach yourself how machine learning works by evolving neural-network drivers
on procedurally generated race tracks.

A population of identical cars — each with a different tiny neural network as
its brain — drives simultaneously on a random track. The cars that get
furthest seed the next generation (selection, crossover, mutation). Every
generation trains on **fresh, never-seen-before tracks**, so the champion
that emerges can drive tracks it has never encountered.

> Full documentation of how each piece works is written as the final build
> phase — see the module docstrings in `racing/` in the meantime.

## Quick start

```bash
python3 play.py                     # drive a random track yourself (arrow keys)
python3 train.py --run-name demo    # train a population (headless)
python3 watch.py --champion runs/demo/champion_best_val.npz --track-seed 999
python3 -m pytest                   # run the test suite
```

## Layout

- `racing/` — the simulation + ML core (zero pygame; fully headless and seeded)
- `train.py`, `watch.py`, `play.py` — command-line entry points
- `tests/` — pytest suite
- `runs/` — training outputs: metrics CSV + genome checkpoints (gitignored)
