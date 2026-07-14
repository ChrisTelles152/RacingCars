# RacingCars — Neuroevolution Racing Simulator

Teach yourself how machine learning works by evolving neural-network drivers
on procedurally generated race tracks.

A population of 512 identical cars — each with a *different* tiny neural
network as its brain — drives simultaneously on a random track. The cars that
get furthest become the parents of the next generation. Repeat for hundreds
of generations, on **fresh, never-repeated tracks**, and out comes a driver
that can race tracks it has never seen.

## Quick start

```bash
python3 play.py                          # drive a random track yourself (arrows)
python3 train.py --run-name demo         # train a population (headless, ~1 h)
python3 watch.py --champion runs/demo/champion_best_val.npz   # watch it drive unseen tracks
python3 evaluate.py runs/demo/champion_best_val.npz  # honest score: frozen test bank
python3 plot_curves.py runs/demo         # learning curves -> runs/demo/curves.png
python3 -m pytest                        # test suite
```

Requirements: Python 3.11+, `numpy`, `pygame` (plus `pytest`, `matplotlib`
for tests/plots) — `pip install -r requirements.txt`.

---

## How the learning works

### The task

A car senses the world through **7 ray-distance sensors** (how far to the
wall in 7 directions) plus its own speed — 8 numbers in total. Every timestep
its brain must output 2 numbers: steering and throttle. That's the whole
interface. No coordinates, no map, no track layout: a brain that only ever
sees local wall distances has nothing track-specific to memorize, which is
exactly what forces it to learn *general* driving.

### The brain

Each brain is a tiny multilayer perceptron, `8 → 16 → 2` with tanh
activations — just **178 numbers** (weights and biases). The genome IS the
network: those 178 floats flattened into a vector. The entire population is
one `(512, 178)` matrix, and one generation of "thinking" for all 512
different brains is two `einsum` calls (see `racing/brain.py`).

### The learning algorithm: a genetic algorithm

Unlike gradient descent (backpropagation), evolution never computes a
derivative. It only needs a *score* per individual:

1. **Evaluate** — all 512 cars drive the same tracks; fitness = laps covered.
2. **Elitism** — the top 8 genomes are copied unchanged. The best driver
   found so far can never be lost to an unlucky mutation.
3. **Selection** — each parent is the best of 3 randomly drawn genomes
   ("tournament selection"). Better drivers parent more children, but weak
   ones occasionally win too, preserving diversity.
4. **Crossover** — half the children mix two parents' genes by coin flip.
   (Contested for neural nets! See "Experiments" below.)
5. **Mutation** — each gene gets Gaussian noise with 10% probability. This is
   the actual search engine. The noise scale σ decays over generations:
   coarse exploration early, fine tuning late.

The whole algorithm is ~60 lines in `racing/evolution.py`, and it never looks
inside a genome — it just recombines rows of a matrix based on scores.

### Fitness design (and reward hacking)

Fitness = **furthest point reached along the track centerline**, in lap
fractions, minus a small crash penalty. Designing this number is where you
meet reward hacking — optimizers exploit *what you wrote*, not what you meant:

| Exploit | Defense |
|---|---|
| Drive backwards over the start line | No reverse gear in the physics; backward progress counts negative |
| Oscillate back and forth across the line | Progress is *monotone max*, and lap crossings are counted with sign |
| Creep at 1 px/s forever | Stagnation rule: no progress for 4 sim-seconds → car is frozen |
| Cut across the infield | Progress is only searched in a small window around the car's last checkpoint — faraway checkpoints aren't candidates (and the wall kills you first) |
| Wiggle in fast circles | Circling gains no *net* centerline progress → stagnation rule |

There is deliberately **no speed bonus**: faster cars simply get further
before the step cap, so speed is rewarded without being gameable.

### Generalization: the reason this isn't just curve fitting

The user-visible magic — a champion driving a track it has never seen —
comes from three mechanisms in `train.py`:

- **Fresh tracks every generation.** A genome is never scored twice on the
  same track, so "memorize the track" is not an available strategy. The
  training objective *is* generalization.
- **Curriculum.** Tracks start wide and gentle (difficulty 0). When the
  *median* car handles them, tracks get narrower and twistier. Hard tracks
  have corners that physically cannot be taken at full speed (max-speed turn
  radius 86 px vs. corner radii down to ~35 px), so braking must be learned.
  Difficulty is sampled from a trailing band so easy tracks stay in the mix.
- **Held-out validation.** 10 fixed tracks (seeds 10000–10009, difficulty
  ladder 0.3 → 1.0) are never trained on. Every 10 generations the top-5
  train genomes are raced on them and the checkpoint keeps the best
  *worst-case* performer (in quarter-lap buckets, mean deciding ties —
  raw argmax over a few noisy tracks is a lottery, the "winner's curse").
- **A frozen TEST bank** (`evaluate.py`, 125 tracks, seeds 20000+). The
  validation set drives checkpointing 30+ times per run, so selecting on it
  slowly overfits it too. The test bank is used for selection exactly zero
  times: those are the honest numbers. This project's own history is the
  cautionary tale — the first champion scored "9.6 laps" on validation but
  crashed on 92% of max-difficulty test tracks.
- **Curriculum overshoot.** Training difficulty runs past 1.0 (to 1.3), so
  the d=1.0 evaluation point sits *inside* the training distribution —
  models are reliable where they interpolate, unreliable where they
  extrapolate. Combined with the long-range forward rays (perception
  horizon > braking distance) and risk-sensitive CVaR fitness, this took the
  max-difficulty test crash rate from 92% to 8%.

### Why it's fast: vectorize across models, not just data

There are no per-car Python objects. The population is a handful of arrays
(`pos (512,2)`, `heading (512,)`, genome matrix `(512,178)`), and one
timestep for everyone is a fixed sequence of numpy array ops:

- **Sensors**: instead of intersecting rays with wall segments, each track is
  rasterized once into a boolean occupancy grid; a ray is 24 samples marched
  through that grid. All 512×7 rays are one fancy-index gather.
- **Collision**: the grid is *inflated by the car's radius* (configuration
  space), so "did the car hit a wall" is a single array lookup at its center.
- **Brains**: two einsums evaluate 512 different networks at once.

Measured: **~1.5 ms per timestep for the whole population** — a full
generation (3 tracks × up to 2000 steps) in a few seconds on a Mac CPU.

---

## Module map

| File | One idea per file |
|---|---|
| `racing/config.py` | Every hyperparameter, grouped in frozen dataclasses |
| `racing/track.py` | Catmull-Rom loop generation, checkpoints, occupancy grids |
| `racing/car.py` | Vectorized kinematic physics |
| `racing/sensors.py` | Grid-marching ray sensors |
| `racing/brain.py` | Genome ↔ weights, batched forward pass |
| `racing/simulation.py` | The episode loop, fitness, kill rules |
| `racing/evolution.py` | The genetic algorithm |
| `racing/persistence.py` | Checkpoints (.npz embeds config) + metrics CSV |
| `train.py` / `watch.py` / `play.py` / `plot_curves.py` | CLIs |
| `viewer.py` | Shared pygame rendering (the core never imports pygame) |

Everything is seeded: the same `--seed` reproduces the same run bit-for-bit,
because track sampling, population init, and mutation each get an independent
RNG stream from one master seed (`np.random.SeedSequence`), and the
simulation itself uses no randomness at all.

## Experiments to try

Each of these is a one-line change in `racing/config.py` (or a CLI flag),
and comparing `plot_curves.py` outputs is the whole experiment:

- **Does crossover help?** `crossover_rate=0.0` vs `0.5` vs `1.0`. Crossover
  of NN weights is genuinely contested ("competing conventions": two parents
  can encode the same behavior with permuted hidden neurons, making their
  blend garbage). What does your data say?
- **Selection pressure**: `tournament_k=2` (gentle) vs `7` (greedy). Greedy
  converges faster and gets stuck more — find the crossover point.
- **Population size vs generations**: 128 cars × 600 gens or 1024 × 75?
- **Mutation schedule**: freeze `sigma_decay=1.0` — does it stop improving?
- **Blind the car**: 3 rays instead of 7. Add rear rays. Widen the fan.
- **Bigger brain**: `hidden=64`. Better driver, or just slower evolution?
- **No curriculum**: `--start-difficulty 1.0 --pin-difficulty`. Can evolution
  bootstrap on hard tracks directly, or does it need the ramp?
  (`--pin-difficulty` freezes the difficulty; without it the adaptive
  schedule would quietly ease the tracks when the population struggles.)
- **Sanity mode**: `train.py --fixed-track-seed 7` trains on ONE track —
  watch it overfit gloriously, then check the champion on any other seed.

## Stretch directions

The seams are already in place:

- **NEAT** (evolving topology too): `simulation.py` only ever calls
  `forward(genomes, obs)` — swap in a new brain/evolution pair.
- **Reinforcement learning**: `run_episode`'s sense → act → step loop is
  gym-shaped; wrap it and train PPO on the same physics, then race the two
  approaches.
- **True multi-track parallelism**: the K tracks per generation are
  embarrassingly parallel (`multiprocessing`, one worker per track).
