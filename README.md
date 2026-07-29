# RacingCars — Neuroevolution Racing Simulator

Neural-network drivers that learn to race procedurally generated tracks with
**no gradients, no training data, and no human demonstration** — only
selection. Pure Python + NumPy, CPU only, no deep-learning framework.

A population of 512 identical cars — each with a *different* tiny neural
network as its brain — drives simultaneously on a random track. The cars that
get furthest become the parents of the next generation. Repeat for hundreds
of generations, on **fresh, never-repeated tracks**, and out comes a driver
that can race tracks it has never seen.

**📄 [Read the full field report →](https://christelles152.github.io/RacingCars/)**
— how it works, all 15 experiments, and the four times the measurements lied.

### Result

The final driver completes a **frozen 125-track benchmark — including its
hardest tier — without crashing once**, having never seen any of those tracks
during training. It also handles corridors that pinch and widen, hairpin
traps it was never trained on, and carries margin onto low-grip surfaces it
has never encountered.

| Milestone | Crash rate at max difficulty |
|---|---|
| First trained champion | 92% |
| Perception rebuild | 8% |
| Fixed champion selection | 4% |
| Precision side rays | 1.8% |
| + variable-width training | **0%** |

The largest single improvement came from fixing *how champions were selected*,
not from changing how the cars learn. That, and eleven other findings, are in
the [experiment ledger](#the-experiment-ledger-what-was-actually-tried) below —
including the four separate occasions when a measurement turned out to be
lying. `SESSION_LOG.md` has the condensed build record.

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

A car senses the world through **11 ray-distance sensors** (how far to the
wall in 11 directions, dense and long-range toward the front) plus its own
speed — 12 numbers in total. Every timestep its brain must output 2 numbers:
steering and throttle. That's the whole interface. No coordinates, no map, no
track layout: a brain that only ever sees local wall distances has nothing
track-specific to memorize, which is exactly what forces it to learn
*general* driving.

### The brain

Each brain is a tiny multilayer perceptron, `12 → 16 → 2` with tanh
activations — just **242 numbers** (weights and biases). The genome IS the
network: those 242 floats flattened into a vector. The entire population is
one `(512, 242)` matrix, and one generation of "thinking" for all 512
different brains is two `einsum` calls (see `racing/brain.py`).

*(The project started at 7 rays and 178 parameters; the ray fan was widened
and lengthened in the first perception rebuild — see the ledger below. Old
checkpoints still load, because every `.npz` embeds the exact config it was
trained with.)*

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
(`pos (512,2)`, `heading (512,)`, genome matrix `(512,242)`), and one
timestep for everyone is a fixed sequence of numpy array ops:

- **Sensors**: instead of intersecting rays with wall segments, each track is
  rasterized once into a boolean occupancy grid; a ray is 36 samples marched
  through that grid. All 512×11 rays are one fancy-index gather.
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

## How it's actually developed: measure, then build

Past the first working version, every change is an **experiment with an error
bar**, run through a rigor pipeline (see `ROADMAP.md`):

- **Three-way split.** Training tracks evolve the population; a 50-track
  *validation* ladder picks champions (`train.py`); a 300-track *decision*
  suite gates every ship/kill choice (`compare.py`); a frozen 125-track
  *test* bank (`evaluate.py`) is read once per milestone for the honest
  number. Selecting on a set overfits it — so the sets are kept separate.
- **Paired A/Bs.** Each experiment runs both arms on the same seeds
  (`run_replicates.py`), and `compare.py` runs a paired one-sided t-test on
  the per-seed differences — cancelling the run-to-run variance that would
  otherwise drown a real effect.
- **Diagnose before fixing.** `evaluate.py --heatmap` localizes *where* a
  champion fails (corridor width vs corner sharpness) so effort goes to the
  real cause.

Two worked examples from this repo's own history, both counterintuitive:

- **The cheap fix won.** The champion's residual crashes were all on narrow
  tracks, so instead of a smarter brain we just *shortened the side sensor
  rays* (finer distance quantization) — a config one-liner, zero new
  parameters — and the hardest-track crash rate halved.
- **More information made it worse.** Adding closure-rate ("time-to-
  collision") inputs *hurt* on hard tracks. A capacity control (same bigger
  genome, but the extra inputs carry no new information) proved it wasn't the
  parameter count — the velocity signal itself was misaligned with a task
  that needs position precision. More features is not always better, and a
  capacity control is how you tell.

## The experiment ledger (what was actually tried)

Every arm below ran as a 3-seed paired A/B through the gate. Two shipped;
ten died — each kill with a lesson. The flagship (sharp side rays +
variable-width training) crashes on **0% of the frozen test bank** including
maximum difficulty; the first champion generation crashed on 92%.

| Arm | Verdict | The lesson |
|---|---|---|
| Precision side rays | **SHIP** | Sensor resolution must match the tolerance being judged; a config one-liner halved hard-track crashes |
| Variable-width training | **SHIP** | Task-distribution diversity: fixed a 32%-crash world-family hole for a ~3% speed tax; margin-keeping even transferred to ice (21%→12%) unprompted |
| Delta-ray closure inputs | KILL | More information can hurt; the capacity control proved the info (not the params) was the problem |
| Self-adaptive sigma | KILL | Sigma collapse: myopic selection anneals exploration away 10x too fast |
| Physics randomization | KILL | Clean null — the failures were never physics-brittleness |
| Low-grip zones (blind) | KILL | Total ice mastery (21%→0% crash) bought with ~1 lap of *global* caution: invisible hazards price in everywhere |
| Hairpin traps | (no arm) | Measured first: the incumbent already drove traps at 0% crash — no gap, no experiment |
| Recurrent memory | KILL | Memory wasn't the missing ingredient; untargeted capacity again |
| Chicane obstacles | KILL | Real capability gain (96%→76%) but unachieved + a −1.7 lap caution tax; needs perception, not exposure |
| Island populations | KILL | Flat on capability, best-in-program safety (0.3% crash) — diversity preserves robustness, doesn't add skill |
| Champion ensembling | KILL | Same-distribution champions fail identically; nothing to decorrelate |
| ES fine-tuning | KILL | Noisy per-iteration objective = noise-dominated gradient; a converged champion has nothing to polish, only robustness to lose |
| Multi-car training | KILL (solo) | −0.79 laps solo; but its champions sweep the podium when races turn to carnage — robustness vs pace, quantified |
| Dense-fan obstacle fix | KILL (improved) | Angular resolution (11 forward rays vs 5) dented the cone blind spot 97%→64% and *halved* its caution tax — but 80% crash at max difficulty remains. See below. |
| Radar obstacle channel | **CAPABILITY SHIP** (variant) | The right modality beats more resolution: 3 true-bearing channels (290 params) → **7% ± 3% cone crash** on the best seed vs 64% for 6 extra rays (338 params). Not the flagship (pace tax ~1.2 laps), but the designated obstacle-world driver. See below. |
| Radar *reliability* (longer runs) | KILL | Doubling training to 600 generations left the solve rate flat (median 20%→16%). Perception, selection, incentives and time each measured and ruled out. See below. |
| Radar *reliability* (2× capacity) | KILL | hidden 16→32 looked like a breakthrough on 25 tracks (2/5 seeds "solved" at 8%) — on 100 tracks those same champions are 11–13%, *worse* than the 16-unit best at 7%. The measurement, not the method, had produced the win. |

### Follow-up: chasing the obstacle blind spot

The one glaring hole — 97% crash on mid-corridor cones — got a dedicated
two-arm investigation, because the KILL above pointed at *perception* not
*exposure*. The failure was **angular aliasing**: 12 px cones slip between
the flagship's 4–6° forward ray spacing, so a cone at 7°/130 px is
literally invisible (0 rays hit it, verified). The fix mirrors the precision
win exactly — that fixed *distance* resolution; this needed *angular*
resolution: a 17-ray fan with 11 forward rays over ±16° (2–5° spacing).

Two arms separated perception from policy, and the result is a clean,
honest *partial*:

| Config | Cone crash | Normal-driving tax |
|---|---|---|
| Flagship (sparse fan) | 97% | — |
| Dense fan, **no** obstacle training | 98% | none (8.97, ≥ flagship) |
| Sparse fan + obstacle training | 76% | −1.67 laps |
| **Dense fan + obstacle training** | **64%** (48% @0.9, 80% @1.0) | **−0.74 laps** |

Three findings, each worth more than the headline number:
- **Perception was necessary but not sufficient.** The dense fan *alone*
  still crashed 98% — seeing a cone isn't avoiding it; the flagship's
  wall-avoidance policy doesn't transfer to cones without training on them.
- **Angular resolution genuinely helped, once the policy could use it.**
  Dense fan + training beat sparse fan + training (76%→64%), and — the
  satisfying part — **halved the caution tax** (−1.67 → −0.74 laps). Exactly
  the hypothesis: a car that can *see* the hazard brakes for it specifically
  instead of slowing everywhere. Better sight buys targeted avoidance.
- **It's still not solved.** 80% crash at max difficulty. Threading a 12 px
  cone in an ~18 px corridor at speed is near the frontier of what 7–17
  reactive rays and this steering model can do. The next lever is no longer
  more rays — it's a dedicated obstacle-bearing input (radar-style, not
  ray-alignment-dependent) or a learned speed penalty near hazards. The
  dense fan stays a `--variant`, not the flagship default: it's a harmless
  perception upgrade, but normal driving is already at 0% crash, so it isn't
  worth its ~50% extra sensing cost as the default.

**Round 3 — the radar channel (the modality wins).** Acting on that lever:
3 extra inputs reporting the nearest *visible* frontal cone's true distance
and bearing (`[d, sin θ, cos θ]`, line-of-sight checked — the review caught
that without occlusion testing, 62% of reports were through-wall phantoms
from other track folds). Genome 242→290, *smaller* than the dense fan's 338.
Results per seed (obstacle-suite crash @0.9/@1.0, then clean-track crash):

| Seed | Cone crash | Clean-track crash | Pace (clean primary) |
|---|---|---|---|
| radar-101 | **4% / 4%** | **0.0%** | 7.51 |
| radar-102 | 8% / 20% | 0.5% | 6.88 |
| radar-103 | 36% / 64% | 20% | 7.92 |

> **Correction (measured properly later).** Those 25-track figures are not
> trustworthy at this scale. Re-measured on **100 tracks** (SE ~3pp), the
> best radar champion is **7% ± 3%** — the honest headline. The original
> "4%" was 1 crash in 25, a lucky sample; a mid-investigation "correction"
> to 8% was itself slightly pessimistic. Sub-20% comparisons now use the
> `obstacle_big` suite; the 25-track suite cannot resolve them.

The blind-spot arc: **97% → 64% (dense fan) → 7% ± 3% (radar, best seed)**.
Three lessons close the investigation:
- **The right representation beats more resolution** — 3 bearing channels
  with *fewer* parameters did what 6 extra rays could not. The win is
  attributable to information, not capacity, by construction.
- **Discovery varies by seed.** One seed solved it, one mostly, one barely.
  Round 4 below chased that variance and found it was not a search problem
  at all.
- **Not the flagship.** Radar champions pay ~1.2 laps of pace on clean
  tracks (guardrail: KILL for promotion), so `radar` stays the designated
  obstacle-world variant — different world, different tool. Notably,
  radar-101 is the project's first champion with ~zero crashes across BOTH
  worlds.

**Round 4 — chasing reliability, and finding a frontier.** One seed solving
it and one failing looked like a discovery lottery worth fixing. Four
hypotheses were measured, and the first three died before any training ran:

| Hypothesis | Test | Verdict |
|---|---|---|
| Radar is a *dead input* (constant early, weights drift) | Weight norms per champion | **No** — radar weights healthy in every seed; the *worst* seed had the highest radar/ray ratio |
| Good champions existed but *selection* missed them | Score every archived champion of the failed seed | **No** — it never produced one (best 75% crash at any generation) |
| Crashing is *under-punished* by the fitness | Compare crashed vs clean episode fitness | **No** — crashing forfeits the rest of the episode: 1.44 vs 6.86 mean fitness. The 0.02 penalty is negligible, but dying costs ~5.4 laps |
| Training is *cut short* (best champion came last, sigma above floor) | 5 seeds × 600 generations, paired | **No** — solve rate 0/5 at both horizons; median 20%→16%, worst 48%→36% |

The 600-generation test used a free paired design: deterministic
per-generation streams mean a 600-gen run replays its own first 300 exactly
(verified bit-identical), so each run reports what it *would* have shipped
at either horizon — with the counterfactual champion chosen by the run's own
validation rule, never by peeking at the probe.

A fifth hypothesis — *policy capacity* (cornering + width adaptation + cone
dodging is too much for 16 hidden units) — produced the investigation's
sharpest lesson. Doubling to 32 units looked like the breakthrough: 2 of 5
seeds "solved" at 8% on the 25-track probe, something no 16-unit seed had
ever done. **On 100 tracks those same champions measure 11–13%, while the
16-unit champion measures 7%.** The apparent win was entirely the
instrument. Capacity: KILL.

What's left, measured: **83% of residual crashes are genuine cone hits**,
not walls hit while dodging. The cars see the cone, are paid to avoid it,
have time and capacity to search — and still hit it. That is a policy class
at its limit, not a search that needs more luck.

**The meta-lesson, learned the hard way.** A 25-track probe flipped this
investigation's conclusions *twice, in opposite directions* — inflating one
champion (4% for a true 7%) and manufacturing a fake winner (8% for a true
13%). Binomial noise at n=25 is ±6pp, which is the entire size of the
effects being chased. The project had already learned "resolution must match
effect size" from the failure heatmap in Sprint 1; it had to learn it again
here, as the author, twice. Sub-20% comparisons now run on `obstacle_big`
(100 tracks, SE ~3pp).

**Honest bottom line on reliability:** radar moved cone crashes from 97% to
the ~7–20% band on *every* seed — a large, robust improvement. Making every
seed land at the bottom of that band is not achievable by more time, more
capacity, better incentives, or better selection; all four were measured.
The remaining gap belongs to the policy class (a reactive 2-layer MLP), and
the next honest lever would be a different class — not another knob.

## Experiments to try

Each of these is a one-line change in `racing/config.py`, a `--variant` in
`experiments.py`, or a CLI flag; run 3 seeds with `run_replicates.py` and
gate with `compare.py`:

- **Does crossover help?** `crossover_rate=0.0` vs `0.5` vs `1.0`. Crossover
  of NN weights is genuinely contested ("competing conventions": two parents
  can encode the same behavior with permuted hidden neurons, making their
  blend garbage). What does your data say?
- **Selection pressure**: `tournament_k=2` (gentle) vs `7` (greedy). Greedy
  converges faster and gets stuck more — find the crossover point.
- **Population size vs generations**: 128 cars × 600 gens or 1024 × 75?
- **Mutation schedule**: freeze `sigma_decay=1.0` — does it stop improving?
- **Blind the car**: 5 rays instead of 11. Add rear rays. Widen the fan.
  (`--variant baseline` restores the pre-tuning long side rays to compare.)
- **Bigger brain**: `hidden=64`. Better driver, or just slower evolution?
- **No curriculum**: `--start-difficulty 1.0 --pin-difficulty`. Can evolution
  bootstrap on hard tracks directly, or does it need the ramp?
  (`--pin-difficulty` freezes the difficulty; without it the adaptive
  schedule would quietly ease the tracks when the population struggles.)
- **Sanity mode**: `train.py --fixed-track-seed 7` trains on ONE track —
  watch it overfit gloriously, then check the champion on any other seed.

## The open thread: reinforcement learning

**Status: designed, not built.** This is the one direction with real upside
left, and it is deliberately parked rather than abandoned.

Everything here is *neuroevolution*: a population of complete solutions,
scored, selected, mutated. A genome drives ~2000 timesteps and receives
exactly **one** number back — "8.7 laps". It never learns which steering
decision was good; evolution just keeps whole brains that happened to score
well. Reinforcement learning inverts that: a policy-gradient method computes a
learning signal at **every timestep** ("that throttle input, in that
situation, beat expectations — do more of it"), which is roughly 2000× more
information extracted per episode. For scale, the GA here consumed on the
order of 460,000 episodes (512 population × 300 generations × 3 tracks).

Why this codebase is unusually ready for it:

- `run_episode`'s sense → act → step loop is already gym-shaped.
- The simulator already steps 512 independent cars in lockstep — PPO wants
  many parallel environments, and that is normally the fiddly part.
- The entire evaluation pipeline is **optimizer-agnostic**. A PPO policy is
  just a mapping from observation to controls; `evaluate.py`, the decision
  suite, and `compare.py` would score it exactly as they score a genome.

What it would need: a gym-style wrapper, an actor-critic network (PyTorch),
the PPO loop itself, and — the genuinely interesting design task — a
**per-step reward** to replace terminal fitness. That last piece reopens
every reward-hacking question in the table above, now at per-step
granularity; expect to re-derive why monotone progress and the stall rule
exist.

Why it is worth doing beyond the algorithm itself: it is a **decisive test of
this project's final claim**. The remaining obstacle weakness is argued to be
a policy-class limit rather than a search limit. If a completely different
optimizer, given the same sensors, plateaus in the same place, that confirms
the conclusion independently. If it sails past, the conclusion was wrong.
Either result is worth having.

Honest costs: a heavy PyTorch dependency in a deliberately lightweight
project, PPO's notorious hyperparameter sensitivity, and a real chance of
producing a *worse* driver after considerable tuning.

### Other seams left open

- **NEAT** (evolving topology too): `simulation.py` only ever calls
  `forward(genomes, obs)` — swap in a new brain/evolution pair.
- **True multi-track parallelism**: the K tracks per generation are
  embarrassingly parallel (`multiprocessing`, one worker per track).
- **Closing the obstacle gap**: the documented frontier. Not more rays — the
  measured failure is genuine contacts by cars that already see the cone.
