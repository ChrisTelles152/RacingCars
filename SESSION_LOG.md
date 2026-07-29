# Session log — 2026-07-14 to 07-16

The complete build, in order. Numbers are from the frozen 125-track test bank
(read once per milestone) unless stated; ship/kill verdicts come from paired
one-sided t-tests on shared seeds, gated on the 300-track decision suite.

## Build
Empty directory -> working simulator: procedural Catmull-Rom tracks with
rejection sampling and dual occupancy grids, vectorized kinematic physics,
grid-marching ray sensors (~10x faster than segment intersection), 242-param
MLP brains as a flat (P,G) genome matrix, GA (elitism / tournament / uniform
crossover / annealed Gaussian mutation), fresh-tracks-per-generation
generalization protocol with adaptive curriculum. ~1.5 ms/step for 512 cars.

## Milestones (crash rate at maximum difficulty)
| Stage | Crash | What changed |
|---|---|---|
| First champion | 92% | validation looked great (9.6 laps); test bank disagreed |
| Perception rebuild | 8% | perception horizon > braking distance; curriculum overshoot to d=1.3; CVaR fitness |
| Fixed selection | 4% | validation ladder 10 -> 50 tracks; the single biggest win, no learning change |
| Precision side rays | 1.8% | heatmap said narrowness; shorter side rays = 4.4 -> 1.9 px resolution |
| + width training | 0% | variable-width corridors; also transferred to unseen ice |

## Experiments: 3 ships, 12 kills, 1 cancelled by measurement
Ships: precision side rays; variable-width training; bearing-based obstacle
sensing (radar, 290 params — beat a 338-param dense-ray arm, so the win is
attributable to representation, not capacity).
Kills: closure-rate inputs, self-adaptive sigma, physics randomization,
low-grip training, recurrent memory, obstacle-only training, island
populations, ensembling, ES fine-tuning, multi-car training, doubled training
time, doubled capacity.
Cancelled: hairpin traps — incumbent already at 0% crash, no gap to close.

## Corrections (measurements that lied)
1. Failure heatmap read all zeros — 10 tracks/cell cannot see an 8% effect.
2. Published "4%" was 1/25; same genome scores 3/25 fresh. True ~7% +- 3%.
3. Doubled capacity "solved" 2/5 seeds at 8% on 25 tracks; on 100 tracks the
   same champions are 11-13%, worse than the smaller net's 7%.
4. "Crashing is under-punished" theory refuted before implementation:
   crashing forfeits the episode remainder (1.44 vs 6.86 mean fitness).
Policy: sub-20% comparisons require obstacle_big (100 tracks, SE ~3pp).

## Bugs caught by adversarial review (3 of 4 mid-flight)
Champion-ratchet keyed on noise (live run decayed 4.77 -> 0.23 laps before it
was killed); numpy view aliasing zeroing a new input channel in the fast path
only; obstacle sensing through walls (62% phantom readings); infinite
recursion in track difficulty back-off + non-atomic checkpoint writes.

## Final state
Flagship = precision rays + variable-width training (runs/width-101..103).
0% crash on the frozen bank at every difficulty; 1.3% on variable-width;
0% on traps never trained on; 12% on unseen ice. Documented frontier:
mid-corridor obstacles at ~7-20% (83% are genuine contacts by cars that see
the obstacle) — a policy-class limit, not a search limit.

## Open thread
Train a policy-gradient (PPO) agent on the identical fitness function and race
it against the GA champion. run_episode is already gym-shaped and the whole
evaluation pipeline is optimizer-agnostic. It doubles as a decisive test of
the frontier claim: an independent optimizer plateauing in the same place
confirms it; solving cones refutes it.
