# ROADMAP — Phase 3: diagnose, close the gap, enrich the world, add rigor

Where we are (frozen test bank, 125 tracks, seeds 20000+): the Tier-1
champion crashes on 8% of max-difficulty tracks (was 92%), 9.41 mean laps
overall, 0 crashes at d <= 0.9. A 300-generation run takes ~70 min
single-process. Everything below is judged against this baseline.

Goals, in priority order:
1. **Rigor**: every claim gets an error bar; decisions never spend the test
   bank's integrity.
2. **Close the remaining gap at max difficulty.** Definition of done:
   d=1.0 crash rate <= 3% with upper 95% CI <= 6% on the 200-track decision
   suite, across the shared seed list.
3. **Richer task distribution** — "generalizes" should mean more than
   "handles constant-width Catmull-Rom loops".
4. **A second algorithm family (ES)** for an optimizer-vs-optimizer lesson.

## The experiment protocol (pre-registered, applies to every A/B)

- **Shared seeds (common random numbers)**: every arm of every A/B runs the
  same master-seed list `{101, 102, 103}`. Run-to-run variance here is
  dominated by the track-draw stream and init — pairing cancels it.
  Analysis = paired, one-sided t-test at alpha 0.05 on per-seed differences.
- **Primary metric**: mean laps on d >= 0.9 decision-suite tracks
  (continuous — far more statistical power than a small-sample crash rate).
  **Secondary**: d=1.0 crash rate on the 200-track decision suite
  (binomial SE ~1.9pp there, so 8% vs 4% is actually detectable).
- **Decision suite ≠ test bank.** New seed range 40000+ (~200 tracks at
  d=1.0 + a spread below) is used for ALL keep-or-kill gating, ensemble
  decisions, and ES guardrails. The frozen 20000+ bank is touched exactly
  once per sprint, at sprint end, for the reported number — never to choose
  between arms. (Selecting on a metric overfits it; we proved this to
  ourselves in Tier 1.)
- **Ship rule**: significant on the primary metric with no regression
  > 0.2 mean laps on any difficulty band of the decision suite. Exception:
  distribution-widening changes (Sprint 3) gate on their OWN new suite,
  with the constant-width suite as a guardrail only — otherwise the rule
  would kill the exact capability it exists to build.
- **Hold-out hygiene**: generator development and knob-tuning happen on a
  dev seed range (29000+). New frozen suites are only frozen AFTER the
  generator parameters are committed, with an explicit
  `assert realized_difficulty == requested` pass (make_track silently backs
  off difficulty on generation failure — a hard seed must not quietly enter
  a "d=1.0" suite easier than labeled).

## Sprint 1 — Infrastructure + diagnosis (all evaluation-side; no training changes)

**1a. Suite machinery in evaluate.py** *(enabler for everything)*
- New TrackConfig fields (width profile, traps) will default OFF (amp=0,
  trap_prob=0) so old checkpoints round-trip to bit-identical tracks.
- Pin the legacy 20000+ bank to a canonical track config (new knobs zeroed)
  regardless of the checkpoint's embedded config — "seed-frozen" must mean
  "track-frozen" even after the generator grows features.
- `--suite` flag to run any checkpoint on any named suite (decision suite,
  future width/trap suites).
- Build the 40000+ decision suite.

**1b. run_replicates.py + paired re-baseline**
Launch N single-process runs (seed list above) as parallel OS processes;
extend evaluate.py to print mean ± std across checkpoints. Re-baseline the
Tier-1 config on the shared seed list — this is the control arm every
Sprint-2 A/B compares against, so it must exist first.
*Teaches: run-to-run variance — "this run got 8%" vs "this method gets X ± Y".*

**1c. Failure heatmap** (`evaluate.py --heatmap`)
Decompose difficulty for evaluation into (d_width, d_curve): 6x6 grid,
10 tracks/cell, champion crash-rate heatmap. Localizes whether the
remaining 8% comes from narrowness, curvature, or only their conjunction —
and therefore which of Sprint 2/3's items actually matters. **Run before
committing to any of them.**
*Teaches: factorized generalization measurement — difficulty is a manifold,
not a scalar.*

**1d. Metrics additions**: per-track realized difficulties; population
mean/p90 sigma; per-generation crash-location histogram (checkpoint index).

**1e. `--workers` flag for train.py** *(demoted from headline to convenience)*
Multiprocessing over the K tracks helps ONLY the single interactive/flagship
run (~2x). It does NOT compose with replicate runs — 2 arms x 3 seeds x 4
processes oversubscribes the P-cores and the speedup evaporates; replicates
always run single-process. Implementation notes if built: persistent Pool,
`VECLIB_MAXIMUM_THREADS=1` set in the parent env before pool creation
(spawn imports numpy before any initializer runs), workers return
(fitness, crashed, laps, steps, realized_difficulty) ordered by track index,
plus a bit-identity test pool-vs-sequential. Skip if flagship wall time is
acceptable.

Gate: replicated baseline table with error bars; heatmap identifies the
dominant failure axis.

**Sprint-1 status: DONE (2026-07-14).** Results:
- Baseline (3 seeds, decision suite): primary 8.765 ± 0.043 mean laps
  (d ≥ 0.9); d=1.0 crash 4.0% ± 1.7%. Test bank (sprint-end report):
  5.3% ± 6.1% at d=1.0, primary 8.954 ± 0.106.
- Heatmap diagnosis: failures live on the corridor-NARROWNESS axis
  (w=0.70 row clean even at max curvature) → Sprint 2 should weight
  lateral information, not just forward braking cues.
- Found & fixed en route: the 10-track validation ladder (2 hard tracks)
  was a champion lottery — every replicate seed locked in a fragile
  gen-60-90 champion (42-65% true crash) wearing perfect val stats.
  Ladder is now 50 stratified tracks (15 at d=1.0); champions archived
  per round for offline re-selection. Fixing selection alone moved the
  method from 52% ± 12% to 4.0% ± 1.7% d=1.0 crash. Also: heatmap needed
  zooming (10 tracks/cell cannot see an 8% effect) — resolution must
  match effect size.

## Sprint 2 — Smarter cars (heatmap-guided; A/Bs on shared seeds)

**Sprint-2 status: DONE (2026-07-14).** Two perception experiments, 3 paired
seeds each, gated on the decision suite:
- **Precision (short side rays) → SHIPPED, now the flagship default.**
  Heatmap-driven: shortened the lateral rays so their quantization dropped
  4.4 px → 1.9 px, at zero genome/compute cost. Primary +0.190 ± 0.075 mean
  laps (d≥0.9), t=4.38 > 2.92; d=1.0 crash **4.0% → 1.8%**; no regressions.
  Meets the phase goal (≤3% crash). The boring, cheap fix beat the fancy one
  because Sprint 1 measured *where* the failure was first.
- **Delta-rays (closure rates) → KILLED**, and the capacity control made the
  reason precise: deltacap (same 418-gene genome, duplicated zero-info rays)
  was NEUTRAL vs baseline (+0.008, t=0.19), but delta was WORSE than deltacap
  (−0.378; d=1.0 crash 4.3% → 7.0%). So it wasn't the bigger genome being
  hard to evolve — the velocity information *itself* hurt on the tight tracks
  that need position precision (gain=8 likely saturating tanh). A
  task-misaligned feature degrades a policy even at matched capacity — the
  exact lesson the capacity-controlled 3-arm design was built to isolate.
- A view-aliasing bug (closure rates silently zeroed in the hot path) was
  caught by the batched-vs-solo test BEFORE any training — capacity-control
  rigor pays for itself.

Below is the original Sprint-2 plan for reference.

**2a. Delta-ray observations (time-to-collision)** — *if the heatmap
implicates late braking / temporal context*
obs = [rays_t, k * (rays_t − rays_{t−stride}), speed].
- **stride 3–4 steps, not 1**: sensor quantization is ~8.3 px on the long
  rays while one step at v_max moves 10 px — a 1-step delta is a one-quantum
  staircase. Over 3-4 steps the signal is several quanta. Pick k so typical
  deltas land mid-range in tanh (~v_max·stride·dt / ray_range reasoning),
  fixed in advance.
- prev_rays initialized by sensing once in init_state (delta = 0 at t=0),
  NOT zeros (k·rays would saturate tanh with garbage on the first decision).
- prev_rays must be gathered/scattered through **all three obs paths**
  (build_obs, the alive-subset branch in step(), the human-controls path),
  with a regression test asserting full-population == alive-subset
  bit-identity with deltas enabled.
- **Three arms, not two**: control (12 in / 242 genes), delta-rays
  (23 in / 418), and a capacity control (23 in / 418 with the extra inputs
  as duplicated rays — same genome size, zero new information). Delta-rays
  must beat the capacity control, not just the baseline, to claim the
  *information* helped rather than the extra parameters.
*Teaches: partial observability / the Markov property; frame stacking; and
what a proper ablation actually is.*

**2b. Self-adaptive per-genome mutation sigma**
Evolving (P,) sigma array: children inherit parent A's sigma mutated
log-normally (tau ~ 0.2, clipped [0.005, 0.5]), then mutate genes with
their OWN sigma; elites keep theirs. Rationale: the clock-driven decay is
nearly frozen by the time the curriculum reaches d=1.3, whether or not the
population still needs exploration. Persist sigma stats in metrics (1d) and
the champion's sigma in checkpoint meta — otherwise the promised
observable ("sigma spikes after promotions") is invisible post-hoc.
*Teaches: self-adaptation (Rechenberg/Schwefel) — evolving the strategy
parameters themselves.*

## Sprint 3 — Harder, richer world (sequential; gating is cumulative:
each item = "current flagship + item" vs "current flagship", paired seeds)

**3a. Variable corridor width** *(the big one — real scope, not a sketch)*
w(s) = nominal · (1 + amp · f(s)), f = 2-3 random-phase Fourier terms in
arc length (loop-periodic, seeded); amp ramps 0 → ~0.3 with difficulty,
**default 0 (off)**. The scalar half_width is load-bearing in five places
that all become width-aware:
  (a) `_has_pinch` → pairwise threshold `margin · (w_i + w_j)/2` (two wide
      sections can otherwise legally merge);
  (b) `_min_curvature_radius` → per-point `radius_i >= margin · w_i`;
  (c) `occ_coll` inflation per-point; floor-check the narrowest section at
      extreme d (18·0.7 − 6 ≈ 6.6 px is razor thin — calibrate);
  (d) rasterizer → per-point brush radius (stamp in radius buckets to stay
      vectorized);
  (e) viewer start-line + the scalar-width assumptions in test_track /
      test_sensors (named test-migration work, not "tests green").
Calibration sub-task before freezing any suite: measure rejection/backoff
rates across d ∈ [0.9, 1.3] at amp 0.3; re-tune knobs until backoff ~0.
Then freeze the width suite (30000+, ~50 tracks) under the no-backoff
assert. First measurement: the CURRENT champion on it — quantifies the
learned "width is constant" prior.
*Teaches: task-distribution design; how deep one scalar assumption runs.*

**3b. Straight-into-hairpin traps**
Post-process control points (probability ramping to ~0.7 at high d,
**default 0**): straighten 2-3 points, then pull the next inward near the
validity limit. Trap suite (31000+, ~50 tracks) frozen under the same
hygiene. *Teaches: hard-example mining, in the task generator.*

**3c. Per-episode physics randomization**
Scale accel/drag/steer_rate/v_max by U(0.85, 1.15) per training episode —
drawn from a **new 4th SeedSequence stream** (NOT track_rng: consuming the
track stream would desynchronize track sequences between arms and destroy
the paired design). Evaluation always at nominal physics. Assert
v_max·1.15·dt still fits the progress window.
*Teaches: domain randomization — the sim2real workhorse.*

**3d. Validation-ladder extension** *(required before the flagship retrain)*
Once width/traps enter training, the 10-track validation ladder (constant
width!) no longer represents the task — champion selection would stay
blind to the new capabilities. Append variable-width and trap validation
tracks (new never-test seeds, e.g. 11000+).

**Flagship retrain** with everything that survived, on the shared seed
list. This is a *confirmation* run, not a gate. Budget one leave-one-out
ablation run if the integrated config regresses, so a bad interaction among
3a/3b/3c is attributable.

## Sprint 4 — Second algorithm family + polish

**4a. OpenAI-ES fine-tune phase** (racing/es.py + finetune_es.py)
Champion as theta; 128 mirrored perturbation pairs → one (256, G) batched
evaluation on fresh hard tracks → rank-shaped update; sigma 0.02–0.05,
~100 iterations (~10 min). Guardrail on the DECISION suite (not the test
bank): no regression > 0.1 mean laps.
*Teaches: ES as black-box gradient estimation; GA vs ES on one fitness
function.*

**4b. Champion ensembling** (stretch)
Average the replicate champions' control outputs; evaluate on the decision
suite; ship `--ensemble` only if it wins there.
*Teaches: ensembling decorrelated errors, in 20 lines.*

## Deferred (with explicit triggers)

- **Recurrent (Elman) memory brain** — if 2a's delta-rays don't close most
  of the remaining gap (would suggest longer temporal context is needed).
- **Fitness sharing / island populations** — if the new sigma/diversity
  metrics show premature convergence while a difficulty tier is unsolved.
- **Low-grip zones, static obstacles** — after width + traps are digested.
- **Multi-car racing** — endpoint of the environment axis; only after the
  solo problem saturates.

## Budget & bookkeeping

- A 3-seed paired A/B = both arms' runs fit the P-cores simultaneously as
  single-process runs: ~75–80 min wall per experiment. (In-run
  multiprocessing does NOT stack with replicate concurrency — same cores.)
- Every sprint ends with: tests green (including named migration work),
  README experiments table updated with the measured result — negative
  results included — commit + push.
- Seed-range registry: 10000+ validation, 11000+ new validation, 20000+
  frozen test bank, 25000+ heatmap cells, 29000+ generator dev, 30000+ width
  suite, 31000+ trap suite, 41000+/45000+ decision suite, 101-103 shared
  training seeds. Frozen suites never change; new capabilities get new suites.

## Program completion status (2026-07-15)

Sprints 3-4, all deferred items, and the Sprint-2 leftover: DONE. Full
ledger in README.md. Headlines:
- **Flagship = precision rays + variable-width training** (the width-101/
  102/103 runs; no separate retrain needed — the arm IS the config).
  Frozen test bank: **0.0% crash at every difficulty**, primary 8.93 ± 0.05.
  The d=1.0 arc across the program: 92% → 8% → 4% → 1.8% → 0%.
- 2 ships / 10 kills. Gap measurement before building killed two redundant
  arms (traps, and almost obstacles); capacity controls and paired seeds
  attributed every result. Negative results are documented as results.
- Known blind spot: mid-corridor obstacles (97% crash) — the follow-up is
  a perception change (denser forward fan / proximity channel), not more
  training. Second follow-up: sighted grip sensing to get ice mastery
  without the blind-caution tax.
- New tools shipped along the way: race.py (exhibition heats),
  ensemble.py, finetune_es.py, experiments.py variant registry,
  A/B queue automation.

## Follow-up: obstacle blind-spot fix (2026-07-16)

Targeted the program's one glaring hole (97% cone crash). Diagnosis: ANGULAR
aliasing — 12px cones slip between the flagship's 4-6° forward ray gaps (a
cone at 7°/130px is invisible, verified). Fix = angular resolution (17-ray
dense fan, 11 forward rays over ±16°), the analog of precision's distance
fix. Two arms:
- **densefan** (dense rays, no obstacle training): 98% cone crash — perception
  ALONE useless; the flagship's wall-avoidance doesn't transfer to cones.
  Harmless to normal driving (decision primary 8.97 ≥ flagship).
- **densefan_obs** (dense rays + obstacle training): 64% cone crash (48%@0.9,
  80%@1.0) — best yet, vs 76% for sparse-fan+training. And it HALVED the
  caution tax (−1.67 → −0.74 laps on the decision suite): better sight →
  targeted avoidance, not blanket slowdown. Hypothesis confirmed.
Verdict: KILL-but-improved. The blind spot is dented, not closed (80% @d=1.0).
Kept as a `--variant`, not the flagship default (harmless perception upgrade,
but no normal-driving benefit for +50% sensing cost). Documented next levers:
a dedicated obstacle-bearing/proximity input (not ray-alignment-dependent),
or a learned speed penalty near hazards.

## Follow-up round 3: radar channel closes the blind spot (2026-07-16)

3 alignment-independent inputs (nearest visible frontal cone's distance +
bearing, line-of-sight checked after review measured 62% through-wall
phantoms pre-fix). Genome 290 — SMALLER than the dense fan's 338.
Per-seed cone crash: 4%/4% (radar-101!), 8%/20%, 36%/64%. The arc:
97% → 64% (dense fan) → **4%** (radar best seed). Lessons: the right
representation beats more resolution (attributable to information by
construction — fewer params); discovery is an innovation lottery across
seeds; guardrail KILL for flagship promotion (~1.2-lap pace tax on clean
tracks) so `radar` is the designated obstacle-world variant. radar-101 is
the first champion ~crash-free in BOTH worlds (0.0% clean, 4% cones).

## Follow-up round 4: radar reliability — a well-characterized frontier (2026-07-16)

Goal: make radar's cone-avoidance discovery reliable across seeds. FIVE
hypotheses measured; all five ruled out:
1. dead input (weights drift while radar reads constant) — NO: radar weights
   healthy in every seed; the WORST seed had the highest radar/ray ratio.
2. selection missed a good champion — NO: the champion archive shows the
   failed seed never produced one (best 75% at any generation).
3. crashing under-punished — NO: crashing forfeits the episode remainder
   (1.44 vs 6.86 mean fitness); the 0.02 constant is irrelevant.
4. training cut short — NO: 5 seeds x 600 gens, paired via deterministic
   replay of the first 300 (verified bit-identical). Solve rate 0/5 at both
   horizons; median 20%->16%.
5. policy capacity — NO: hidden 16->32 "solved" 2/5 seeds on the 25-track
   probe, but those champions measure 11-13% on 100 tracks vs 7% for the
   16-unit best. The instrument, not the method, produced the win.

Residual failure mode: 83% of crashes are genuine cone hits (not
dodge-induced wall crashes) -> policy-class frontier.

MEASUREMENT LESSON (twice, in opposite directions): the 25-track obstacle
suite has ~6pp binomial noise, the size of the effects being chased. It
inflated one champion (4% for a true 7%) and manufactured a fake winning
arm (8% for a true 13%). New `obstacle_big` suite (100 tracks, seeds
34000+, SE ~3pp) is now required for any sub-20% comparison.

HONEST STATUS: radar takes cone crashes 97% -> ~7-20% on EVERY seed (best
champion 7% +- 3%). Landing every seed at the bottom of that band needs a
different policy class, not another knob.
