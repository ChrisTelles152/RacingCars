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

## Sprint 2 — Smarter cars (heatmap-guided; A/Bs on shared seeds)

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
  frozen test bank, 25000+ reserved, 29000+ generator dev, 30000+ width
  suite, 31000+ trap suite, 40000+ decision suite, 101-103 shared training
  seeds. Frozen suites never change; new capabilities get new suites.
