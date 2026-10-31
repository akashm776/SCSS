# Experiment ledger

Evidence and completion status as of **October 31, 2026**.
[Paper-style report](../README.md) · [Current progress](current-progress.md).
This is the current index for the
[SCSS research program](research-program.md), formerly OTCO. Issued protocols
retain their preparation-time wording; completion status below takes precedence
over old “next experiment” text. Results are not selected for favorable outcomes.

Unless stated otherwise, the recent contrastive work is CLIP/CUB-200. Loss
contrasts are treatment minus the specified control (negative helps); R@1 changes
are **percentage points**, not relative percentages. Different evaluation pools,
loss denominators, training budgets and model families are not directly rankable.

## 1. Construction and original downstream tests

| Study and protocol | Status / question | Outcome and limitation | Evidence |
|---|---|---|---|
| [Legacy experiments](legacy-experiments.md) | Completed; initial OT/synthetic construction work | ResNet-50 + DistilBERT with sigmoid-style objective. Different architecture/objective from CLIP; not a controlled architecture comparison. | [Historical record](legacy-experiments.md) |
| [Original CLIP ablations](clip-experiments.md) | Completed; eight 50-epoch arms | No auxiliary arm beats baseline final canonical mean R@1: baseline 1.968%, uniform-top-8 1.907%, hardest-real 1.942%. All arms have best species accuracy at epoch zero. No OT advantage established. | [Archive](../experiment_results/clip_2026-08-30_to_09-01/README.md) |
| [Staged gradients](clip-gradient-stages.md) | Completed; geometry through native fine-tuning | Per-query synthetic margin proxy improves during native warmup. Hardness is not equivalent to a helpful update. | [Curriculum archive](../experiment_results/clip_curriculum_2026-09/README.md) |
| [Dense warmup / head probes](clip-warmup-readiness.md) | Completed; proposed readiness signal | Weak positive shared-head alignment near update 100 becomes near-zero/negative later. Local proxy, not a causal readiness mechanism. | [Curriculum archive](../experiment_results/clip_curriculum_2026-09/README.md) |
| [Early pulse](clip-early-pulse.md) | Completed; 13 epochs, 100 auxiliary updates | Canonical mean R@1 1.631% vs 1.614% native (+0.017 pp, net two successes); species −0.259 pp. No convincing overall win; do not compare directly with 50-epoch endpoints. | [Protocol/results](clip-early-pulse.md) |

## 2. Realized optimizer utility and frozen predictors

| Study and protocol | Status / design | Outcome and limitation | Evidence |
|---|---|---|---|
| [Paired AdamW updates](clip-paired-updates.md) | Completed; seed 42 | Early treatment helps held-out loss in 16/16 trials, later 0/16. Small effects; trials share a training seed/state. | [Archive](../experiment_results/clip_paired_updates_2026-09/README.md) |
| [Two-seed replication](clip-paired-seed-replication.md) | Completed; seeds 123/456 | Early effects repeat; later effects smaller and mixed across seeds. | [Archive](../experiment_results/clip_paired_seeds_2026-09/README.md) |
| [Intermediate checkpoints](clip-paired-intermediate.md) | Completed; five-state curves | Non-monotonic, seed-dependent utility; no universal switch-off step. | [Archive](../experiment_results/clip_paired_intermediate_2026-09/README.md) |
| [Predictor screening](clip-usefulness-predictors.md) | Completed; development analysis | Alignment 11/15 vs step 9/15; both 9/12 outside ±1e-6 band. Screening, not a frozen test. | [Archive](../experiment_results/clip_usefulness_predictors_2026-09/README.md) |
| [Prospective frozen rules](clip-usefulness-prospective.md) | Completed; new seeds 789/2026/31415, 15 states, 720 branches | Mean seed balanced accuracy 90.28% alignment vs 83.33% step (+6.94 pp); ordinary accuracy 13/15 for both, non-near-zero 12/13 for both. Modest predictive evidence, not a policy gain. | [Archive](../experiment_results/clip_usefulness_prospective_2026-09/README.md) |
| [Evaluation-pool transfer](clip-evaluation-transfer.md) | Completed; replay of same 720 branches | New-pool alignment 15/15 vs timing 11/15. Reused monitored evaluation data; not new training seeds or an untouched test. | [Archive](../experiment_results/clip_evaluation_transfer_2026-09/README.md) |

## 3. Training, horizon and exposure controls

| Study and protocol | Status / design | Outcome and limitation | Evidence |
|---|---|---|---|
| [Gated training](clip-gated-training.md) | Completed; three reused seeds | Alignment minus baseline canonical R@1 +0.00575 pp. Auxiliary doses differ: 400/400/401 alignment vs 76 timing vs 901 always-on. Not a dose-controlled gate win. | [Archive](../experiment_results/clip_gated_training_2026-09/README.md) |
| [Exposure-matched timing](clip-exposure-matched.md) | Completed; matched auxiliary update count | Alignment minus mean random timing R@1 +0.01151 pp, small/mixed seed effects; species −0.00767 pp. Gradient dose, compute and burst pattern not all matched. | [Archive](../experiment_results/clip_exposure_matched_2026-09/README.md) |
| [Saved-checkpoint diagnostic](clip-checkpoint-diagnostic.md) | Completed; 12 states, 36 branches, 1,824 updates; disjoint meta/report halves of reused diagnostic data | Sustained 50-step pressure: early 3/3 help, later 9/9 harm. Frozen gate predicts immediate signs 11/12, sustained 9/12; always-off also 9/12 sustained. | [Archive](../experiment_results/clip_checkpoint_diagnostic_2026-09/README.md) |
| [Continuation time course](clip-continuation-timecourse.md) | Completed; new matched streams, 24 pairs, 2,400 updates | Horizon-50 early 6/6 help, later 18/18 harm; R@1 worsens 22 pairs, ties two. Alignment change alone does not identify marginal utility. | [Archive](../experiment_results/clip_continuation_timecourse_2026-09/README.md) |
| [Marginal utility](clip-marginal-utility.md) | Completed; 12 pairs, 1,536 updates including rolled-back trial steps | Sustained history reduces next-step auxiliary usefulness in all 12 pairs at offsets 10/25/50; primary +0.00001563. Not a retrieval gain or isolated mechanism. | [Complete archive](../experiment_results/clip_marginal_utility_2026-09/README.md) |
| Marginal utility first attempt | **Partial**, six pairs | Exact subset reproduced by completed second run. Retained under failed-run provenance, excluded from additional sample counts. | [Partial/overlap audit](../experiment_results/clip_marginal_utility_partial_2026-09/README.md) |

## 4. Policy evaluation and reporting sensitivity

| Study and protocol | Status / design | Outcome and limitation | Evidence |
|---|---|---|---|
| [Frozen policy test](clip-policy-test.md) | Completed; 12 matched sets, 48 branches, 2,400 updates | Gate minus random timing loss −0.00003434 overall, but sign reverses across reporting partitions/streams. Gate vs native helps six early sets, harms six later sets. | [Archive](../experiment_results/clip_policy_test_2026-09/README.md) |
| [Policy robustness](clip-policy-robustness.md) | Completed CPU analysis; zero updates | Favorable average survives leave-one-seed-out, not leave-one-stream/checkpoint/partition-out. One of 16 reporting-batch deletions flips sign. Post-hoc sensitivity, not independent confirmation. | [Report](../experiment_results/clip_policy_robustness_2026-09/REPORT.md) |
| [Fixed-endpoint pool replay](clip-policy-pool-test.md) | Completed; 48 existing endpoints replayed, 2,400 updates | Full512 gate minus random loss −0.00017561; 30/32 B64 partitions favorable. All seed and stream means favorable, but later gate vs native still harmful. Replay is not new seed replication. | [New public archive](../experiment_results/clip_policy_pool_test_2026-09/README.md) |
| [Image influence](clip-image-influence.md) | Completed CPU analysis; zero updates | Sign survives all 32 fixed-group and 512 single-query deletions; removing five favorable query contributions flips query-only mean. Those five supply 6.19% of gross benefit: cancellation sensitivity, not dominant outliers. No data trimming. | [Report](../experiment_results/clip_image_influence_2026-09/REPORT.md) |

## 5. New seeds, native history and optimizer-state controls

| Study and protocol | Current status / design | Outcome and limitation | Evidence |
|---|---|---|---|
| [Native-history policy](clip-native-history-policy.md) | **Completed**; new seeds 271828/161803/141421, native-only starts 100/500, 3,900 updates | Sustained − native loss −0.00111556 early, +0.00149077 later; six/six signs each. No prior auxiliary exposure required. Gate − random −0.00017777, favorable in 2/3 seeds; not a retrieval win. | [Archive](../experiment_results/clip_native_history_policy_2026-10/README.md) |
| [Preserved gated-history replication](clip-fresh-split-policy.md) | **Completed**; same seeds/data/streams, gated trunks, 3,900 updates | Sustained effect −0.00111556 early, +0.00149492 later. Gate − random −0.00013686, favorable in 2/3 seeds; later gate vs native harms all six pairs. Reporting images newly excluded for these runs, not historically untouched. | [Archive](../experiment_results/clip_fresh_split_policy_2026-10/README.md) |
| [Paired-history comparison](clip-paired-history-comparison.md) | **Completed** CPU analysis; frozen before gated run, after native results; zero extra updates | Primary late history interaction +0.0000041470, mixed seed signs. Exact early matching; no equivalence or representation-causation claim. | [Comparison and frozen plan](../experiment_results/clip_paired_history_comparison_2026-10/README.md) |
| [Optimizer reset](clip-optimizer-reset.md) | **Completed**; native-only starts, retained/reset AdamW × native/sustained, 3,900 updates | Primary late reset interaction +0.00014638 (3/3 seed means positive); late harm persists. Early reset effect −0.00161221, late +0.00163715. Clears moments and counters, not just momentum direction. | [Archive](../experiment_results/clip_optimizer_reset_2026-10/README.md) |

The final four rows reuse the same three seeds and reporting design. Reset-study
retained branches are replay controls, not extra evidence units. Twelve matched
sets per run are three seeds × two ages × two streams, not twelve independent seeds.

## 6. Weight-state × external LR control

[Weight-state x LR experiment A](clip-weight-lr.md) is **completed and verified**.
It crosses native
weights 100/500 with LR trajectories 100/500 and native/sustained objectives under
fresh AdamW, sharing processed batches/captions/RNG across all cells. Budget:
1,500 native trunk plus 2,400 branch updates. Primary analysis: weight and LR main
effects and their interaction on sustained-minus-native loss at horizon 50.

Early weights: −0.00161221 under early LR, −0.00154085 under later LR. Later
weights: +0.00175803 under early LR, +0.00176398 under later LR. All six pairs
share each cell's sign. Weight main effect +0.00333753, LR +0.00003865,
interaction −0.00006540. The sign follows weights under these two modestly
different LR segments; no general LR-irrelevance or geometry-causation claim.
Seeds/reporting pool are reused, and the early diagonal is historical replay.
[Verified archive](../experiment_results/clip_weight_lr_2026-10/README.md) ·
[Per-seed results, retrieval and limits](research-status-2026-10-07.md).

A generator family, meta-only utility prediction,
storage/data-efficiency comparisons and external transfer tests follow in the
[research agenda](research-program.md). None is presented here as a result.

## 7. Difficulty regimes — completed

[Experiment B](clip-difficulty.md) compares native, hard-real, semi-hard-real,
hard synthetic top-8 and semi-hard synthetic ranks 16–23 at weights 100/500.
All ten branches in each seed/stream share early LR, fresh AdamW, continuation
inputs and RNG. Hard/semi supports are disjoint, with actual margins and
training-only gradient-pressure probes recorded. Fixed coefficient is not
pressure matching. Native/hard-synthetic branches must replay Experiment A's
early-LR cells at both weights. Budget: **4,500 optimizer updates**, sixty branches.
Primary late-minus-early synthetic semi-hard-minus-hard loss effect:
**−0.00110136**, negative in all three seed means and six stream contrasts.
Semi-hard synthetic is less harmful later (+0.00041808 versus +0.00175803),
but still harmful in all six pairs. It also has the better early aggregate effect
at horizon 50, so the primary endpoint is not a hard-to-semi-hard crossover.
Shorter secondary horizons do show an aggregate crossover. Real-negative controls
have the opposite interaction (+0.00018804). Semi-hard applies substantially lower
gradient pressure; no pressure-independent geometry explanation or retrieval win.
[Verified archive](../experiment_results/clip_difficulty_2026-10/README.md) ·
[Full result and limits](research-status-2026-10-07-difficulty.md).

## 8. Initial pressure calibration — completed

[Experiment C](clip-pressure.md) freezes hard/semi-hard synthetic construction ×
2%/7% initial mean auxiliary/native gradient-norm targets, native control and two
historical replay arms. Calibration is deterministic, training-only, unweighted
at alpha = 1, with per-batch records; resulting coefficients stay fixed during
continuation. The primary is the mean later semi-minus-hard effect across targets;
construction × pressure, weight-state interaction and absolute native comparisons
are prespecified separately. Budget: **5,700 optimizer updates**, 84 branches.
Calibration membership and protocol files retain their original local checksum
pins. All calibration checks and native/historical B replays passed; 84 endpoint
caches were recomputed. Primary later mean semi-minus-hard loss effect:
**−0.0003773937**, negative in 3/3 seed means and 5/6 stream contrasts. Later
semi-hard wins at 7% in all six streams but at 2% in only three. The construction ×
pressure interaction is **−0.0006161545**, negative in all six. Every calibrated
late arm still harms in aggregate and each seed mean; semi-hard-low helps in only
one individual continuation. Weight-state interactions are weaker and mixed.
Initial calibration is accurate, but later pressure diverges and all branch
updates clip; this does not isolate geometry or establish reliable retrieval gains.
[Verified archive](../experiment_results/clip_pressure_2026-10/README.md) ·
[Full result and limits](research-status-2026-10-08-pressure.md).

## 9. D0 geometry and feasibility audit — completed and verified

The [frozen D0 protocol](clip-geometry-audit.md) completed through an explicitly
verified recovery: 166 saved forwards reused plus 458 new forwards, for 624 total
and 39,936 query exposures. All three seeds and both weight states are present.
Audit updates/backward/virtual steps and reporting evaluations are zero. Native
preparation cost 1,500 updates in recovery plus 500 in the interrupted attempt;
repeated preparation is not additional evidence.

Extended zero-nonpositive rates are 19.19% early and 26.92% later. At k=8, common
C/I eligibility falls from 36.10% to 25.42%, and only 22.47% of exposures are
eligible at both states. No tested k can support an all-query I treatment.
Ideal C uniform mixtures cross to nonpositive margins in 96.44% early / 87.97%
later of feasible queries. The source regimes and generated geometry must be
distinguished; no training utility or curriculum crossover was measured.

All 2,668 inventory files and 118 source members verify. Selected aggregate
statistics were recomputed from raw records; no invalid mixtures or applicable
I-Synth invariant violations were found. The original partial archive remains
unchanged, and its observations are counted once in the completed grid.
D1 was subsequently frozen and implemented; see entry 10 below.

[Full interpretation](research-status-2026-10-09-geometry.md) ·
[Evidence and verification scope](../experiment_results/clip_geometry_audit_2026-10/README.md).

## Reading the history responsibly

The [September 24 record](research-status-2026-09-24.md), [defense workbook](research-defense.md),
and [background plan](research-study-plan.md) preserve the earlier reasoning and
OT-specific questions. Original proposal language is not evidence of what worked.
See [verification scope](evidence-and-reproduction.md) for exact reproducibility
boundaries, and the [current progress](current-progress.md) and [report results](paper/results.md) for current claims.

## 10. D1 common-population source regimes — completed October 16

[Protocol](clip-source-regime.md) · [Colab run instructions](clip-source-regime-run.md).

D0-derived k=2 eligibility is intersected across both states and frozen per
exposure: 10,667/19,200 continuation exposures and 416/768 calibration exposures.
All batches have at least 26 eligible queries. H/M/C/I supports and their Select
representatives are frozen from the starting-state D0 records; embeddings evolve.
Native CLIP uses every query. Nine branches use a common masked auxiliary
objective, with eight separately calibrated coefficients targeting 2% initial
mean raw gradient pressure. Budget: 1,500 native preparation + 5,400 continuation
updates, plus state-preserving diagnostics.

The single primary estimand is the late-minus-early C-Select minus I-Select
reporting-loss contrast at 50 updates. Uniform-Synth comparisons are secondary.
This is conditional on common eligible training exposures, measured on the full
reporting pool. The completed archive contains all 108 branches. All eight early
aggregate loss effects improve on native, and all eight late effects worsen it.
Early Synth beats the specified Select control in all four regimes for each seed
mean. The primary side interaction is +0.000128312, with mixed seed signs; it does
not establish the proposed C/I curriculum. All early aggregate R@1 changes are
negative. [Completed interpretation](research-status-2026-10-16-source-complete.md)
· [Comparison data](../experiment_results/clip_source_complete_2026-10-16/comparison.json).

The following entries preserve the operational history preceding completion.

D1 execution update (2026-10-14): [automatic resume](clip-source-resume.md) is
implemented as a separate source bundle. It preserves the frozen scientific
protocol and original delivery, saves native state every 25 updates and branch
state every 10 updates, reuses completed branches/calibrations, and authenticates
old-run migration. Checkpoint I/O is not a new treatment or independent evidence.

D1 startup repair (reported 2026-10-15): the [replacement launcher](clip-source-resume-bootstrap.md)
installs dependencies before starting a fresh worker. This addresses an optional
chat-template lookup failure after the original worker mixed imported Hub modules
with newly installed package files. The issued source bundle and resume identity
are unchanged; this is an execution repair, not new experimental evidence.

D1 storage amendment (2026-10-15): at the user's request, the
[latest-only launcher](clip-source-latest.md) removes older resume generations
and abandoned uploads while retaining the latest required states and completed
results. It keeps the issued payload/contract compatible and records its storage
policy separately. Training and estimands are unchanged.

D1 lease-read repair (2026-10-15): a missing `resume_lease.json` stopped the
final seed's second stream. The [new operational launcher](clip-source-lease-repair.md)
serializes heartbeat writes and checks and retries transient missing-file reads.
Persistent absence and changed ownership still stop execution. This preserves
latest-only storage, the issued checkpoint identity and scientific operations.

## 11. D2 placement execution contract — validation only

| Stage | Status | Evidence and scope |
|---|---|---|
| Original D2-P v1 | FAIL | First Wrong mixed-precision margin check fails. Later offline analysis diagnoses inconsistent native/auxiliary scoring paths. [Report](../experiment_results/clip_placement_preflight_failure_2026-10-18/REPORT.md). |
| v2a original environment | INVALID | Python-version gate stops before the paired diagnostic; not a scientific treatment failure. |
| v2a, explicit Python 3.13.16 amendment | PASS | Same-input first-batch comparison under consistent FP32 scoring. [Report](../experiment_results/clip_placement_v2a_2026-10-31/REPORT.md). |
| v2b | PASS | 306 recorded candidate-logit VJP checks and scale-path checks. [Report](../experiment_results/clip_placement_v2b_2026-10-31/REPORT.md). |
| v2c | PASS | 24 calibrated coefficients in six seed/state cells; scheduled mean pressure 1.996835%–2.001782%. [Report](../experiment_results/clip_placement_v2c_2026-10-31/REPORT.md). |
| Full-population D2-P v2 | Pending | No acceptance record for all continuation exposures yet. |
| D2 utility comparison | Not launched | No placement training result or retrieval claim. |

These stages committed zero optimizer updates. The saved-artifact auditors verify
recorded evidence; they are not independent GPU-gradient recomputations. The new
FP32 head changes the numerical regime: Native and H-Uniform controls for eventual
D2 training must use that regime, and H-Uniform has been recalibrated in v2c.
[Validation appendix](paper/numerical-validation.md) · [Current progress](current-progress.md).
