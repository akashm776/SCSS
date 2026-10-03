# Experiment ledger

Status as of **October 3, 2026**. This is the current index for the
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

## 6. Proposed—not implemented or run

The next proposed mechanism study crosses early/later native weights with LR
trajectories under fresh AdamW. A generator family, meta-only utility prediction,
storage/data-efficiency comparisons and external transfer tests follow in the
[research agenda](research-program.md). None is presented here as a result.

## Reading the history responsibly

The [September 24 record](research-status-2026-09-24.md), [defense workbook](research-defense.md),
and [background plan](research-study-plan.md) preserve the earlier reasoning and
OT-specific questions. Original proposal language is not evidence of what worked.
See [verification scope](evidence-and-reproduction.md) for exact reproducibility
boundaries, and the [latest report](research-status-2026-10-03.md) for current claims.
