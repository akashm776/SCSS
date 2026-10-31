# Research evidence index

For questions, designs, outcomes and limitations of every study, start with the
[experiment ledger](../docs/experiment-ledger.md). For the overall aim, see
[SCSS research program](../docs/research-program.md), formerly OTCO.

[Paper-style results](../docs/paper/results.md) · [Current progress](../docs/current-progress.md) · [Claim-to-evidence index](../docs/paper/evidence.md)

## Latest completed evidence

| Archive | What it contains |
|---|---|
| [D1 source regimes](clip_source_complete_2026-10-16/comparison.json) | Completed 108-branch comparison; early synthesis advantage, inconsistent side curriculum, loss/R@1 discrepancy; [interpretation](../docs/research-status-2026-10-16-source-complete.md) |
| [D0 geometry](clip_geometry_audit_2026-10/README.md) | Completed 624-forward audit; source labels do not determine generated placement |
| [Initial-pressure calibration](clip_pressure_2026-10/README.md) | Seven-arm comparison; later semi-hard advantage survives in mean, strongest at 7%; all calibrated late arms still harm on average |
| [Difficulty regimes](clip_difficulty_2026-10/README.md) | Five-arm comparison; reduced later synthetic harm, unresolved pressure differences, no horizon-50 aggregate crossover |
| [Weight-state × LR](clip_weight_lr_2026-10/README.md) | Fresh AdamW factorial; early/later utility signs persist under both LR trajectories |
| [Optimizer reset](clip_optimizer_reset_2026-10/README.md) | Four-arm retained/reset × native/sustained time course, exact retained replay |
| [Paired-history comparison](clip_paired_history_comparison_2026-10/README.md) | Frozen plan, CPU comparison and prior-treatment interaction |
| [Gated-history policy](clip_fresh_split_policy_2026-10/README.md) | Preserved fresh-seed policy replication, now completed |
| [Native-history policy](clip_native_history_policy_2026-10/README.md) | Native-only starting states and four continuation policies |
| [Fixed-endpoint pool replay](clip_policy_pool_test_2026-09/README.md) | Reporting-pool robustness on reproduced policy endpoints |
| [Image influence](clip_image_influence_2026-09/REPORT.md) | CPU sensitivity analysis, no additional optimizer updates |

Earlier original CLIP, curriculum, paired-update, predictor, training, diagnostic,
time-course, marginal-utility and policy archives are linked individually from the
[ledger](../docs/experiment-ledger.md). The marginal-utility partial archive is an
exact subset of the completed run, not additional evidence.

Each newer filtered export keeps original manifests/source snapshots, numerical
records and an `export_audit.json`. Omitted tensors/caption-bearing inputs are listed
explicitly. Public archives are not complete Colab recovery/input ZIPs. Original
source snapshots are historical evidence, not a second live test suite.

[Verification commands and scope](../docs/evidence-and-reproduction.md) ·
[Latest scientific interpretation](../docs/paper/results.md)

- [Completed D0 geometry/feasibility audit](clip_geometry_audit_2026-10/README.md): verified recovery, 624 forwards; 166 reused observations are counted once.

## Placement validation: no training utility results

| Archive | Scope |
|---|---|
| [D2-P v1 failure](clip_placement_preflight_failure_2026-10-18/REPORT.md) | Formal first-check failure plus separately identified offline diagnosis |
| [v2a forward diagnostic](clip_placement_v2a_2026-10-31/REPORT.md) | Authenticated first batch; consistent FP32-head contract passes |
| [v2b gradient validation](clip_placement_v2b_2026-10-31/REPORT.md) | Saved VJP metrics and state-preservation audit pass |
| [v2c calibration](clip_placement_v2c_2026-10-31/REPORT.md) | All 24 coefficients pass initial scheduled-pressure verification |

The full-population audit and D2 continuation experiment remain pending. These
PASS records do not establish a utility gain. See [validation scope](../docs/paper/numerical-validation.md).
