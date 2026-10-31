# Appendix C. Claim-to-evidence index

[Contents](../../README.md#contents) · [Results](results.md) · [Reproducibility](reproducibility.md)

This page connects the report's claims to readable analyses and retained numerical evidence. Technical manifests are available inside the archives; the study names below are the reader-facing identifiers.

| Claim or question | Readable analysis | Evidence |
|---|---|---|
| Original long-run CLIP treatments did not improve final canonical R@1 over baseline | [Original CLIP experiments](../clip-experiments.md) | [Original archive](../../experiment_results/clip_2026-08-30_to_09-01/README.md) |
| Early/later utility pattern survives history and optimizer-state controls | [Ledger: controls](../experiment-ledger.md) | [History comparison](../../experiment_results/clip_paired_history_comparison_2026-10/README.md), [optimizer reset](../../experiment_results/clip_optimizer_reset_2026-10/README.md) |
| Tested LR trajectories do not remove the early/later sign pattern | [Weight × LR results](../research-status-2026-10-07.md) | [Archive](../../experiment_results/clip_weight_lr_2026-10/README.md) |
| Relative difficulty effects depend on initial pressure; late aggregate effects remain harmful | [Pressure results](../research-status-2026-10-08-pressure.md) | [Archive](../../experiment_results/clip_pressure_2026-10/README.md) |
| Source-side labels do not determine generated placement | [D0 geometry results](../research-status-2026-10-09-geometry.md) | [Archive](../../experiment_results/clip_geometry_audit_2026-10/README.md) |
| Early synthesis advantage, inconsistent source-side curriculum, and loss/R@1 discrepancy | [Completed D1 analysis](../research-status-2026-10-16-source-complete.md) | [Comparison data](../../experiment_results/clip_source_complete_2026-10-16/comparison.json), [audit](../../experiment_results/clip_source_complete_2026-10-16/audit.json) |
| D2 original precision contract fails | [Failure analysis](../../experiment_results/clip_placement_preflight_failure_2026-10-18/REPORT.md) | [Failure archive](../../experiment_results/clip_placement_preflight_failure_2026-10-18/) |
| FP32-head, surrogate-gradient, and calibration checks pass within their scopes | [Validation appendix](numerical-validation.md) | [v2a](../../experiment_results/clip_placement_v2a_2026-10-31/), [v2b](../../experiment_results/clip_placement_v2b_2026-10-31/), [v2c](../../experiment_results/clip_placement_v2c_2026-10-31/) |

[Figure 1 and detailed tables](detailed-results.md) are generated from the retained D1 comparison data by [the summary script](../../scripts/build_research_summary.py). The script checks aggregate arithmetic against equal-seed means. It does not rerun training or recompute checkpoint metrics.

Earlier partial runs, replay controls, predictors, and policy experiments remain in the [complete ledger](../experiment-ledger.md). Partial and resumed records are not counted as independent replications. Scientific claims should use completed results and their stated populations, rather than preparation-time expectations in historical protocols.
