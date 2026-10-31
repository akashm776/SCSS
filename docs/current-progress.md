# Current progress

[Report](../README.md) · [Research ledger](experiment-ledger.md) · [Evidence index](paper/evidence.md)

**Updated October 31, 2026.** Completed training evidence ends with D1. Subsequent placement work validates execution; it does not yet measure training utility.

## What we have learned

- Added synthetic-negative pressure can help early reporting loss and harm it late in the tested matched continuations. History, optimizer-reset, and learning-rate controls preserve that pattern.
- D1's two-source synthesis beats its specified real-negative control early across all four source regimes and each seed average. The proposed C-versus-I curriculum is inconsistent.
- Loss and retrieval disagree: all early D1 aggregate R@1 changes are negative. No general training or retrieval improvement is established.
- Source ranking does not determine generated placement. This motivates the planned direct pairwise-margin comparison.

[Main results](paper/results.md) gives the scientific argument; [limitations](paper/limitations.md) states its boundaries.

## Placement study: stage gates

| Stage | Status | What the evidence establishes |
|---|---|---|
| D2-P v1 | **FAIL** | The first prescribed mixed-precision margin check violated its contract. Preserved as failed. |
| v2a: consistent FP32 head | **PASS** | Authenticated first-batch forward comparison satisfies the placement contract. |
| v2b: straight-through gradients | **PASS** | 306 recorded candidate-logit VJP checks satisfy the specified backward rule. |
| v2c: initial pressure | **PASS** | All 24 coefficients across six seed/state cells pass scheduled 2% calibration. |
| Full-population D2-P v2 | **Pending** | Must audit all required continuation exposures under the new numerical regime. |
| D2 continuation training | **Not launched** | No placement-policy utility comparison is available. |

The recorded calibration means range from **1.996835% to 2.001782%**. Individual batches range from **1.6313% to 2.7054%**; the target is an initial four-batch average, not a guarantee about every update or optimizer displacement.

All completed D2 validation stages committed **zero optimizer updates**. Their archived acceptance records and local audits have different scopes from an independent GPU replication. [Validation appendix](paper/numerical-validation.md).

## Next scientific step

The next gate is the **full-population D2-P v2 audit**, retaining the fixed gap, masks, source identities, straight-through convention, and calibrated coefficients. A pass would establish initial execution-contract coverage, not future feasibility or utility. D2 training remains gated; Native and H-Uniform controls must use the same new FP32 head as the placement arms.

The present repository work organizes the report and evidence before that next experiment. Operational launchers and archived protocols remain available through the [Colab guide](../colabs/README.md).
