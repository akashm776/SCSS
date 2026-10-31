# Appendix B. Placement validation

[Contents](../../README.md#contents) · [Current progress](../current-progress.md) · [Evidence index](evidence.md)

D2 proposes fixed-source Wrong, Boundary, and Correct pairwise-margin policies. The global cosine gap is approximately **0.0010687**. These stages test whether the implementation executes that treatment. None provides a continuation-loss or retrieval result.

## B.1 Failed original precision contract

D2-P v1 stopped at its first prescribed Wrong-arm mixed-precision similarity check: 30 of 34 eligible exposures exceeded the margin tolerance. The formal acceptance record is **FAIL**, with only one of 630 scheduled checks completed. This is a demonstrated contract violation, not an operational interruption or evidence that straight-through placement is universally infeasible.

Subsequent offline analysis of authenticated saved tensors found nearly exact ideal FP32 geometry but inconsistent native/auxiliary numerical paths. For 24 of 34 exposures, the frozen similarity- and logit-acceptance intervals did not overlap. Those additional diagnoses are reconstructed evidence, not checks completed by the original run. [Failure report](../../experiment_results/clip_placement_preflight_failure_2026-10-18/REPORT.md).

## B.2 Paired forward diagnostic: v2a

The revised diagnostic changes the scoring/scaling/loss head to consistent FP32 arithmetic, while preserving encoder execution, the original D1 carrier, frozen targets, source identities, gap, and scale-generation value. It reproduces archived v1 observables before interpreting the paired comparison. The native scale remains live; the auxiliary scale is detached.

An initial attempt was **INVALID** because its Python version differed from the required environment. The explicit Python 3.13.16 amendment subsequently **passed** on the authenticated first batch. This does not establish full-population feasibility.

The complete native symmetric loss changes from **3.08009338 to 3.07541132** on the same inputs, a difference of **−0.00468206**. This is an arithmetic change, **not a training improvement**. D1 outcomes therefore remain historical references rather than exact controls for the new head. [v2a report](../../experiment_results/clip_placement_v2a_2026-10-31/REPORT.md).

## B.3 Surrogate backward validation: v2b

The training candidate uses a detached target plus the live D1 carrier's zero-valued straight-through residual. At identical tensors, its prescribed local candidate-to-carrier Jacobian is the identity. This does not imply equal auxiliary-loss gradients across placements or equality after branches diverge.

The recorded audit passes **306 vector–Jacobian-product checks**, with zero recorded relative vector-difference errors, forward reproduction, native scale connectivity, and auxiliary scale disconnection. It validates the specified surrogate path on the audited batch, not agreement with the derivative of the rebuilt constrained generator. [v2b report](../../experiment_results/clip_placement_v2b_2026-10-31/REPORT.md).

## B.4 Calibration: v2c

The calibrated target is the empirical mean raw auxiliary/native gradient-norm ratio across four scheduled batches per seed/state. Each of H-Uniform, Wrong, Boundary, and Correct has its own coefficient, frozen after calibration.

| Quantity | Recorded result |
|---|---:|
| Seed/state cells | 6 |
| Coefficients | 24 |
| Unweighted batches / weighted replays | 24 / 24 |
| Eligible state/exposure instances | 832 |
| Placement arm/exposure checks | 2,496 |
| Empty batches | 0 |
| Verified scheduled mean pressure | 1.996835%–2.001782% |
| Largest relative error from 2% | 0.158268% (allowed 5%) |
| Individual batch pressure | 1.6313%–2.7054% |
| Matching state-restoration records | 48 / 48 |

All batches are treated, so scheduled and conditional means coincide in this run. Larger placement coefficients compensate for smaller unweighted gradients; they do not demonstrate greater utility. Raw-pressure calibration is not optimizer-displacement matching. [v2c report and coefficients](../../experiment_results/clip_placement_v2c_2026-10-31/REPORT.md).

## B.5 Remaining scope and failure semantics

The full-population audit remains pending. Initial acceptance requires all three targets to be feasible for the required exposures at both states. During eventual training, the assigned target must remain valid; undefined unused counterfactual targets are diagnostic and do not by themselves terminate a branch.

Keep geometric infeasibility, precision-contract violation, operational interruption, and invalid provenance distinct. A PASS certifies the checks performed under the recorded contract, not scientific usefulness or 50-update feasibility. A branch failure makes only dependent estimands unavailable; the primary must not become an unplanned survivor average.

The public auditors authenticate retained artifacts and recompute selected numerical summaries. They do not independently recompute GPU gradients. Full-population acceptance and controlled utility comparisons are still required before drawing conclusions about placement.
