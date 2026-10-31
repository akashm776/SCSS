# D2-P v2b: first-batch straight-through gradient validation

This stage follows the authenticated v2a PASS under Python 3.13.16. It performs
one grad-enabled encoder forward on exactly seed 271828, stream 2026100101,
checkpoint 100, batch 0. It does not perform calibration, reporting evaluation,
optimizer steps, virtual steps, or D2 continuation training.

## Inputs and lineage

The unchanged D1 checkpoint, first batch, 34 eligible queries, original dense
carrier, global gap, targets and v2a scoring head are retained. Four original
v2a output files are embedded byte-for-byte under
`configs/clip_placement_v2b_reference`. Their hashes and the parent contract are
frozen in `configs/clip_placement_v2b.json`. The downloaded ZIP is archived with
the independently recomputed forward audit under
`experiment_results/clip_placement_v2a_2026-10-31`.

The v2a record authenticates exact v1 reproduction, full state preservation and
all three forward contracts. Maximum similarity error was 1.00582838e-7; maximum
logit error was 1.01786501e-5. Native loss changed from 3.08009338379 to
3.07541131973. This is a paired arithmetic difference, not a training gain.

## Acceptance order

1. Authenticate the full parent receipt, configuration, index, record, input
   checkpoints, code dependencies, and runtime. Recompute the parent placement
   contracts from saved observations; do not rely only on its PASS label.
2. Restore the original weights and RNG. Execute the original encoder and carrier
   with bf16 autocast; construct the frozen targets and cast-only FP32 head.
3. Require **exact** reproduction of all archived v1 placement observables and
   v2a native/placement observations with gradients enabled. If no-grad versus
   grad-enabled execution changes these values, report INVALID and investigate.
   Do not weaken this gate or proceed to backward probes.
4. For each of the 34 eligible exposures and each of the three arms, differentiate
   that candidate's scalar logit separately. Require:
   - its VJP into the actual live image-feature matrix equals H-Uniform's VJP,
     computed from the same live D1 carrier under the FP32 head;
   - its VJP into the carrier is zero outside that exposure and equals `tau*q`
     on that exposure;
   - its VJP into live query features is zero elsewhere and equals `tau*z` on
     that exposure. Targets must be detached.
   One-hot probes avoid cancellation hidden by an aggregate logit sum. Reference
   gradients must have norm >1e-12. Relative vector error <=1e-5 and cosine
   >=0.99999 are inherited from the original gradient contract.
5. Check the native derivative with respect to the live scale and underlying
   `logit_scale` parameter against the analytic symmetric cross-entropy
   derivative. Tolerance is `1e-6 + 1e-5*abs(expected)`. Require native connectivity.
6. Separately require the auxiliary candidate, partition, and complete loss to
   be disconnected (`None`, not just a small gradient) from both scale tensors.
7. Record full trainable-model gradient norms for Native and H-Uniform and, for
   each placement arm, norms and cosines against those two references. These are
   descriptive; they are not expected to agree. No coefficient is calculated.
8. Verify probes do not mutate parameters, buffers, existing gradients, modes,
   RNG, optimizer/scheduler or backend settings. Restore and authenticate the
   complete saved state on success and failure.

Backward calls run outside autocast; the actual original bf16 carrier graph and
FP32 scoring graph remain intact. No FP32 surrogate carrier replaces the live
path. No finite-difference agreement with the constrained generator is expected.

## Run and outputs

Paste the complete `colabs/clip_placement_v2b_drive_one_cell.py` into the same
high-RAM A100 Colab runtime. It includes the passing forward reference, so it
needs only the existing read-only D1 Drive folder. Output is separate under
`clip_placement_v2b_<bundle hash>`.

The immutable check record contains reproduction gates, per-query VJP metrics,
native/auxiliary scale checks, full-gradient summaries, and state fingerprints.
On a gradient-contract FAIL, partial completed measurements and the precise
violation are retained. Operational/undiagnosed numerical errors are INCOMPLETE;
provenance or forward-reproduction mismatches are INVALID. Attempts are retained
and committed checks are authenticated before reuse.

PASS permits the subsequent pressure-calibration stage only. It does not certify
full-population gradient behavior, future trajectory feasibility, or utility.
`launch_eligible` remains false and D2 training is never launched.
