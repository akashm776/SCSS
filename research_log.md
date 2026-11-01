# Research execution log

The completed study history remains in the [research ledger](docs/experiment-ledger.md).

## November 1, 2026 — Full-population D2 preflight prepared

**Question.** Does the validated FP32 placement implementation satisfy its frozen
initial contract across every planned exposure?

**Current evidence.** v2a passes first-batch forward placement; v2b passes its
straight-through gradient contract; v2c passes all 24 scheduled-pressure
coefficients. These stages establish no continuation utility.

**Gap and executable test.** v2d covers 600 continuation and 24 calibration
batch/state observations. It retains the gap, masks, source identities, carrier,
head and coefficients. It checks every exposure's placement and local candidate
VJPs, authenticates upstream full-model gradient evidence, and requires exact
calibration-forward replay. It has no optimizer-step or training-launch path.

**Validation.** CPU tests cover coverage gating, archived calibration, actual
forward/gradient checks, restoration after failures, empty masks, journal
recovery, and archive corruption. The issued bundle pins its dependencies and
preserves all older source manifests. The full suite passed 806 tests; after adding
a final live-input mutation guard, all 63 placement-focused tests passed. The
packaged historical-base overlay also authenticated the parent evidence and gap.
GPU execution is pending.

**Structural update.** The public current-progress page and Colab guide now point
to the full-population audit. A complete passing GPU acceptance record is the
next required evidence before implementing and launching continuation training.
