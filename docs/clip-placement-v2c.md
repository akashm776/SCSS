# D2-P v2c: scheduled pressure calibration

This stage follows the authenticated v2b PASS. It measures and verifies initial
raw auxiliary gradient pressure, with **zero optimizer or virtual optimizer
updates**. A PASS permits the full-population D2-P v2 audit, not D2 training.

Run `colabs/clip_placement_v2c_drive_one_cell.py` as one Colab cell using the same
A100 40GB/high-RAM environment and Python 3.13.16 build as v2b. The cell installs
the pinned application dependencies, authenticates the read-only D1 inputs and
embedded v2b receipt, and writes to a separate `clip_placement_v2c_<bundle>` Drive
folder. It never modifies D1 or earlier D2-P output folders.

## Frozen measurement

- Seeds: 271828, 161803, 141421; native starting weights: 100 and 500.
- Four original scheduled calibration batches per seed, with the archived
  captions, source identities and D1 eligibility mask. No replacement batches.
- Native loss uses all 64 pairs. Auxiliary loss averages eligible queries only.
- Original bf16 encoders and D1 carrier; unchanged target geometry and global
  gap; exact v2a FP32 scoring/loss head with detached auxiliary scale.
- H-Uniform, Wrong, Boundary and Correct each get a new coefficient. H-Uniform
  uses the same live D1 carrier as the placement arms, without placement.
- Each observational batch restores the checkpoint-100 RNG and selected weight
  state. Each read restores parameters, buffers, modes, gradients, RNG, backend,
  optimizer and scheduler state on success and failure.

For each seed/state/arm, let `c` be the empirical mean of the four scheduled
ratios `||g_aux|| / ||g_native||`. Set `alpha = 0.02 / c`. Empty masks contribute
zero; their cosine is undefined. Nonempty auxiliary and all native norms must be
finite and above 1e-12; `c > 1e-12` and finite positive alpha are required.
There is no coefficient cap, clipping, gap change or adaptive refitting.

Then repeat all four restored forwards and actually differentiate `alpha * L_aux`.
Do not just multiply the saved unweighted norm. Weighted scheduled mean must lie
within the frozen 5% relative tolerance of 0.02. Native norm/loss and placement
observables must reproduce exactly. Record both scheduled and conditional-on-
treatment means, per-batch norms/cosines, coefficients and preservation hashes.
The coefficient is shared across both continuation streams within seed/state/arm.

All three placement targets must satisfy the original initial whole-family
geometry, actual similarity, actual logit and separation contracts on calibration
exposures before gradients are measured. Expected contract violations produce
FAIL with partial evidence. Configuration/provenance mismatches produce INVALID;
operational or unexplained numerical interruptions produce INCOMPLETE. Only six
completed passing cells produce PASS.

## Records and recovery

`run_config.json` pins the design, gap, inputs, parent evidence, environment and
complete source overlay. `index.json` identifies immutable content-addressed
cell records. `acceptance.json` is the latest receipt; `receipts/` and `attempts/`
preserve prior attempts, including interrupted ones. A rerun reuses a completed
cell only with matching contract and content hashes; an interrupted cell restarts
from its authenticated initial state. Counts in the receipt refer to committed
evidence, not cumulative GPU work across interrupted attempts.

Expected completed coverage: six cells, 24 unweighted batch observations, 24
weighted replays, and 24 coefficients. No continuation batches are trained.

## Interpretation

PASS establishes initial scheduled raw-gradient calibration for these batches.
It does not match AdamW displacement, certify later placement feasibility, prove
training utility, or replace the full-population audit. D1 controls remain
historical references; the new H-Uniform coefficients belong to the FP32 regime.

Local CPU tests exercise execution semantics, empty masks, rejection behavior,
weighted autograd and state restoration. They do not substitute for this GPU run.
