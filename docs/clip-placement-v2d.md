# Full-population D2 preflight (v2d)

[Current progress](current-progress.md) · [Colab guide](../colabs/README.md)

**Prepared; GPU acceptance pending.** This is the next gate after the completed
forward, gradient, and calibration validations. It performs zero optimizer
updates and never launches D2 training.

## Run

Use [the full-population cell](../colabs/clip_placement_v2d_drive_one_cell.py) in an
A100 40GB, high-RAM Colab runtime with the same Python 3.13.16 build and pinned
PyTorch/CUDA stack as v2c. Paste the entire file into one cell. The completed D1
folder must remain at its existing Drive path. Parent calibration evidence is
embedded in the bundle; no separate upload is needed.

The cell writes a new `clip_placement_v2d_<bundle>` folder on Drive. Rerunning the
same cell resumes authenticated completed checks and records a new attempt.
Preserve the entire output folder, including any failure evidence. The runner
requires the historical base commit plus an authenticated source overlay; run
the issued cell rather than invoking the runner directly from a newer checkout.

## Frozen scope

- Three seeds, two starting states, two continuation streams, 50 planned batches:
  **600 continuation batch/state checks**.
- Four calibration batches per seed at both states: **24 additional checks**.
- All D1 eligibility masks, captions, H source identities, and the global gap
  remain fixed. Repeated exposures keep their multiplicity.
- Every batch is observed at its native starting checkpoint with the frozen
  checkpoint-100 RNG rule. Offsets identify planned exposures, not trained states.
- Original bf16 encoder and carrier construction; unchanged detached FP32
  geometry, straight-through candidate, and the authenticated v2 FP32 head.
- Every nonempty batch checks the full Wrong/Boundary/Correct family: geometry,
  actual similarity/logit margins, signs, and separation. No query filtering,
  clamping, gap adjustment, or precision change is allowed.

The full 64-pair native loss remains active. An empty auxiliary mask produces a
scale-disconnected zero, with undefined gradient cosine.

## Gradient and calibration evidence

Every treated exposure gets the v2b one-hot candidate VJP checks for its source,
carrier, and query feature paths. The native scale derivative and disconnected
auxiliary scale paths are checked on each nonempty batch. These probes use live
encoder outputs but stop at the features. Full encoder parameter-gradient
validation and weighted pressure verification are inherited from the authenticated
v2b/v2c evidence; they are not rerun across 624 batches. Descriptive auxiliary
feature gradients are labeled separately from model-parameter gradients.

All six v2c cells and their 24 coefficients are authenticated and independently
recomputed from their saved norms. No coefficient is refitted. The 24 calibration
forwards must reproduce their archived native loss, placement arrays, and head
precision settings exactly. A mismatch invalidates reuse of those coefficients.

Each observation restores model parameters/buffers, existing gradients, modes,
RNG, backend settings, optimizer and scheduler state. Replaying the head after
VJP diagnostics must give identical native and auxiliary tensors on the same
carrier and targets.

## Acceptance and recovery

`run_config.json` pins the source overlay, environment, parent calibration,
original design, inputs and gap. `frozen_coefficients.json` contains the unchanged
coefficients. `index.json` links immutable per-batch records; `attempts/` and
`receipts/` retain interrupted and resumed histories. Reuse requires identical
contracts and verified content. An interrupted check is restarted from its
starting state, never treated as completed.

- **PASS:** all 624 checks pass, with authenticated upstream gradient/calibration
  evidence. Initial execution coverage is complete.
- **FAIL:** a prescribed numerical or gradient contract violation is demonstrated.
  The runner stops and retains the measurements and exact exposure context.
- **INVALID:** provenance/configuration or exact calibration-forward replay fails.
- **INCOMPLETE:** an operational or unexplained numerical interruption prevents
  completion. This is not a scientific feasibility result.

Only complete PASS makes the initial preflight launch-eligible. This program has
no training path. A separate continuation runner must still enforce assigned-
target runtime feasibility, preserve branch failures, and use new Native and
H-Uniform controls under the FP32 head. Passing initial exposures does not certify
50-update trajectories or training utility.

After downloading the output, authenticate it with:

```bash
python -m scripts.audit_clip_placement_v2d /path/to/output-or-zip
```

Local CPU tests check implementation and rejection/recovery behavior. They do not
substitute for the full A100 acceptance run.
