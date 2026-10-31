# D2-P v2a: paired forward-only FP32 scoring diagnostic

This is a new diagnostic, not a repair of the frozen v1 acceptance record. It
tests only seed 271828, stream 2026100101, native checkpoint 100, continuation
batch 0: the same 64 pairs and 34 eligible exposures as the archived v1 failure.
The gap remains **0.0010687001049518585**. All v1 placement tolerances remain fixed.

## Run

Paste the complete `colabs/clip_placement_v2a_drive_one_cell.py` into an A100 40GB,
high-RAM Colab using the pinned D1 runtime (torch 2.11.0+cu128, torchvision
0.26.0+cu128, Python 3.13.15). The cell includes the authenticated v1 reference;
no upload of the old preflight output is required. It reads the existing D1 folder
and writes a separate `clip_placement_v2a_<bundle hash>` folder on Drive.

The bootstrap installs the smaller pinned dependencies but does not replace
torch/torchvision. A runtime mismatch is INVALID rather than an automatic upgrade.
The cell does not run D2 training, gradient probes, calibration, or optimizer steps.

## Frozen implementation

1. Authenticate the original source dependencies, D1 results/checkpoints, gap
   manifest, archived failure bytes, and exact package/GPU/runtime configuration.
2. Restore checkpoint 100 and its common RNG. Build and authenticate only the
   first scheduled batch, including processed-image/caption identities.
3. Execute one original mixed-precision encoder forward. Retain its original
   native similarity matrix, logits, and single generated scale tensor.
4. Execute the unchanged D1 dense-weight H-Synth carrier once under the original
   autocast context. Execute the unchanged detached FP32 target-family function
   once. Construct each ST candidate once. Both heads share those candidates.
5. Reproduce **every archived placement field exactly**: absolute tolerance 0,
   relative tolerance 0, including geometry, auxiliary loss/partition, scale,
   actual similarity/logit fields, and candidate dtype. Identity fields and
   source hashes must also match exactly. No tolerance adaptation is permitted.
   The archive has no full native matrix or native loss; these newly measured
   observables are not misrepresented as archived reproduction checks.
6. Only after reproduction passes, run the new head: cast live embeddings and
   candidates to FP32, with no renormalization; disable autocast/TF32 and reduced
   precision matmul reductions; use `highest` float32 matmul precision. Compute
   one native matrix and its transpose, FP32 scaling and both loss directions.
7. Reuse the current-run original scale, cast to FP32 without recomputing its
   exponential. Native uses the live tensor; auxiliary uses its detached value
   for both candidate and partition. Auxiliary similarities are not detached.
   The runner uses `no_grad`: this only specifies future graph semantics and does
   **not** validate them. Gradient validation remains a later stage.
8. Evaluate all three arms' geometry, actual similarity, actual logit, sign, and
   adjacent-spacing contracts. Preserve full failure coverage rather than stop
   after the first Wrong margin. Verify shared inputs and full restored state.

Casting cannot recover upstream rounding, and cast-only scores need not equal
the ideal renormalized geometry. Success is empirical. The intervention is the
whole scoring/loss head; it does not identify the contribution of each operation.

## Evidence and statuses

`configs/clip_placement_v2a.json` freezes the specification and hashes of four
unchanged archived files. `placement_v2a_source_manifest.json` pins the dependency
overlay. `run_config.json` records that contract and runtime; `index.json` points
to the immutable paired record under `checks/`.

The paired record contains the full 64x64 similarity/logit matrices, both native
partition arrays, per-row and per-column cross-entropy, both directional means,
and actual symmetric native loss. Differences are stored per element with
minimum, maximum, mean, maximum absolute value and quartiles. The auxiliary
records contain all 34 exposures in all three arms, with actual margins/logit
gaps, ideal margins, partitions, loss inputs and per-exposure contract decisions.
The current shared features/carrier/targets have a digest. Matching archived
observables does not authenticate bitwise equality of previously unsaved features.

Native row/column partitions are `logsumexp` diagnostics computed from the actual
native logits in each head's context. They are not claimed to be captures of
internal intermediates from the fused cross-entropy kernel. The auxiliary
partition and pre-softplus tensors are the actual auxiliary loss inputs.

- **PASS**: reproduction, all three forward-placement layers, and preservation
  passed on this batch. Only subsequent ST gradient validation is permitted.
- **FAIL**: reproduced v1, but the new head violates a frozen placement contract.
- **INVALID**: provenance/configuration or archived-output reproduction differs.
- **INCOMPLETE**: an operational interruption or undiagnosed numerical error
  prevents a valid completed comparison.

`acceptance.json` always says `launch_eligible: false`. It is a v2a receipt, not
full D2-P acceptance. Each invocation also writes its own receipt and attempt
history. An interrupted observation is recomputed from authenticated starting
state; no partial tensor results are merged. Completed checks are reused only
after contract/hash authentication. Old attempts and committed records remain.

Exact reproduction is deliberately stricter than a closeness comparison. If
deterministic replay cannot reproduce the recorded outputs, retain INVALID and
investigate; do not relax the gate within this version.

## Validation and subsequent work

Local CPU tests exercise the original-head fork, cast-only scoring, comparison
gate, full symmetric native-loss bookkeeping, failure coverage, and bundle
integrity. They are not the A100 diagnostic and do not establish its result.

After a real v2a PASS: ST gradient validation, then pressure calibration, then a
full-population D2-P v2 audit. Training requires that later audit. If this head
is adopted, rerun Native and rerun/recalibrate H-Uniform under it. D1 results
remain historical references. No inherited D1 coefficient is assumed to retain
2% raw-gradient pressure.
