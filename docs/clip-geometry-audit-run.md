# Running Experiment D0

Status: implemented and locally tested; the full GPU audit has **not run**.
The [frozen protocol](clip-geometry-audit.md), [configuration](../configs/clip_geometry_audit.json),
[data/state plan](../configs/clip_geometry_audit_plan.json) and
[freeze manifest](../configs/clip_geometry_audit_freeze_manifest.json) retain their
original bytes and preparation-time status. This page records implementation
status without amending that freeze. D1 remains unfrozen.

Paste the entire [self-contained Colab cell](../colabs/clip_geometry_audit_drive_one_cell.py)
into a **high-RAM A100** runtime. The cell requires one A100 with at least 39 GB
VRAM, 20 GiB host RAM and 15 GiB free runtime disk. It requires
**torch 2.11.0+cu128 / torchvision 0.26.0+cu128** and rejects a different CUDA
stack. The remaining numerical packages and source base are pinned. Local CPU
tests run before checkpoint preparation. No archived ZIP is required.

The default cell regenerates three authenticated native trunks: **1,500 native
optimizer updates**, accounted for separately as checkpoint preparation. Every
starting snapshot and trunk trace must match the frozen references. D0 then
performs **624 fixed-state batch forwards and 39,936 query exposures, with zero
audit optimizer updates, backward passes, scheduler steps or virtual steps**.
It performs no auxiliary objective, pressure calibration, reporting forward,
reporting loss, R@1 or D1 continuation.

The primary six initial batches are a view of the extended continuation-data
population; they are not counted twice. C's four calibration batches per seed
are audited separately for geometry and feasibility, without gradients.

The implementation uses a shared FP32 cosine path with deterministic ties for
H/M/C/I and k=1–8. It records ideal FP32 and reference mixed-precision candidates,
checks the conditional I-Synth invariant, and restores model, gradients, modes,
optimizer, scheduler, RNG and precision settings around each batch observation.
Impossible supports remain null with reasons; invalid mixtures preserve valid
source/real statistics. An invariant, identity or numerical-input failure saves
its context and stops the run. There is no fallback support, automatic retry,
resume, k selection or D1 launch.

Outputs are written to `MyDrive/OTCO/evaluation_transfer/clip_geometry_audit_<timestamp>`;
logs go to `clip_geometry_audit_logs`. Wait for **DRIVE_SYNC_COMPLETE** and download
the whole new run folder and log. A partial-sync marker is not a successful audit.
Earlier experiment folders are untouched. No model checkpoints, captions, images
or embeddings are written as D0 measurements; the source overlay includes the
small frozen historical provenance records.

Output contents:

- `REPORT.md`: feasibility, geometry, separability, semantics and cross-state
  eligibility transitions. Its k=8 geometry tables are descriptive, not a D1 choice.
- `summary.json`: every k, state, seed and stream; pooled distributions, quantiles,
  histograms, conditional denominators, coverage and missing reasons. Macro means
  average streams within seed then equally across seeds, and are null if any
  required stratum is empty. Calibration is never duplicated by stream.
- Per-seed stream/calibration folders: 624 closed gzip JSONL shards of individual
  numeric query records, their gzip batch summaries, state-preservation audits,
  processed-batch identity hashes, source IDs and file checksums. Missing support
  and synthetic fields are null, never zero. Failed probes retain context.
- `shards.json`, `completion.json`, native-preparation records, frozen protocol,
  plan, manifest, source overlay and environment provenance. Completion requires
  the full unique batch grid; infeasible treatments do not make a completed
  computational audit a failure.

`all_query_candidate_sizes` lists every k>=2 with 100% C/I availability and both
valid candidate paths across extended and calibration populations at both states.
It does not choose k or imply feasibility after D1 starts changing the model.
Support equality and real-candidate equality are recorded separately: equal
H/I supports can still select different real candidates.

## Optional authenticated checkpoints

The Python runner accepts `--snapshot-dir /path/to/native_states`. Under that
folder, each `seed_<seed>` directory must contain `step_000100.pt`,
`step_000500.pt` and `trunk_audit.json` in the existing native-trunk format.
Restricted deserialization and full snapshot/trunk fingerprints are checked
before use. A missing pair is regenerated; a partial or altered pair fails
without fallback. These are native snapshots, not previous D0 output folders.
The one-cell default does not require or save these large files.

## Rebuild and test

```bash
python -m scripts.build_clip_geometry_audit_cell
python -m pytest -q tests/test_clip_geometry_audit.py
```

Tests cover strict margin/tie definitions, undefined supports, degenerate mixtures,
I-Synth and k=1 invariants, FP32/bf16 paths, restoration on success and failure,
prohibited audit operations, conditional summary arithmetic, unique complete
batch membership, bundle integrity, and a complete toy preparation/audit run.
The toy run also loads authenticated snapshots without training and rejects
modified snapshots. These checks validate implementation, not scientific outcomes
or bitwise replay on the target A100 stack.
