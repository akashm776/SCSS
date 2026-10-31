# Weight-state x learning-rate experiment A

Status: implemented; GPU experiment **not run**. Experiment ID: `clip_weight_lr_v1`.
This is an exploratory follow-up to the native-history and AdamW-reset studies.
The synthetic construction is unchanged. Difficulty-band selection is a separate,
future experiment and is not included here.

## Question and frozen design

Does the incremental effect of sustained uniform-top-8 supervision depend on the
starting weights, the external LR trajectory, or their interaction, under fresh AdamW?

Cross all three factors within each training seed and continuation stream:

| Factor | Levels |
|---|---|
| Native-only weights | Completed updates 100 and 500 |
| External LR trajectory | Original scheduler positions 100 and 500 |
| Objective | Native only; native plus sustained uniform-top-8 auxiliary |

- Seeds: **271828, 161803, 141421**; streams: **2026100101, 2026100102**.
- Regenerate each seed's 500 native-only trunk updates; keep both snapshots in RAM.
- Every continuation uses fresh AdamW moments **and parameter step counters**.
- Fixed auxiliary coefficient: **0.06005351588542947**. No recalibration or gates.
- Horizon **50**; report offsets **0, 1, 5, 10, 25, 50**.
- **1,500 trunk + 2,400 branch = 3,900 optimizer updates**. Six matched factorial
  sets, 24 weight/LR cells, 48 branches, 288 point records, 48 endpoint caches.
- Same 512-image canonical reporting pool and 4,458-image training population
  as the preceding studies. Reporting examples are excluded from all new training
  and probes, but are historically reused, not a fresh final test.

The first applied LR is the LR stored after the donor's completed update 100 or
500. After each optimizer update the donor scheduler advances once. The original
100-step warmup and 3,850-step cosine horizon are retained; warmup does not restart.
Only group `lr`/`initial_lr` and scheduler state come from the LR donor. Parameter
identity/order, betas, epsilon, decay, clipping and the logit-scale clamp are fixed.
These are the natural early/later schedule segments, not artificially amplified LR
contrasts. Null or weak differences do not establish LR irrelevance generally.

## Matching and exact replay

All eight branches in a seed/stream share the **same processed batches, captions,
and initial RNG**, including across weight ages. Use the historical update-100
continuation indices and caption epoch for both ages, and restore update-100 RNG.
This removes the age-specific continuation data/RNG difference present in the old
studies. It is not literal resumption of the original training loader.

The inherited native-history reference pins the split, historical plans, data roles,
probe construction, trunk audit, and both starting-state fingerprints. Historical
probe batches are constructed only to reproduce initialization/RNG; no probe runs.
The new pinned reference is extracted from the completed optimizer-reset archive,
with original result-file hashes and parent bundle/manifest provenance. Both
early-weight/early-LR branches must exactly replay all six report points, live-state
hashes and update traces from that reset study. They run first; failure stops the set.
Later-weight diagonal branches are **not** claimed to replay old results because
they now use the common early continuation stream and RNG.

Each branch checks fresh AdamW counters (1 through 50), donor scheduler position,
identical native/treated starting states, fixed LR traces within each LR level,
RNG matching across all cells, no-update report equality at fixed weights, source
snapshot immutability, and frozen parameters. Observations and persistence callbacks
must preserve RNG, model buffers/modes, gradients and optimizer/scheduler state.
Protocol, replay references and source manifest are persisted before model updates.

## Prespecified analysis

At each horizon, first compute **within-cell** effects:

```text
D(w, lr) = reporting_loss(sustained | w, lr) - reporting_loss(native | w, lr)
```

Negative helps. Let `EE`, `EL`, `LE`, `LL` denote early/late weights (first letter)
and early/late LR (second letter). The three primary descriptive contrasts at
**horizon 50** are:

```text
weight main effect = ((LE - EE) + (LL - EL)) / 2
LR main effect     = ((EL - EE) + (LL - LE)) / 2
weight x LR        = (LL - LE) - (EL - EE)
```

Average streams within each seed, then seeds equally. Show every seed, all four
absolute treatment effects, and simple effects at each LR/weight level. All other
horizons and mean bidirectional R@1 changes (percentage points) are secondary.
No report-based treatment decisions, threshold fitting, extra seeds, trimming,
significance/equivalence claims, or selection of a favorable horizon.

These contrasts concern the auxiliary's incremental effect, not raw endpoint
performance. The independent training units remain **three seeds**. The six early
diagonal replay branches per objective do not add independent historical evidence.
Starting weights encode native training history; a weight-associated result does
not identify representation geometry as its cause. Fresh AdamW still transforms
gradients adaptively. A fixed coefficient does not match effective gradient pressure.

## Run in Colab

Paste the entire [self-contained cell](../colabs/clip_weight_lr_drive_one_cell.py)
into one **A100** Colab runtime; high RAM is recommended. No old checkpoint ZIP is
needed. Requires the historical **torch 2.11.0+cu128 / torchvision 0.26.0+cu128**
stack (not cu130), other pinned numerical packages, and 15 GiB free runtime disk.
The cell verifies its source bundle, clones the pinned base, applies the overlay,
runs CPU tests, and then starts the bounded GPU experiment.

Results: `MyDrive/OTCO/evaluation_transfer/clip_weight_lr_<timestamp>`.
Logs: `clip_weight_lr_logs`. Wait for **DRIVE_SYNC_COMPLETE**, then download the
whole new results folder and log. Existing manifests for this experiment prevent
blind reruns. No automatic retry/resume. Earlier experiment folders are untouched.
Snapshots stay in RAM; no large checkpoint tensors are written.

`comparison.json` contains the three primary contrasts, cell effects, per-seed
results and timecourses. Each seed/stream's `result.json` retains all cell reports,
traces and audits. `completion.json` is written only after replay/grid/budget checks.
The archived `branch_plans.json` is the historical replay plan; the actually used
plans are `common_branch_plans.json`.

## Local development checks

```bash
python -m scripts.build_clip_weight_lr_cell
python -m pytest -q tests/test_clip_weight_lr.py
```

CPU tests exercise exact fresh-optimizer behavior, LR off-by-one semantics, matching,
replay failure, inert observations, factorial arithmetic and complete toy execution.
They do not constitute a CUDA replay or a completed scientific experiment.
