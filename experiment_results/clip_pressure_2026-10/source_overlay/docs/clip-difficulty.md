# Difficulty-regime experiment B

Status: implemented; GPU experiment **not run**. Experiment ID `clip_difficulty_v1`.
This is the next exploratory arm after the [completed weight/LR experiment](research-status-2026-10-07.md).
Its issued source bundle and results remain unchanged.

## Question and frozen design

Does the relative utility of extreme-hard versus a moderate rank band change
between native weight checkpoints 100 and 500 under the same external LR?
“Hard early, semi-hard later” is a directional hypothesis, never a switching rule.

| Factor | Levels |
|---|---|
| Native-only weights | Completed updates 100 / 500 |
| External LR | Update-100 schedule segment for every arm and both weights |
| Intervention | Native; hard synthetic; hard real; semi-hard real; semi-hard synthetic |

Seeds **271828, 161803, 141421**, streams **2026100101, 2026100102**. Fresh AdamW
moments and step counters in all branches. Same processed continuation batches,
captions and initial RNG across the ten branches within a seed/stream. The common
stream is Experiment A's early continuation, not literal training-loader resume.
Regenerate 500 native trunk updates per seed; no archived checkpoint is required.

- Fixed coefficient **0.06005351588542947** in all four auxiliary arms.
- Horizon **50**, observations **0, 1, 5, 10, 25, 50**.
- **1,500 trunk + 3,000 branch = 4,500 optimizer updates**.
- Six matched sets, twelve weight cells, sixty branches and endpoint feature
  caches, 360 report records. Three independent training seeds, not sixty.
- Training-only gradient-pressure probes before updates at offsets **0, 10, 25**:
  180 observations, no optimizer steps. These cost additional forward/backward
  computation and are not used to tune coefficients or choose actions.
- Same 4,458 training images and reused 512-image reporting pool. Report examples
  are excluded from all new training/probes; no fresh external test claim.

## Candidate rules, ties and overlap

For each text query, rank the other 63 batch images by current cosine similarity.
Exclude only its paired image. Do **not** filter by species or positive margin:
a task-negative may outrank its positive. No reporting examples enter selection.

1. **Hard support:** the historical top eight, using the exact existing `topk`
   implementation. This preserves Experiment A's intervention and tie behavior.
2. **Semi-hard support:** remove those eight; stable-sort the remaining candidates
   by descending similarity, breaking ties by batch index. Skip the first seven
   remaining candidates and select the next eight. These are ranks **16–23** in
   this explicitly defined ordering, roughly the 25th–37th percentiles from hardest.
3. **Real arms:** choose the highest-similarity member of the corresponding support;
   ties use the lowest batch index. With untied scores these are ranks 1 and 16.
4. **Synthetic arms:** equally mix the corresponding eight image embeddings and
   L2-normalize. One extra negative is added through the same relative-denominator
   auxiliary loss as prior experiments. Selection/mixture weights and auxiliary
   logit scale are detached; gradients pass through embeddings and denominator.

“Semi-hard” is an operational moderate-rank label, **not a guarantee of positive
margin or decision-boundary proximity**. Equal support size avoids comparing an
eight-way mixture with a narrower band. Missing support raises an error; there is
no fallback or data-dependent change to the band. Hard and semi supports must be
disjoint even when scores tie. Ties can still make them geometrically similar.

Every training step records support hashes, zero-overlap/diagonal checks, the gap
between hard minimum and semi-hard maximum similarity, contributor margin ranges
and nonpositive fractions, and actual real/synthetic candidate margins. Mixing and
normalizing can change difficulty ordering; contributor ranks alone do not establish
that a synthetic candidate is easier. Every branch's initial selection audit must
match at fixed weights. Subsequent selections track each branch's own current model.

The four auxiliary arms share a coefficient, **not matched effective gradient
pressure**. Probes report native and weighted-auxiliary gradient norms, their ratio
and cosine on the next training batch, with RNG/buffers/gradients/modes restored.
Together with total pre-clipping gradient norms, these help diagnose pressure
differences. This first comparison changes regime at fixed coefficient; it does
not isolate geometry from gradient magnitude. No calibration is fitted here.

## Replay and state controls

Before novel regimes at each weight state, native and hard-synthetic branches must
exactly reproduce Experiment A's corresponding **early-LR** cell: all six reporting
points, live state hashes, and every historical update-trace field. The new pinned
reference records parent result-file hashes and the original manifest/bundle IDs.
Historical native trunks, data identities, split and continuation captions also
must replay. Replayed branches are controls, not independent new evidence.

All branches check fresh AdamW counters 1–50, scheduler positions 101–150 after
updates, matched LR/RNG traces, identical starts at fixed weights, unchanged source
snapshots and frozen parameters. Reports, pressure probes and persistence are
guarded against changing training state. A completed branch's traces are persisted
before starting the next branch. Failure stops the run; no automatic retry/resume.

## Prespecified analysis

For each weight state and arm, report sustained-arm minus matched native loss
`D(w, arm)` and reporting-pool bidirectional R@1 difference in percentage points.
Average the two streams within seed, then equally across the three seeds.

The single primary contrast at horizon 50 is:

```text
Q(w) = D(w, synthetic_semihard) - D(w, synthetic_hard)
primary = Q(500) - Q(100)
```

Negative primary means a relative shift toward semi-hard. It does **not** by itself
establish the hypothesized crossover, which requires `Q(100) > 0` and `Q(500) < 0`.
Nor does a crossover establish useful supervision: always inspect each arm's
absolute effect versus native; both candidates could harm. No controller is trained.

Secondary: real-negative equivalents, all arm-versus-native effects and earlier
horizons, R@1 separately from loss, selection geometry and gradient pressure.
Show individual seeds and stream pairs. No favorable horizon selection,
significance/equivalence claim, monotonic curriculum claim, or report-tuned policy.
The reused reporting pool is development evidence for subsequent choices.

## Colab delivery and local checks

Paste [the entire self-contained cell](../colabs/clip_difficulty_drive_one_cell.py)
into a single **A100** runtime, preferably with high RAM. Requires the pinned
**torch 2.11.0+cu128 / torchvision 0.26.0+cu128** stack, not cu130. Preflight rejects
a mismatch; it does not silently replace CUDA packages. Fifteen GiB free runtime
disk is required. CPU tests run before GPU work.

New outputs: `MyDrive/OTCO/evaluation_transfer/clip_difficulty_<timestamp>`;
logs: `clip_difficulty_logs`. Wait for **DRIVE_SYNC_COMPLETE**, then download the
whole new run folder and log. Old experiment folders are untouched. No model
checkpoint tensors are written; starting states stay in RAM.

```bash
python -m scripts.build_clip_difficulty_cell
python -m pytest -q tests/test_clip_difficulty.py
```

`comparison.json` contains nested per-seed effects and primary/secondary contrasts.
Each set's `result.json` contains branch reports, traces, support audits and pressure
observations. `completion.json` certifies completed grid/budget/runtime checks.
Local CPU tests are implementation checks, not a completed CUDA experiment.
