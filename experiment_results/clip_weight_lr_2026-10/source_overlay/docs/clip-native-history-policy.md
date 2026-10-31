# Native-history policy experiment v1

Status: implemented; no real GPU run yet. This is the recommended alternative to
launch next, since the earlier gated-history replication has not been started.
That experiment, its source files, protocol and embedded Colab bundle are preserved
unchanged. Running this cell does not run the earlier experiment first.

## Question and hypothesis

Does additional uniform-top-8 synthetic-negative pressure still help at an early
state and harm at a later state when **neither starting state has any prior
auxiliary treatment**? Separately, does the frozen alignment policy beat
dose-matched random timing on these native-history states?

The old later states had gated-treatment history, so age and prior treatment were
entangled. The hypothesis is that the early/later response may persist without
that treatment history; the competing possibility is that it weakens or reverses.
Neither outcome is assumed. This test removes prior auxiliary exposure, not all
differences associated with age: model parameters, AdamW moments, learning rate
and data exposure still change between updates 100 and 500.

## Fixed input/output contract

Input: the same pretrained CLIP, CUB split, seeds, gate rule and continuation plans
as `clip_fresh_split_policy_v1`, but no archived trained checkpoint. Output: a
separately named `clip_native_history_policy_*` folder containing all decisions,
update audits, matched branch results, comparisons, 48 endpoint embedding caches,
source pins and a completion record. A separate full subprocess log covers errors
before a run folder exists. Validation: toy-model state/gradient/RNG and persistence
tests, fixed-budget checks, source checksums and verified Drive read-back.

## Frozen design

| Component | Choice |
|---|---|
| Experiment ID | `clip_native_history_policy_v1` |
| Seeds | 271828, 161803, 141421 — same planned seeds as the unrun parent |
| Starting-state generation | **500 native-only updates per seed; alpha = 0 throughout; no gate probes** |
| RAM snapshots | Completed updates **100 and 500**, both labeled `baseline`, zero prior auxiliary updates |
| Reporting split | Same 512 canonical-caption CUB images; split seed 2026100104 |
| Training population | 4,458 images; excludes the old 1,024-image holdout and the new 512-image report |
| Gate inputs | Same frozen 1,024 training probe images; disjoint from reporting images |
| Future streams | Same two 50-update streams per state; seeds 2026100101 and 2026100102 |
| Branches | Gated, native, sustained, random timing matched to total gated exposure |
| Branch gate | Mean cosine > 0.030847286945687016; alpha = 0.06005351588542947 |
| Gate refreshes | Branch offsets 0, 10, 25; never during native starting-state generation |
| Reporting | Branch offsets 0 and 50; full-512 symmetric contrastive loss and bidirectional mean R@1 |
| Schedule | Batch 64, 69 batches/epoch; cosine horizon 3,850, warmup 100 |
| Total optimizer budget | **1,500 native starting-state updates + 2,400 branch updates = 3,900** |

All four branches at a given state/stream restore identical model weights,
optimizer moments, scheduler and RNG, and use the identical processed batches.
Starting states are generated all the way to 500 **before branching**, so no branch
can contaminate the subsequent native training path. Snapshot metadata, update
coefficients and scheduler positions are checked. No native-history update even
constructs the auxiliary loss. Reporting and probe callbacks cannot be passed
into the native trunk function.

The gated branch still receives the frozen training-only probes. The sustained
branch is always treated for its 50 updates; native is never treated. One
precommitted permutation supplies random-matched timing with the same total dose
as gated. Zero/full exposure and identical schedules are retained without redraw.
The random comparator is conditional on gated dose, not a deployable online rule.

## Endpoints and interpretation

The **primary is unchanged** from the parent: gated minus random-matched full-512
loss at continuation update 50, averaged over streams within starting states,
states within seeds, then the three seeds equally. Negative favors gating.

The new mandatory **secondary history diagnostic**, saved as
`comparison.json:sustained_minus_native`, reports:

1. Sustained-minus-native loss and R@1 from the update-100 state.
2. The same two effects from the update-500 state.
3. Their later-minus-early contrast, per seed and averaged equally across seeds.

For each absolute effect, negative loss helps and positive R@1 helps. A positive
later-minus-early loss contrast only means treatment is less helpful/more harmful
later; it does **not** establish actual later harm without a positive later-state
absolute loss effect. Loss and retrieval remain separate endpoints. All individual
set results and gated-minus-native/sustained controls are also retained.

Three seeds are the replication units, not 12 independent matched sets. If both
variants are eventually run, their deliberately shared seeds, split, streams and
permutations make them paired history conditions, **not six independent seeds**.
This run alone cannot identify whether optimizer moments, representations, LR or
candidate semantics explain any remaining early/later difference. It cannot prove
that prior treatment causes a difference relative to older unmatched experiments.

No threshold/alpha refitting, report trimming, outcome-based checkpoint selection,
automatic extra seeds or automatic retries. These reporting images are excluded
from all training/probes in this run but were available in older CUB studies;
this is not historically untouched or external test data. The original validation
set is not evaluated. No significance claim is implied by three seed means.

## Colab run

Paste the **entire** [native-history cell](../colabs/clip_native_history_policy_drive_one_cell.py)
into a single-A100 Colab runtime. Prefer high-RAM. Requires at least 15 GiB free
runtime disk and the existing Torch 2.11.0+cu128 / torchvision 0.26.0+cu128 stack.
Other numerical packages are pinned. The cell fetches the pinned repo base,
pretrained model and dataset; **no old checkpoint ZIP or previous-run folder is
required**. It does not depend on running the earlier cell.

Output: `MyDrive/OTCO/evaluation_transfer/clip_native_history_policy_<timestamp>`.
Logs: `MyDrive/OTCO/evaluation_transfer/clip_native_history_policy_logs`.
Wait for `DRIVE_SYNC_COMPLETE`, then download the complete new run folder and log.
No large trained-model checkpoints are written: starting snapshots stay in RAM.
The 48 numerical endpoint caches occupy roughly 100 MiB plus audit files.

A prior manifest for **this experiment** blocks a blind rerun, but an earlier
gated-history run does not. Failure produces traceback/log and partial results;
partial sync is never completion. RAM snapshots cannot be resumed after runtime
loss. Inspect partial outputs before deciding on a separately identified restart.

## Reproduce and preservation

Build: `python -m scripts.build_clip_native_history_policy_cell`.
Test: `python -m pytest -q tests/test_clip_native_history_policy.py`.
Run all regressions before handoff. The package includes the unchanged helper
sources and validates SHA256 pins. `configs/clip_native_history_parent_pins.json`
records the earlier experiment's sources, builder and cell to enforce preservation.
Parent bundle: `f84d5fd0dd5ec4ba3c47758bdf4e08ed43b31d92dbcc1d8c8ff3efb71a981a81`.

Expected structural update after results: determine whether the early/later pattern
survives absence of prior auxiliary exposure; then choose a bounded history or
real-negative control based on those results. No GPU results are claimed yet.
