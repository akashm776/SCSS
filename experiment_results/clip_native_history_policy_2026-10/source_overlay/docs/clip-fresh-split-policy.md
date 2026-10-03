# Fresh-seed / new-reporting-split policy replication

Status: implemented, not GPU-executed. This is a prospective bounded follow-up
within the previously studied CUB dataset, not a new-dataset confirmation.

## Why this test

The September full-pool replay favored gated over dose-matched random timing by
0.00017561 loss units. The subsequent CPU image-influence analysis found that this
pooled sign survived all 32 fixed paired-pool group deletions and all 512
single-query deletions. Nevertheless, the small net effect was cancellation
sensitive, and later gating continued to hurt relative to native training.
More evaluation rearrangements of those same endpoints cannot establish transfer.

The next question is: **does the frozen policy's small timing advantage recur on
new training seeds and a newly excluded reporting split, and does the early/later
contrast against native training persist?**

## Frozen contract

| Component | Precommitted choice |
|---|---|
| Training seeds | 271828, 161803, 141421 |
| New report | 512 canonical-caption training images, deterministic identity ranking with seed 2026100104 |
| Exclusions | Historical 1,024-image holdout remains excluded; new report excluded too |
| Training | 4,458 images, batch 64, 69 drop-last batches/epoch |
| Gate probes | Same frozen 16 × 64 training images/captions; report cannot enter probes |
| Starting-state histories | Native updates 1–100; frozen alignment gate after 100, refreshed at completed updates 100 and 250 |
| Starting states | Completed updates 100 and 500, freshly generated for every seed |
| Continuations | Two new 50-update streams at each state; seeds 2026100101 and 2026100102 |
| Arms | Gated, native, sustained, random timing matched to gated total dose |
| Gate | Mean gradient cosine strictly > 0.030847286945687016; alpha 0.06005351588542947 |
| Branch refreshes | Completed continuation updates 0, 10, 25; report never controls actions |
| Random comparator | One fixed permutation per set, seed 2026100103; first K positions used for gated dose K |
| Reporting | Offsets 0 and 50; symmetric contrastive loss with all 512 candidates and mean bidirectional R@1 |
| Budget | 3 × 500 trunk updates + 3 × 2 × 2 × 4 × 50 branch updates = **3,900** |

The report ranking excludes both the old holdout and frozen training-probe IDs.
No old report outcomes are used to choose individual images. The resulting split,
continuation source indices and timing permutations are saved and synced before
the first update. Image-key overlap and archived probe-caption hashes are checked
before training. The old validation set is not evaluated.

Each four-arm set restores the same model, optimizer moments, scheduler and RNG
state; all arms consume identical processed batches, caption choices and learning
rates. Observations/persistence must preserve model buffers, modes, gradients,
optimizer and RNG. Frozen parameters and branch snapshots are audited immutable.
Gate decisions are synced before their affected updates. Random timing sees only
the gated total exposure, not scores or reporting outcomes. It is a conditional
timing comparator, **not a deployable online policy**. Zero/full exposure and a
random schedule identical to gated are retained and reported, never redrawn.

## Endpoints and interpretation

Primary: gated minus random-matched **full-512 symmetric loss at update 50**,
averaging the two streams within each starting state, the two starting states
within each training seed, then the three training seeds equally. Negative helps.
Report all three seed means, all individual sets, exposure and coincident schedules.
Secondary: gated-minus-native and gated-minus-sustained loss and R@1, with early
and later states shown separately. Positive R@1 differences help.

Three training seeds are the replication units, not 12 independent sets or 512
independent policy runs. The two streams reuse the same seed-specific states.
No outcome-dependent checkpoint selection, coefficient/threshold tuning, report
trimming or automatic extension is permitted. No significance or superiority claim
follows merely from a negative pooled primary. A timing gain that does not beat
native training is not a demonstrated practical auxiliary-training gain.

This is **not an untouched external test**: the new report images were available
as training images in older studies, and the frozen rule was developed in this
dataset. They are held out from *all updates and gate probes in this new run*.
Changing the training population from 4,970 to 4,458 also means this is not an
exact replication of old states. The scheduler retains horizon 3,850 to preserve
step-indexed LR, even though epoch lengths change. New streams are matched
continuations, not literal source-loader resumes. Keep this run separate from old
results; a further new-dataset test would still be needed for broader transfer.

## Run in Colab

Paste the entire [self-contained cell](../colabs/clip_fresh_split_policy_drive_one_cell.py)
into a **single-A100 Colab runtime**, preferably high-RAM. Requires the existing
Torch 2.11.0+cu128 / torchvision 0.26.0+cu128 stack and at least 15 GiB free runtime
disk. Other numerical dependencies are pinned. The cell fetches the pinned public
repo base plus embedded source overlay, pretrained CLIP and CUB data.
**No old checkpoint ZIP or deleted Drive source folder is required.**

The full subprocess transcript is saved under
`MyDrive/OTCO/evaluation_transfer/clip_fresh_split_policy_logs` even when preflight
fails. Result folders begin `clip_fresh_split_policy_`; download the complete new
folder and its log after `DRIVE_SYNC_COMPLETE`.

Starting states are RAM-only snapshots; no large new model checkpoints are
written. Forty-eight endpoint embedding/scalar caches (~100 MiB plus audits) allow
CPU verification and later evaluation without GPU replay. Each cache is
round-trip/hash checked. Completed sets, decisions and reports sync progressively.
The final tree is read-back SHA256 verified and the cell separately flushes Drive.

A prior run manifest blocks blind reruns. Failure emits the full traceback and,
when the run folder exists, `failure.json`; partial Drive sync is not completion.
RAM snapshots are lost on runtime termination. Inspect partial output before
deciding whether a separately identified restart is appropriate; there is no
automatic retry/resume or budget expansion.

## Local validation

Run `python -m pytest -q tests/test_clip_fresh_split_policy.py` plus the regression
suite. Rebuild the cell with `python -m scripts.build_clip_fresh_split_policy_cell`.
Tests cover split/probe separation, deterministic fresh streams, exact trunk
exposure and snapshots, matched branch restoration, observational full-pool
reporting/caches, failure-before-update behavior, aggregation and package pins.
Local toy-model tests do not substitute for the real GPU experiment.
