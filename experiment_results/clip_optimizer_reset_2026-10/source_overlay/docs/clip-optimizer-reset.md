# Bounded optimizer-state reset experiment v1

Status: implemented and locally tested; no real GPU experiment launched here.
Separate experiment ID: `clip_optimizer_reset_v1`. Earlier experiments and their
delivery cells remain unchanged. This protocol was selected after inspecting the
native-only and paired-history results; it is exploratory, not independent confirmation.

## Scientific question

Does accumulated AdamW state contribute to the early/later difference in response
to sustained uniform-top-8 auxiliary negative pressure? The paired-history study
found later loss worsening under both native and prior-gated history, with a small
mean history interaction and mixed seed effects. This intervention tests optimizer
state at fixed model weights, not a new gate or refitted treatment coefficient.

For each native-only starting state at updates **100 and 500**, cross:

| Optimizer | Continuation objective |
|---|---|
| Retained | Native only |
| Retained | Sustained auxiliary |
| Reset | Native only |
| Reset | Sustained auxiliary |

Reset means clear **all first/second moments and per-parameter step counters**.
AdamW then lazily initializes them at its next update (first parameter step = 1).
Do not change model weights, parameter groups, betas, epsilon, weight decay,
gradient clipping, logit-scale clamp, RNG or the external scheduler. The scheduler
continues from starting update 100 or 500, without restarting warmup. Every branch
uses the identical processed training batches and external LR trajectory at that
starting age. The reset is applied after restoration, separately for each reset arm.

This is **not a momentum-direction-only intervention**: second moments and
bias correction also change, affecting effective update magnitudes. A reset effect
would implicate optimizer state broadly, not identify which component matters.
If later harm persists, that does not prove representations are the cause; LR,
training exposure and other age-associated properties still differ across ages.

## Frozen design and budget

- Same three seeds: 271828, 161803, 141421.
- Same two continuation streams: 2026100101, 2026100102.
- 500 native-only starting-state updates per seed; snapshot 100 and 500 in RAM.
- 50 continuation updates per arm; fixed auxiliary coefficient 0.06005351588542947.
- Observations at **0, 1, 5, 10, 25, 50**. No gradient probes or adaptive decisions.
- Same 512 canonical-caption report images, excluded from new training; same
  4,458-image training population and original 1,024-image diagnostic exclusion.
- Batch 64; same scheduler horizon 3,850 and original training configuration.
- **1,500 starting-state + 2,400 branch = 3,900 optimizer updates** total.
- 12 matched sets, 48 branches, 288 point records, 48 endpoint embedding caches.

The primary is fixed at starting update **500**, continuation update **50**:

`[loss(reset_sustained) - loss(reset_native)] - [loss(retained_sustained) - loss(retained_native)]`

Average the two streams within each seed, then the three seeds equally. Negative
means reset makes treatment less harmful/more helpful. Always show absolute
sustained-minus-native effects under both optimizer conditions, all three seed
contrasts, and the fixed time course at both starting ages. The other timepoints
are secondary; do not choose a favorable horizon after results. Report mean
bidirectional R@1 separately. No significance claim, threshold search, automatic
extra seeds, report-based decisions or data trimming.

## Mandatory replay controls

Starting checkpoints were never written by the previous experiment, so they must
be regenerated, not retrieved from a checkpoint ZIP. The bundle includes a small
replay reference derived from the checksum-verified native-history run
`clip_native_history_policy_20261001T234400_383003Z` (backup manifest SHA256
`7ff9f85f2329e3ac4443ec08731d9f4b1fc6d7030ab1fecbcaa588a4f687ecc5`).

Before accepting a reset result, enforce exact agreement with that run for split,
training/continuation plans, report/probe identities, native trunk audit, starting
model/optimizer/scheduler/RNG fingerprints, and both retained native/sustained
branches' update traces and initial/final reports and state hashes. Extra report
observations must preserve RNG, model/buffers, gradients and optimizer/scheduler
state. Reset branches intentionally have different optimizer-state hashes, but
must start with the same weights, report, scheduler, parameter groups and RNG.

All retained branches execute before the corresponding reset branches. Replay
failure stops the run; partial output is not a completed mechanism experiment.
Do not loosen tolerances or silently replace the reference after a mismatch.
Reused seeds/streams/report data mean this is a within-state intervention, not new
independent seeds, a new external dataset or historically untouched evaluation.
Constructing the historical probe batches reproduces the original setup only;
their gradient probes are never executed in this experiment.

## Colab

Paste the entire [optimizer-reset cell](../colabs/clip_optimizer_reset_drive_one_cell.py)
in an **A100** Colab runtime (prefer the same A100-SXM4-40GB and high RAM).
Requires **torch 2.11.0+cu128 / torchvision 0.26.0+cu128**, not cu130, and at least
15 GiB runtime disk. No automatic replacement of the Torch/CUDA stack is allowed.
Other numerical package versions remain pinned to the reference run.

The cell needs no old checkpoint ZIP or uploaded native results. It clones the
pinned repository base, applies the checksum-verified overlay, runs its tests,
and persists the protocol/reference/source manifest before training. Source
fingerprints provide strict replay checks, not immunity to environment drift.

Results: `MyDrive/OTCO/evaluation_transfer/clip_optimizer_reset_<timestamp>`.
Full subprocess logs: `clip_optimizer_reset_logs`. Wait for `DRIVE_SYNC_COMPLETE`,
then download the whole results folder and log. More report evaluations add runtime
despite the unchanged optimizer-update budget. Endpoint caches occupy roughly
100 MiB; no large model checkpoints are written. RAM states cannot be resumed
after runtime loss. An existing manifest for this experiment blocks blind reruns;
previous experiments do not. No automatic retry or training is started by local tests.

Local tests: `python -m pytest -q tests/test_clip_optimizer_reset.py`.
Build: `python -m scripts.build_clip_optimizer_reset_cell`.
`comparison.json` contains the primary, per-seed effects and all paired time courses;
`completion.json` is written only after the full grid, budget and replay checks pass.
