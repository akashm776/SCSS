# D1 partial resume evidence — 16 October 2026

Superseded by the [completed three-seed audit](research-status-2026-10-16-source-complete.md). The results below retain the earlier partial snapshot for provenance.

The downloaded run contains usable, internally consistent partial evidence, but D1 is incomplete. This audit does not change the frozen experiment or produce its final three-seed primary estimate.

## Evidence and verification

Inputs in `/Users/akashmittal/Downloads`:

- `clip_source_resume_20261015T014921_451960Z-20261016T122040Z-1-001.zip`
- `clip_source_resume_logs-20261016T122043Z-1-001.zip`
- Earlier comparison: `clip_source_resume_20261015T014921_451960Z-20261015T125309Z-1-001.zip`

The [machine-readable audit](../experiment_results/clip_source_partial_2026-10-16/audit.json) records archive SHA256 values, per-pair effects at every observation horizon, diagnostics, and error excerpts. The [audit script](../scripts/audit_clip_source_partial.py) reads the archives without modifying them.

All 6,032 Drive inventory entries, 135 archived source files, and 12 checkpoint files match their recorded hashes. All 70 completed branch JSON files in the older archive are byte-identical in the newer archive. Each checkpoint pointer retains one generation. These checks establish consistency with the recorded manifests, not independent proof of scientific correctness.

The five completed seed/stream pairs pass the saved audit checks: 900 updates per pair, all required observations and probes, matched initial states, fixed coefficients and exposure identities, and state-preserving probes. Their recorded native replay checks pass. The frozen analysis function correctly refuses the incomplete grid with `Incomplete/duplicate D1 grid; no survivor averaging`. Neither a final completion marker nor a final comparison exists.

## Durable progress

| Component | Saved progress |
|---|---:|
| Native trunks | All three at 500/500; 1,500 preparation updates |
| Complete branches | 90/108; 4,500 continuation updates |
| Active branch | Seed 141421, stream 2026100102, w100/native, 40/50 |
| Remaining | 10 updates on the active branch plus 17 × 50 = **860 updates** |

Thus 6,040/6,900 planned optimizer updates are durable, approximately 87.5%. This is an update count, not a wall-clock estimate; evaluations, probes, and synchronization also cost time.

The active checkpoint is `resume_state/seed_141421_stream_2026100102_854476efa16b4a218175097690896a84.pt`. It was hash-verified and read through the runner's restricted loader. This file alone is not a complete resume package: its pointer, contract, native starting states, calibration, population records, and accumulated results also matter.

## What stopped execution

1. **01:49 log: processor-loading failure.** The optional Hugging Face `additional_chat_templates` lookup raised a 404 before training. The trace and installation sequence are consistent with an already-imported Hub version surviving an in-process package replacement. The subsequent bootstrap launches a fresh worker after installation; later training passed this point.
2. **02:04 log: interrupted run.** This log has no terminal exit marker and ends around the second seed's second stream. The log alone does not establish how it stopped. Later resume output and the unchanged branch files show completed work was retained.
3. **10:53 log: lease-file read failure.** Saving the final stream's first native branch failed because `resume_lease.json` could not be read. This is distinct from the explicit “Concurrent D1 writer detected” error. The user reports no deletion from the active folder. Transient Drive visibility or a heartbeat/read race is a plausible explanation, not a proven cause. The saved active checkpoint remains at update 40. The lease repair adds serialized heartbeat/check operations and bounded read retries while preserving ownership checks.
4. **19:43 log: runtime-version guard.** The new runtime failed the required `torch == 2.11.0+cu128` check before training. The log does not report the actual installed Torch version. The saved run records Torch 2.11.0+cu128 and torchvision 0.26.0+cu128. This later failure does not show that the lease repair failed; training never reached that test.

No recorded failure here establishes corrupted weights, changed treatment definitions, or a failed scientific invariant. The immediate continuation requirement is to restore and verify the pinned runtime, then resume the existing run with the lease repair and latest-only retention. Do not bypass the version guard or start a new scientific run to work around it.

## Preliminary H50 results

The following is **descriptive evidence from the two fully completed seeds only** (271828 and 161803). Average the two streams within each seed, then weight those seeds equally. Do not substitute this table for the prespecified three-seed analysis. The third seed's completed first stream is reported separately in the audit JSON.

Values are reporting-loss differences against matched native continuation; negative is improvement.

| Arm | Early, w100 | Late, w500 |
|---|---:|---:|
| H-Select | -0.000049 | +0.000223 |
| H-Synth | -0.000561 | +0.000362 |
| M-Select | -0.000232 | +0.000213 |
| M-Synth | -0.000814 | +0.000003 |
| C-Select | -0.000105 | +0.000142 |
| C-Synth | -0.000813 | +0.000152 |
| I-Select | -0.000022 | +0.000136 |
| I-Synth | -0.000659 | +0.000244 |

Uniform synthesis beats its Select representative in all four early regime averages. Late, all eight averages are worse than native, although M-Synth is nearly tied. Late M-Synth beating M-Select therefore does not establish improvement over native. These are averages, not assertions about every individual pair. The third seed's available first stream also has lower loss than native for all eight early arms and higher loss for all eight late arms.

The primary side hypothesis is not consistent across completed seeds. With Q = C-Select minus I-Select, and Delta = Q(late) minus Q(early):

| Seed | Streams available | Q early | Q late | Delta |
|---|---:|---:|---:|---:|
| 271828 | 2/2 | +0.000087 | -0.000025 | -0.000112 |
| 161803 | 2/2 | -0.000252 | +0.000037 | +0.000289 |
| 141421 | 1/2 | -0.000136 | +0.000047 | +0.000183 |

The first seed's mean has the proposed I-early/C-late crossover; the second seed has the opposite crossover. The incomplete third seed currently points in the second direction. A reliable state-dependent C/I curriculum is not established.

The loss differences are small. Retrieval changes are mixed: in the two-seed table, mean bidirectional R@1 changes range from -0.122 to +0.024 percentage points. This is reused development evidence, without a fresh confirmation set or an established uncertainty interval.

## Pressure diagnostics

All 48 seed/state/intervention calibration cells were checked. Verified mean raw gradient ratios range from 0.01996657 to 0.02002695, with maximum relative error 0.1672% from the 0.02 target. Calibration records report zero optimizer updates and preserved state.

Clipping is active on all 4,500 updates in the completed pairs, including native branches. Across intervention/state cells, averaging each diagnostic over the five completed pairs, initial effective AdamW displacement differences range from 13.75% to 15.76%; at offset 49 they range from 0.65% to 0.91%. These unbalanced five-pair diagnostic means are descriptive, not the primary outcome aggregation.

The 2% condition successfully controls initial average raw gradient norm on calibration batches. It does not guarantee a 2% optimizer displacement. The diagnostics support retaining that distinction when interpreting these results; they do not by themselves explain which geometric mechanism caused an advantage.

Complete the remaining frozen branches before producing the primary result. The current evidence motivates caution about late intervention and suggests an early benefit of uniform combination relative to the specified real representative, but it does not justify changing arms, pressure, or stopping rules based on these partial outcomes.
