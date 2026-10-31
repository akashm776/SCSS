# D1 completed evidence — 16 October 2026

D1 completed all 108 branches and 6,900 planned optimizer updates. This report supersedes the [partial analysis](research-status-2026-10-16-source-partial.md). It uses the frozen H50 endpoint, averages two streams within each seed, and weights all three seeds equally.

## Verification

Input: `/Users/akashmittal/Downloads/clip_source_resume_20261015T014921_451960Z-20261016T172003Z-1-001.zip`.

The [audit script](../scripts/audit_clip_source_complete.py) checked all 7,129 Drive inventory entries, 135 archived source files, the scientific freeze files, and 12 checkpoint files against their recorded hashes. The local analysis code matches the archived code. Recomputing the complete comparison from all six seed/stream result files reproduces the archived `comparison.json` exactly. All 90 completed branch files from the preceding download are byte-identical. All 432 saved probe preservation checks pass.

The completion record reports 648 observations, 5,400 branch updates, 1,500 native preparation updates, exact native trunk/continuation replay, fixed coefficients and support identities, and zero committed optimizer updates from calibration or probes. This audit checks the archived records and arithmetic; it does not independently rerun GPU training or reporting forward passes.

The remaining `failure.json` is historical. The runner writes `complete_pending_drive_flush` before its final sync and does not rewrite that status afterward. Consequently that manifest wording is expected here: the completion marker, matching inventory, reproduced full-grid comparison, and supplied final log's `DRIVE_TREE_VERIFIED`/`DRIVE_SYNC_COMPLETE` establish successful completion. The runtime and lease failures from earlier attempts did not prevent this final run.

Evidence: [audit.json](../experiment_results/clip_source_complete_2026-10-16/audit.json), [recomputed comparison.json](../experiment_results/clip_source_complete_2026-10-16/comparison.json).

## Primary C-Select versus I-Select question

Define Q = D(C-Select) − D(I-Select). Negative Q favors C. The frozen primary interaction is Q(late) − Q(early).

| Seed | Q early | Q late | Interaction |
|---|---:|---:|---:|
| 271828 | +0.000086814 | -0.000024934 | -0.000111748 |
| 161803 | -0.000251531 | +0.000037406 | +0.000288937 |
| 141421 | -0.000127970 | +0.000079777 | +0.000207747 |
| Equal-seed mean | -0.000097562 | +0.000030750 | **+0.000128312** |

The aggregate direction is opposite the proposed I-early/C-late curriculum: C is relatively better early, and I relatively better late. This is not consistent across seeds. Only seed 271828 has the proposed crossover; the other two have the opposite crossover. These small, heterogeneous effects do not establish a reliable side-based curriculum or a statistically established reverse curriculum.

## Absolute H50 effects against native

Negative loss differences improve on matched native continuation. R@1 is mean bidirectional retrieval, in percentage points; positive improves retrieval.

| Arm | Early loss difference | Late loss difference | Early R@1 change | Late R@1 change |
|---|---:|---:|---:|---:|
| H-Select | -0.000123700 | +0.000137865 | -0.08138 | -0.03255 |
| H-Synth | -0.000685671 | +0.000387684 | -0.03255 | +0.06510 |
| M-Select | -0.000272764 | +0.000215344 | -0.03255 | -0.01628 |
| M-Synth | -0.000824949 | +0.000152951 | -0.09766 | -0.01628 |
| C-Select | -0.000126386 | +0.000172882 | -0.04883 | 0.00000 |
| C-Synth | -0.000842937 | +0.000249853 | -0.03255 | +0.06510 |
| I-Select | -0.000028824 | +0.000142133 | -0.01628 | 0.00000 |
| I-Synth | -0.000766799 | +0.000244596 | -0.03255 | -0.03255 |

All eight aggregate early loss effects improve on native; all eight aggregate late effects worsen it. This is not true of every individual seed/arm: for example, late M-Synth improves loss for seed 161803, but worsens it for the other two. Native remains better than every late treatment by aggregate reporting loss under this protocol.

Every early arm's aggregate R@1 change is negative. Thus the loss improvements do not establish retrieval improvements. Late H-Synth and C-Synth show small positive R@1 changes despite worse loss. Keep both metrics visible rather than treating either as a universal measure of utility.

## Secondary synthesis question

G = D(Synth) − D(Select); negative favors synthesis.

| Source regime | G early | G late |
|---|---:|---:|
| H | -0.000561972 | +0.000249819 |
| M | -0.000552185 | -0.000062393 |
| C | -0.000716551 | +0.000076971 |
| I | -0.000737975 | +0.000102463 |

Early synthesis beats the specified real representative in all four regimes for each of the three seed averages. Its advantage is smaller late in every regime. Late M-Synth beats M-Select on average, but both are worse than native. This supports a state-dependent advantage of this two-source construction for reporting loss, conditional on the fixed common treatment population and pressure condition. It does not prove synthetic negatives are intrinsically superior to real negatives or that boundary placement caused the difference.

The numerical early leaders, C-Synth and M-Synth, differ by only 0.000017989 in mean loss effect; their ordering should not be treated as an established source winner.

## Diagnostics and limits

All 48 calibration cells verify initial raw gradient ratios between 0.01996657 and 0.02002695; maximum relative error from 0.02 is 0.1672%. Clipping is active in all 5,400 branch updates, including native. Across arm/state cells, mean effective AdamW displacement differences at offset zero range from 14.20% to 15.86%, falling to 0.68%–0.92% at offset 49. The calibration succeeded on its stated raw-gradient target; it did not equalize optimizer displacement.

The primary interaction changes sign across horizons: it is negative at 1, 5, 10, and 25 updates but positive at the frozen 50-update endpoint. Earlier horizons are descriptive and cannot replace the primary endpoint after inspection.

These are three reused development seeds, not six independent replications or a fresh confirmation set. No significance or population-wide reliability claim follows from the displayed means. The intervention is applied to frozen cross-state-eligible training exposures, while the outcome is measured on the full reporting pool. Source identities were fixed at the initial state, so this does not test an adaptive online negative-selection policy.

The supported conclusion is narrow: early uniform synthesis provides a consistent loss advantage over the specified one-real-negative controls; the proposed side curriculum lacks consistent support; and no tested late treatment improves aggregate reporting loss over native. D2's causal boundary-placement question remains unanswered.

## Reproduce the audit

```sh
.venv/bin/python -m scripts.audit_clip_source_complete \
  --archive /Users/akashmittal/Downloads/clip_source_resume_20261015T014921_451960Z-20261016T172003Z-1-001.zip \
  --previous /Users/akashmittal/Downloads/clip_source_resume_20261015T014921_451960Z-20261016T122040Z-1-001.zip \
  --output experiment_results/clip_source_complete_2026-10-16
```
