# Evidence and reproduction

[Reader guide](paper/reproducibility.md) · [Current progress](current-progress.md) · [Claim-to-evidence index](paper/evidence.md)

Current public name: SCSS. Historical paths and issued bundle contents remain
OTCO. The GitHub repository URL and Python project identifier are deliberately
unchanged. No historical bundle was rebuilt solely to implement the rename.

## Three different kinds of verification

1. **Public evidence verification:** check the retained files against checksums
   and pinned original manifests; recompute arithmetic from retained numerical
   rows. This can run on a clone without datasets or a GPU.
2. **Full downloaded-result verification:** additionally check omitted source
   inputs and endpoint NPZ caches, recomputing loss/R@1. This needs the original
   downloaded runs, not merely the filtered GitHub archive.
3. **Experimental replication:** rerun training under the frozen protocol and
   environment. Neither checksums nor CPU arithmetic constitute a new GPU run
   or an independent scientific replication.

Manifests are consistency/provenance records, not cryptographic signatures of an
independent authority. A verified local copy does not prove a remote Drive flush.

## Public checks

```bash
uv sync
uv run --with pytest python -m pytest -q
uv run python -m scripts.archive_clip_followup_evidence --verify-only
uv run python -m scripts.archive_clip_september30_evidence --verify-only
uv run python -m scripts.archive_clip_october_evidence --verify-only
uv run python -m scripts.archive_clip_weight_lr_evidence --verify-only
uv run python -m scripts.archive_clip_difficulty_evidence --verify-only
uv run python -m scripts.archive_clip_pressure_evidence --verify-only
```

The first archive command covers evaluation transfer, gated training, exposure
matching and checkpoint diagnostics. The second covers time course, marginal
utility (including its failed/partial predecessor) and policy tests; it verifies
that six partial pairs are identical to the completed run and not extra samples.
The October command covers pool replay, native history, gated history and reset;
it also recomputes the paired-history comparison and checks the frozen plan.
The weight/LR command checks its retained replay/matching records and recomputes
all seed/timepoint factorial contrasts. See the [completed report](research-status-2026-10-07.md).
The difficulty command checks fixed-LR replay records, all arm/interaction
arithmetic, support and pressure records, and its derived diagnostics. See the
[difficulty report](research-status-2026-10-07-difficulty.md).
The pressure command additionally checks frozen calibration membership,
individual norm/ratio arithmetic, coefficients, verification tolerances, clipping
records and all three historical B replay arms. See the
[pressure report](research-status-2026-10-08-pressure.md).
Earlier archives have their own provenance/readme records; these commands are
not a claim that every historical artifact was regenerated.

The generic project dependency range supports local development, not bitwise replay
of a past Colab environment. The recorded D1-era GPU environment is A100-SXM4-40GB,
Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 4.57.3,
datasets 2.21.0, numpy 2.1.3 and PyYAML 6.0.3. Issued cells enforce their own pins;
inspect a mismatch rather than removing guards or silently replacing packages.

## Latest placement artifact audits

Recompute the saved-artifact audits into a separate local directory:

```bash
uv run python -m scripts.audit_clip_placement_v2a experiment_results/clip_placement_v2a_2026-10-31/source.zip --output /tmp/scss-v2a-audit
uv run python -m scripts.audit_clip_placement_v2b experiment_results/clip_placement_v2b_2026-10-31/source.zip --output /tmp/scss-v2b-audit
uv run python -m scripts.audit_clip_placement_v2c experiment_results/clip_placement_v2c_2026-10-31/source.zip --output /tmp/scss-v2c-audit
```

These commands authenticate recorded inputs and recompute selected numerical
summaries, not GPU gradients or training. The D1 [comparison](../experiment_results/clip_source_complete_2026-10-16/comparison.json)
and [audit](../experiment_results/clip_source_complete_2026-10-16/audit.json) support
the generated report; its full-run auditor additionally requires the original run archive.

The explicit [Python 3.13.16 amendment](clip-placement-v2a-python31316.md) applies
to the later placement validation. It does not rewrite D1's environment record.
Use each stage's exact configuration; an environment mismatch is not repaired by
silently disabling validation. [Colab navigation](../colabs/README.md).

## Re-export from the full downloaded runs

Keep these extracted directories or the corresponding complete ZIPs outside Git:

```text
clip_policy_pool_test_20260930T120952_487061Z
clip_native_history_policy_20261001T234400_383003Z
clip_fresh_split_policy_20261002T024902_873109Z
clip_optimizer_reset_20261002T043855_470158Z
clip_paired_history_comparison_v1_plan
clip_weight_lr_20261007T024147_218129Z
clip_difficulty_20261007T114311_800923Z
clip_pressure_20261007T235549_427102Z
```

The paired-history analyzer requires the native/gated runs to be extracted. The
other run inputs may be directories or single complete ZIPs. Use an empty output
location: the exporter refuses to overwrite evidence.

```bash
uv run python -m scripts.archive_clip_october_evidence \
  --downloads /path/to/downloaded-runs \
  --output-root /path/to/new-evidence-export
```

Before export this verifies the pinned original manifests (1,524 manifested files
across the four runs), original source snapshots, 192 endpoint caches, paired-history
controls, and reset replay/summary checks. File counts exclude the four backup
manifests themselves. Full-input checks are read-only; no training is launched.

Public export intentionally excludes raw captions/images, checkpoint tensors and
NPZ caches. JSON numerical records, source overlays, cache metadata, original
manifests and export audits are retained. Original manifests list omitted files;
their presence is provenance, not a claim that those files are in Git.

Export the weight/LR run separately with
`python -m scripts.archive_clip_weight_lr_evidence --downloads /path/to/downloaded-runs --output-root /path/to/new-evidence-export`.
This checks 548 original manifest members, recomputes 48 cached endpoints, verifies
native/early-diagonal replay records and independently recomputes the factorial
arithmetic. It retains 489 files (including the backup manifest), omitting 60.

Export the difficulty run with
`python -m scripts.archive_clip_difficulty_evidence --downloads /path/to/downloaded-runs --output-root /path/to/new-evidence-export`.
It verifies 902 original manifest members, recomputes 60 cached endpoints, retains
831 files (including the backup manifest) and omits 72. Its additional derived
`diagnostics.json` is checked by recomputation in clone-only mode. Per-step gradient
and selection records are audited, but the underlying gradients/features were not
saved and cannot be reconstructed from final endpoint caches.

## Issued experiments now completed

Export the pressure run with
`python -m scripts.archive_clip_pressure_evidence --downloads /path/to/downloaded-runs --output-root /path/to/new-evidence-export`.
It verifies 1,322 manifested files, recomputes 84 cached endpoints, retains 1,224
original files (including the backup manifest) and omits 99. Derived diagnostics
are checked by recomputation in clone-only mode. Calibration gradients and runtime
state preservation are recorded/audited, not recreated by the CPU verifier.

| Experiment | Issued cell | Current results |
|---|---|---|
| Initial-pressure calibration | [Cell](../colabs/clip_pressure_drive_one_cell.py) | [Archive](../experiment_results/clip_pressure_2026-10/README.md) |
| Difficulty regimes | [Cell](../colabs/clip_difficulty_drive_one_cell.py) | [Archive](../experiment_results/clip_difficulty_2026-10/README.md) |
| Weight-state × LR | [Cell](../colabs/clip_weight_lr_drive_one_cell.py) | [Archive](../experiment_results/clip_weight_lr_2026-10/README.md) |
| Native history | [Cell](../colabs/clip_native_history_policy_drive_one_cell.py) | [Archive](../experiment_results/clip_native_history_policy_2026-10/README.md) |
| Gated-history replication | [Cell](../colabs/clip_fresh_split_policy_drive_one_cell.py) | [Archive](../experiment_results/clip_fresh_split_policy_2026-10/README.md) |
| Plan-freezing paired-history launcher | [Cell](../colabs/clip_paired_history_drive_one_cell.py) | [Frozen plan and comparison](../experiment_results/clip_paired_history_comparison_2026-10/README.md) |
| AdamW reset | [Cell](../colabs/clip_optimizer_reset_drive_one_cell.py) | [Archive](../experiment_results/clip_optimizer_reset_2026-10/README.md) |
| Image influence, CPU only | [Cell](../colabs/clip_image_influence_one_cell.py) | [Report](../experiment_results/clip_image_influence_2026-09/REPORT.md) |

These are historical reproducibility entry points, **not instructions to rerun all
experiments**. Starting-state generation/replays consume compute but do not create
independent seeds. Failure logs and partial output must not be relabeled complete.
The latest cells retain `MyDrive/OTCO/evaluation_transfer` intentionally.

## Experimental interpretation

- Average streams within seed, then seeds equally; show the individual seed effects.
- Keep meta/controller data separate from reporting. Data used to choose later
  experiments is not an untouched final test, even if excluded from gradient updates.
- Show fixed timepoints, absolute native/treated effects and interaction contrasts.
- Loss, reporting-pool retrieval, canonical retrieval and species accuracy are
  different metrics. Match their units and evaluation population.
- Preserve negative results, partial-run overlap and protocol selection timing.
- No raw captions, personal PDFs, downloaded papers or large model weights belong
  in this public evidence release.

## Completed D0 recovery

[The D0 archive](../experiment_results/clip_geometry_audit_2026-10/README.md)
retains the complete original summary losslessly compressed, compact derived
statistics, input/source manifests and completion/recovery records. Verification
authenticated all 2,668 downloaded files and 118 source-overlay members, the
624-batch grid, native state/trunk fingerprints and the 166 reused shard hashes.
Selected summary arithmetic was recomputed from all 39,936 raw query records;
GPU embeddings/forwards were not rerun. Remaining full-summary metrics were
checksum-verified, not independently recomputed. Raw query shards remain in the
original downloaded run/ZIP and are required for full re-verification.

The final Drive tree is verified and every downloaded member matches its hash.
The recovery subprocess/flush log was not included in this download. This limit
does not make the complete, matching artifact tree an incomplete experiment.
