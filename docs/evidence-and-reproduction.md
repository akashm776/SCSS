# Evidence and reproduction

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
```

The first archive command covers evaluation transfer, gated training, exposure
matching and checkpoint diagnostics. The second covers time course, marginal
utility (including its failed/partial predecessor) and policy tests; it verifies
that six partial pairs are identical to the completed run and not extra samples.
The October command covers pool replay, native history, gated history and reset;
it also recomputes the paired-history comparison and checks the frozen plan.
Earlier archives have their own provenance/readme records; these commands are
not a claim that every historical artifact was regenerated.

The generic project dependency range supports local development, not bitwise replay
of a past Colab environment. The recorded recent GPU environment is A100-SXM4-40GB,
Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 4.57.3,
datasets 2.21.0, numpy 2.1.3 and PyYAML 6.0.3. Issued cells enforce their own pins;
inspect a mismatch rather than removing guards or silently replacing packages.

## Re-export from the full downloaded runs

Keep these extracted directories or the corresponding complete ZIPs outside Git:

```text
clip_policy_pool_test_20260930T120952_487061Z
clip_native_history_policy_20261001T234400_383003Z
clip_fresh_split_policy_20261002T024902_873109Z
clip_optimizer_reset_20261002T043855_470158Z
clip_paired_history_comparison_v1_plan
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

## Issued experiments now completed

| Experiment | Issued cell | Current results |
|---|---|---|
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
