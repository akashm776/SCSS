# Analysis and maintenance scripts

[Research report](../README.md) · [Evidence index](../docs/paper/evidence.md) · [Colab guide](../colabs/README.md)

Run these commands from the repository root. Analysis reads retained results;
experiment execution lives in `src/` and `colabs/`.

## Common tasks

| Task | Entry point | Inputs and outputs |
|---|---|---|
| Rebuild the report figure and tables | `python -m scripts.build_research_summary` | Reads completed D1 comparison data; writes Figure 1 and Appendix A |
| Audit placement calibration | `python -m scripts.audit_clip_placement_v2c INPUT --output OUTPUT` | Reads an archived run or ZIP; writes an audit and report |
| Audit the earlier placement stages | `audit_clip_placement_v2a.py`, `audit_clip_placement_v2b.py` | Same CLI; validates recorded forward/gradient evidence |
| Verify public pressure evidence | `python -m scripts.archive_clip_pressure_evidence --verify-only` | Checks retained files and recomputes summaries without exporting |
| Verify difficulty or weight/LR evidence | `archive_clip_difficulty_evidence.py`, `archive_clip_weight_lr_evidence.py` | Use `--verify-only` for the retained public archives |

The [reproduction guide](../docs/evidence-and-reproduction.md) describes input
requirements and the limits of each audit. Saved-result verification is not a
new training run or an independent gradient computation.

## Script families

- `analyze_*`: derive contrasts or diagnostics from recorded results.
- `audit_*`: authenticate a run and check its recorded execution contract.
- `archive_*`: verify original downloads and export compact public evidence.
- `plot_*`: render historical experiment figures.
- `build_*_cell.py`: package versioned Colab launchers. These are release tools,
  not the normal entry point for evaluating results.

## Maintenance boundaries

Some files are pinned by issued source manifests. Changes to those files require
a new experiment bundle and explicit versioning. Generated one-cell launchers
and archived source snapshots are preserved byte-for-byte; edit a maintained
implementation rather than an embedded payload.

For analysis refactors, retain output schemas and verify results against the
archived evidence. Keep data loading, contract validation, computation, and
presentation separate where they have distinct responsibilities. Prefer explicit
names and short comments explaining numerical or provenance constraints.
