# Colab experiment guide

[Research report](../README.md) · [Current progress](../docs/current-progress.md) · [Reproducibility](../docs/paper/reproducibility.md)

This directory preserves issued experiment cells and their runners. For evaluating the research, start with the report and retained results. Use this guide when reproducing a specific stage; filenames alone do not indicate which experiment is current.

## Current next run

**[Full-population D2 preflight cell](clip_placement_v2d_drive_one_cell.py)** · [Protocol](../docs/clip-placement-v2d.md). Run on A100 40GB/high-RAM Colab. Covers 624 initial batch/state checks with frozen coefficients and zero training updates. GPU acceptance is pending.

## Placement validation: latest completed stages

| Stage | Read first | Issued cell | Recorded status |
|---|---|---|---|
| v2a: paired FP32-head diagnostic | [Python 3.13.16 amendment](../docs/clip-placement-v2a-python31316.md) | [Forward-only cell](clip_placement_v2a_py31316_drive_one_cell.py) | PASS on authenticated first batch |
| v2b: surrogate gradients | [Protocol](../docs/clip-placement-v2b.md) | [Gradient-validation cell](clip_placement_v2b_drive_one_cell.py) | PASS on audited batch |
| v2c: initial pressure | [Protocol](../docs/clip-placement-v2c.md) | [Calibration cell](clip_placement_v2c_drive_one_cell.py) | PASS across six seed/state cells |

These stages commit zero optimizer updates. **The full-population audit is ready to run; its GPU acceptance and D2 continuation training remain pending.** The calibration cell is not a D2 training launcher. All cells require the authenticated upstream inputs specified in their protocols.

## Completed training and geometry studies

| Study | Protocol / results | Reproduction entry point |
|---|---|---|
| D1 source regimes | [Protocol](../docs/clip-source-regime.md) · [Completed results](../docs/research-status-2026-10-16-source-complete.md) | [Resume launcher guide](../docs/clip-source-lease-repair.md) |
| D0 geometry | [Protocol](../docs/clip-geometry-audit.md) · [Completed results](../docs/research-status-2026-10-09-geometry.md) | [Original cell](clip_geometry_audit_drive_one_cell.py) · [Recovery guide](../docs/clip-geometry-recovery.md) |
| C: calibrated pressure | [Protocol](../docs/clip-pressure.md) · [Results](../docs/research-status-2026-10-08-pressure.md) | [Cell](clip_pressure_drive_one_cell.py) |
| B: difficulty | [Protocol](../docs/clip-difficulty.md) · [Results](../docs/research-status-2026-10-07-difficulty.md) | [Cell](clip_difficulty_drive_one_cell.py) |
| A: weights × learning rate | [Protocol](../docs/clip-weight-lr.md) · [Results](../docs/research-status-2026-10-07.md) | [Cell](clip_weight_lr_drive_one_cell.py) |

Earlier history, optimizer, predictor, and training experiments are indexed in the [research ledger](../docs/experiment-ledger.md). Their original run instructions remain attached to each study.

## Historical variants

- [D2-P v1](../docs/clip-placement-preflight.md) is preserved as a failed numerical contract, not a current placement launcher.
- The [original v2a environment](../docs/clip-placement-v2a.md) predates the explicit Python 3.13.16 amendment.
- D1's initial, resume, bootstrap, storage, and lease-repair cells document successive operational changes to the same scientific protocol. They are not separate independent experiments.

## File conventions

`*_one_cell.py` files are generated pasteable Colab bundles. `run_*.py` files are experiment runners; shared modules supply input authentication, logging, and recovery. Corresponding `scripts/build_*_cell.py` files build issued cells. Editing a runner does not retroactively change an already issued embedded bundle.

Historical files remain in place because imports, embedded payloads, and provenance records depend on them. Readable navigation is provided here rather than renaming or deleting evidence-bearing launchers. Generic local installation is not a historical GPU replay environment; follow each protocol's version checks and recovery rules.
