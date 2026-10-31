# Running Experiment C

Status: implemented; GPU run **not performed**. The [frozen protocol](clip-pressure.md)
and its [original manifest](../configs/clip_pressure_freeze_manifest.json) retain
their preparation-time wording and checksums. Implementation does not amend them.

Paste the entire [self-contained Colab cell](../colabs/clip_pressure_drive_one_cell.py)
into a single A100 runtime, preferably with high host RAM. Use **torch 2.11.0+cu128
and torchvision 0.26.0+cu128**; cu130 is rejected rather than silently replaced.
Other numerical dependencies and the repository base are pinned by the cell.
No archived checkpoint or result ZIP is needed; compact replay references are
embedded. Fifteen GiB free runtime disk is required. CPU tests run before GPU work.

The cell regenerates native trunks, calibrates both weight states on the frozen
training batches, and runs 84 matched branches: **5,700 optimizer updates** in
total. Calibration and probes add forward/backward work without optimizer updates.
Three historical branches at each seed/stream/weight must replay B before any
calibrated branch at that state. Calibration failure or replay mismatch stops the
run; there is no automatic retry, retuning, fallback or resume.

Outputs use `MyDrive/OTCO/evaluation_transfer/clip_pressure_<timestamp>`; logs use
`clip_pressure_logs`. Wait for **DRIVE_SYNC_COMPLETE**, then download the whole
new run folder and log. Earlier experiment folders are untouched. Starting states
stay in RAM; the run writes 84 endpoint feature caches, not large model checkpoints.

Records include:

- Original and expanded runtime protocols, freeze manifest, pinned calibration
  membership, native/Experiment B references, source overlay and source manifest.
- Per-seed calibration batches and captions, per-weight individual unweighted
  norms/ratios, coefficients, and remeasured weighted norms/ratios. Partial and
  failed calibration records are persisted before stopping. Both weight-state
  calibrations are saved before any continuation for that seed.
- Each branch's starting/reset audit, update traces and candidate geometry, six
  reporting observations, four observational pressure/clipping probes, and cache.
- `comparison.json`: all arm-versus-native effects, target-specific Q, the primary
  later mean Q, construction × pressure and age interactions, with per-seed
  timecourses. `completion.json` is written after replay, calibration and budget checks.

Calibration is measured with all relevant losses sharing one training forward.
Each calibration batch measurement restores the common initial RNG, modes,
buffers and gradients; verification repeats the same native gradients exactly.
The complete live model/optimizer/scheduler state is restored after visiting both
calibration weight states. Observations and persistence callbacks are guarded.

Diagnostics include total pre-clipping gradient norms measured directly by
autograd during pressure probes and the actual update's norm in its trace. These
independent reductions may differ slightly in floating-point arithmetic. Clip
scales use the pinned implementation's `min(1, max_norm / (norm + 1e-6))` rule.
They do not control coefficients. Initial calibration is not matching AdamW
updates, individual batches, parameter groups, clipping or future pressure.

```bash
python -m scripts.build_clip_pressure_cell
python -m pytest -q tests/test_clip_pressure.py
```

Tests cover deterministic calibration membership, per-batch ratio arithmetic,
weighted-gradient verification and failures, state preservation, fixed strengths,
replay gates, bf16 update equivalence, analysis arithmetic and complete toy execution.
They are implementation checks, not a completed CUDA experiment.
