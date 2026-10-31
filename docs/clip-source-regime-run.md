# Run D1 in Colab

Paste the entire `colabs/clip_source_regime_drive_one_cell.py` into one Colab cell. Select **A100 GPU and high RAM** first. It requires one A100 with at least39 GB,20 GiB host RAM,15 GiB free runtime disk, torch2.11.0+cu128 and torchvision0.26.0+cu128. The cell refuses to silently replace the CUDA/PyTorch stack; missing `nvidia-smi` produces a runtime-selection instruction.

The cell clones the pinned repository base and overlays checksummed source plus the frozen D0-derived population. No old checkpoint ZIP or D0 run folder is required. It regenerates/authenticates the three native trunks, calibrates all eight treatments, then runs108 branches. Budget:6900 committed optimizer updates plus calibration, reports and paired virtual diagnostics. This is a substantial GPU run.

The protocol is `docs/clip-source-regime.md`. The single primary comparison is C-Select versus I-Select's state interaction at horizon50. No treatment is selected using reporting data.

Outputs go to `MyDrive/OTCO/evaluation_transfer/clip_source_regime_<timestamp>` and logs to `clip_source_regime_logs`. Wait for **DRIVE_SYNC_COMPLETE**, then download the run folder and its log. `PARTIAL_DRIVE_SYNC_COMPLETE` means only saved partial files were flushed; it is not successful completion. A prior D1 run blocks an ordinary rerun to avoid duplicate trials. Preserve any partial folder for inspection. This version does not save large model checkpoints or offer automatic resume.

Local checks run with `.venv/bin/python -c 'import sys; sys.modules["readline"]=None; import pytest; raise SystemExit(pytest.main(["-q","tests/test_clip_source_regime.py"]))'` on this Mac. Rebuild the self-contained cell with `.venv/bin/python -m scripts.build_clip_source_regime_cell` only after passing tests. A change to the frozen protocol/population requires a new version, not recalibration against outcomes.
