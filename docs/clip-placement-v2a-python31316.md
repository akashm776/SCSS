# v2a environment revision: Python 3.13.16

The original bundle `4c2dfad8a323` stopped with INVALID on
2026-10-31 before the model forward. Colab reported Python
`3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]`, whereas D1 used
3.13.15. Reported PyTorch 2.11.0+cu128, CUDA 12.8 and cuDNN 91900 matched.
This is a separately versioned environment contract, not a reinterpretation of
that INVALID attempt or permission to accept arbitrary Python versions.

Run the complete `colabs/clip_placement_v2a_py31316_drive_one_cell.py` in the same
A100 high-RAM Colab runtime. It uses a new output folder whose name starts with
`clip_placement_v2a_py31316_`. No interpreter installation or downgrade is needed.

Only the required Python patch version and full build string change. All other
recorded environment fields must exactly match the archived v1 run. The launcher
writes and prints observed environment metadata before validating it, and mismatch
messages include expected and observed values. Both original and actual runtime
metadata are retained in the new immutable run contract.

The scoring diagnostic itself imports the unchanged v2a implementation. Gap,
population, sources, targets, scale generation, arithmetic, and acceptance
thresholds are unchanged. Exact reproduction of every archived placement field
is still mandatory **before** the FP32 head is evaluated. A reproduction mismatch
remains INVALID, not a reason to widen tolerances. Reproduction on this batch
does not establish general equivalence between Python builds.

PASS is still only a first-batch forward diagnostic result. No gradient probes,
calibration, or training are launched. Original v1 and v2a bundles, source files,
specifications and failure records are preserved.
