# D1 implementation validation — 2026-10-13

D1 is implemented and packaged. It has not been run on the A100, and no D1 treatment outcomes exist.

- Local regression suite: **278 passed**. This includes the 17 new D1 tests, the Colab preflight test list and D0 recovery tests.
- Clean pinned-base checkout plus the exact embedded source overlay: **256 passed, 11 skipped**. Skips are tests of generated delivery files absent from the overlay; the new D1 delivery checksum is verified separately against the issued cell. The D1 execution test passes in the clean checkout.
- All 118 issued D0 recovery source members remain unchanged, including its generated manifest verified inside the original embedded bundle.
- All 624 raw D0 query shards were checked against the archived shard hashes when deriving the frozen D1 population. Source order and eligibility were recomputed from margins. The resulting common population has 10,667 continuation exposures and 416 calibration exposures; every batch has at least 26 eligible rows.
- End-to-end toy execution regenerates and authenticates native states, validates processed identities, calibrates both states, executes all nine arms, checks native replay, persists traces/probes/reporting points/caches, and produces the prespecified analysis.
- Tests cover masked row means with full native denominators, detached auxiliary scale, gradients through live source features, frozen Select IDs after ranking changes, invalid mixtures/supports, FP32 and CPU bf16 calibration, restoration on failures, and diagnostics that leave subsequent trajectories unchanged.
- Virtual AdamW displacement is compared with actual paired steps after optimizer moments evolve and with clipping active. The diagnostic restores weights, optimizer, scheduler, gradients, modes, buffers, RNG and backend settings.

Source bundle SHA256:
`c63e7336dda5eca66c2ad3b661077f71e15becb8791d2ba43a625f4b15e53309`

Protocol/population freeze SHA256:
`c65026efd643e5be5bc8171ae2e2f1c6716b1063433caec82b19a08031d57dc9`

Population file SHA256:
`66b17ee9fa1dcf7b7325ab8552fbe2f8362c5fda68b9ecfd011292badaaff7fa`

The numerical CUDA stack, actual CLIP replay and real-data pressure calibration are still runtime gates. Local CPU tests do not certify those GPU results. The only warnings observed were pre-existing NumPy private-namespace deprecation warnings in the unchanged D0 snapshot loader.

[Protocol](clip-source-regime.md) · [Run instructions](clip-source-regime-run.md) · [Colab cell](../colabs/clip_source_regime_drive_one_cell.py)
