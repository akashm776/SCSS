# D1 automatic-resume validation — 2026-10-14

The new execution bundle preserves the issued D1 scientific code, frozen population and protocol. No GPU experiment was launched during implementation.

- Full local suite: **293 passed** (including the 15 resume tests and inherited D1/D0 regression checks).
- Clean pinned repository plus source overlay: **270 passed, 12 skipped**. Skips cover generated delivery files omitted from the source overlay; the delivered cell's checksum/source consistency is separately tested locally.
- Final packaging adds only the clarified maximum replay window and a root-level `execution_protocol.json` provenance record. The affected full-run restart test and delivery verification passed again; the restart test also passed in the final clean overlay.
- All **126 original D1 source files remain byte-identical**. The original one-cell delivery and scientific freeze remain intact.

Recovery tests compare resumed versus uninterrupted traces, reporting points, optimizer trajectories, native snapshots, and final result records exactly. They cover multiple interruption positions, repeated interruption while skipping completed branches, native dataloader/private-generator reconstruction, persisted gradients and modes, checksum/contract rejection, failed checkpoint uploads and pointer publication, old-run completed-branch reuse, ambiguous run selection, lease handling, and full-run restart without recalibration or additional native/branch updates.

Final source bundle SHA256:
`c343fa83103113fe2ad5e16524f086a773c653a048ccd4ee2112082a790f468c`

Source members: 135. The unchanged scientific freeze is:
`c65026efd643e5be5bc8171ae2e2f1c6716b1063433caec82b19a08031d57dc9`

CPU tests verify serialization/recovery logic and numerical equivalence on miniature models. Actual Colab Drive disconnection behavior, CUDA memory usage and full CLIP execution remain runtime checks. Existing NumPy private-namespace deprecation warnings came from unchanged historical D0 loaders.

[Automatic-resume instructions](clip-source-resume.md) · [New Colab cell](../colabs/clip_source_resume_drive_one_cell.py)
