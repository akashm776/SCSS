# Completed D0 geometry/feasibility audit

Run: `clip_geometry_recovery_20261009T011659_954111Z`.
Recovery bundle: `92f3431d62b1bb35e9a5740e3da28c92f38a2c1ff3e04dcdde2bbf16e964c254`.

**Completed and verified:** 624 fixed-state forwards, 39,936 query exposures,
three seeds, two continuation streams, two native weight states and the separate
calibration-feasibility population. The 166 authenticated observations from the
interrupted first attempt were reused; 458 were newly computed. Reused rows are
not additional samples. Audit optimizer/backward/scheduler/virtual steps and
reporting forward evaluations are all zero. Recovery regenerated 1,500 native
preparation updates; including the interrupted attempt, 2,000 preparation updates
were executed across attempts. Those repeated native updates are not new evidence.

All 2,668 original inventory files, 118 source-overlay members, frozen input
identities, native start/trunk fingerprints, the full unique batch grid and reused
shard lineage were verified. The final Drive inventory is `verified_tree`.
The download does not contain the recovery's full subprocess log, so the external
`DRIVE_SYNC_COMPLETE` console marker was not independently observed. Nevertheless,
all downloaded files match the final verified inventory and completion records.

`analysis.json` independently aggregates selected measurements from every saved
query record, including feasibility, geometry, semantics and paired eligibility.
636 reported aggregate counts/means/medians and corresponding stream-within-seed
macro means were checked. Individual record checks include source membership,
source signs, overlap arithmetic, k=1 equality, missingness and 156,411 applicable
I-Synth invariants. No invalid mixture was recorded. Embeddings/GPU forwards were
not rerun, and metrics outside that selected set are authenticated by checksum,
not independently recomputed.

Contents:

- `verification.json`: checks performed, execution accounting and limits.
- `analysis.json`: selected recomputed summaries and per-seed extended rates.
- `summary.json.gz`: exact original 143,501,381-byte full JSON summary, losslessly
  compressed; its uncompressed checksum is in the original Drive inventory.
- `REPORT.md`: the original generated report, unchanged.
- Protocol, plan, source/freeze manifests, shard inventory, recovery/completion
  records and the original Drive inventory.
- `retained_manifest.json`: checksums of this compact retained evidence. The
  original `DRIVE_BACKUP_MANIFEST.json` describes the full downloaded run, not
  the compact archive.

The 624 raw query shards, per-batch summaries/audits and full source overlay remain
in the original downloaded run folder/ZIP. They are intentionally not duplicated
here. Full verification requires that original artifact:

```bash
python scripts/archive_clip_geometry_evidence.py /path/to/clip_geometry_recovery_20261009T011659_954111Z
```

[Research interpretation and D1 design implications](../../docs/research-status-2026-10-09-geometry.md).
No utility, loss or retrieval result is claimed. D1 remains unfrozen.
