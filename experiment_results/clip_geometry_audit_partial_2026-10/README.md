# D0 partial download, 2026-10-08 run

This is **partial evidence**, not the completed D0 audit. Run:
`clip_geometry_audit_20261008T221455_073360Z`.

The downloaded manifest says `running`; the Drive inventory says `in_progress`.
There is no `completion.json`, final summary, final tree-verification marker or
error traceback. The log ends after seed 271828, stream 2026100102, batch 33.
The user subsequently confirmed that the runtime stopped or disconnected.
This is an interrupted run; the snapshot alone does not establish its cause.

Verification: all 629 inventory files match their recorded SHA256 and sizes;
all 111 source-overlay members match the issued local sources; the protocol,
plan, freeze manifest, native starting snapshots and trunk audit match their
frozen references. All 166 saved batch audits preserve state and report zero
optimizer/backward steps. The raw rows contain 10,624 query exposures, zero
invalid mixtures and 41,001 applicable passing I-Synth invariant checks across
k=1–8. These are repeated measurements, not independent statistical replicates.
Native preparation completed 500 updates for one seed; no other seed is present.

`analysis.json` is recomputed from the compressed query records. It separates
all available rows, the one complete stream, and the available primary view.
Every metric includes its available count. The one complete stream contains
3,200 query exposures per state. Conditional source margins count source
members; candidate margins count queries. The groups are overlapping views,
not additional samples. There are no full-grid macro means or utility outcomes.

Original query shards remain in Downloads in the run folder above; they are not
duplicated in this compact archive. `DRIVE_BACKUP_MANIFEST.json` authenticates
that original tree, not the contents of this compact archive. `source_manifest.json`
is the issued overlay manifest. The accompanying `run.log` and manifests are
unaltered copies. Reproduce with:

```bash
python scripts/audit_clip_geometry_partial.py /path/to/clip_geometry_audit_20261008T221455_073360Z --output /tmp/d0_partial_analysis.json
```

The script is specific to this partial run's first seed and complete first stream;
it is not a full-D0 completion verifier. See the
[partial research note](../../docs/research-status-2026-10-08-geometry-partial.md).
