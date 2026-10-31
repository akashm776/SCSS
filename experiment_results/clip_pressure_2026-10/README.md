# Initial-pressure calibration experiment C — completed

Run: `clip_pressure_20261007T235549_427102Z`.
[Frozen protocol](../../docs/clip-pressure.md) · [Completion](completion.json) ·
[Full comparisons](comparison.json) · [Derived diagnostics](diagnostics.json) ·
[Export audit](export_audit.json) · [Scientific report](../../docs/research-status-2026-10-08-pressure.md).

Seven arms, native weights 100/500, common early LR and fresh AdamW. Four training-only
calibration batches per seed target initial mean auxiliary/native gradient ratios
of 2% and 7%; coefficients stay fixed. Three reused seeds × two streams.
**5,700 optimizer updates; 84 branches/caches, 504 reports, 336 pressure probes.**

Horizon-50 arm minus native loss (negative helps):

| Intervention | Early weights | Later weights |
|---|---:|---:|
| Hard, 2% | −0.0009868904 | +0.0003929658 |
| Semi-hard, 2% | −0.0011100267 | +0.0003236494 |
| Hard, 7% | −0.0018044836 | +0.0017727088 |
| Semi-hard, 7% | −0.0023160732 | +0.0010872379 |

Primary mean later semi-minus-hard: **−0.0003773937**, negative in 3/3 seed means
and 5/6 stream contrasts. At 7%, semi-hard wins all six late comparisons; at 2%,
three of six. Construction × pressure is −0.0006161545 and negative in all six.
Every calibrated later arm still harms on average; semi-hard-low helps in only
one individual continuation. Semi-hard is already better early on average at
both targets. Relative weight-state interactions are mixed; no directional
hard-early/semi-hard-later curriculum or reliable retrieval win is established.

Calibration max relative mean error: 0.1377%. Initial continuation pressure is
closely matched, but later semi-hard pressure declines below hard. Clipping is
active on all 4,200 branch updates. This controls an initial raw norm statistic,
not optimizer updates or entire pressure trajectories.

1,322 manifested files verified; 84 cached endpoint losses and retrieval metrics
recomputed. 1,224 original files retained, 99 caption-bearing/binary inputs omitted.
Clone-only checks verify retained bytes, calibration/replay/clipping records and
independent contrast arithmetic. Gradient records cannot be reconstructed from
endpoint caches; no new GPU run or independent scientific replication occurred.

```bash
python -m scripts.archive_clip_pressure_evidence --verify-only
python -m scripts.archive_clip_pressure_evidence \
  --downloads /path/to/downloads --output-root /path/to/new-export
```

Export refuses to overwrite evidence. Historical native/hard/semi-hard branches
replay B and are not new evidence. The issued protocol, source snapshots and
original preparation-time status remain frozen.
