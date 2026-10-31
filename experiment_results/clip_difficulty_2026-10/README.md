# Difficulty-regime experiment B — completed

Run: `clip_difficulty_20261007T114311_800923Z`.
[Frozen protocol](../../docs/clip-difficulty.md) · [Completion](completion.json) ·
[Full comparisons](comparison.json) · [Derived diagnostics](diagnostics.json) ·
[Export audit](export_audit.json) · [Scientific report](../../docs/research-status-2026-10-07-difficulty.md).

Five arms, native weights 100/500, common early LR and fresh AdamW. Three reused
training seeds × two streams; matched continuation inputs/RNG. 4,500 optimizer
updates, 60 branches/caches, 360 reports and 180 observational pressure probes.

Horizon-50 arm minus native loss (negative helps):

| Intervention | Early weights | Later weights |
|---|---:|---:|
| Hard synthetic | −0.0016122062 | +0.0017580303 |
| Semi-hard synthetic | −0.0018507969 | +0.0004180784 |
| Hard real | −0.0001464353 | +0.0001972352 |
| Semi-hard real | −0.0002021395 | +0.0003295696 |

Primary synthetic regime interaction: **−0.0011013612**, negative in all three
seed means and six stream contrasts. Semi-hard synthetic is less harmful later
in all six pairs, but remains harmful in all six. The aggregate horizon-50 result
is not the predicted crossover: semi-hard is already better on average early.
Shorter secondary horizons do show an aggregate crossover. Real controls have
the opposite interaction. Semi-hard has substantially lower gradient pressure;
no pressure-independent geometry explanation or retrieval gain is established.

902 manifested files verified; 60 cached endpoints recomputed. 831 files retained,
72 caption-bearing/binary inputs omitted. Clone-only verification checks retained
replay/matching/support/pressure records, all seed/timepoint arithmetic and derived
diagnostics; omitted NPZ metrics require the full download. No GPU training is run.

```bash
python -m scripts.archive_clip_difficulty_evidence --verify-only
python -m scripts.archive_clip_difficulty_evidence \
  --downloads /path/to/downloads --output-root /path/to/new-export
```

Export refuses to overwrite existing evidence. Original protocol and source
snapshots remain frozen; their preparation-time status is superseded by this
completed record. Replayed native/hard branches are not independent new evidence.
