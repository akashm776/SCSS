# Weight-state × LR experiment — completed

Run: `clip_weight_lr_20261007T024147_218129Z`.
[Frozen protocol](../../docs/clip-weight-lr.md) · [Completion](completion.json) ·
[Full timecourse and seed results](comparison.json) · [Export audit](export_audit.json) ·
[Scientific report](../../docs/research-status-2026-10-07.md).

Three training seeds, two streams, weights 100/500 × LR trajectories 100/500 ×
native/sustained uniform-top-8; all branches use fresh AdamW moments/counters and
common continuation batches/captions/RNG. 3,900 updates, 48 branches, 288 observations.

Horizon-50 sustained minus native loss (negative helps):

| Weights | Early LR | Later LR |
|---|---:|---:|
| 100 | −0.0016122062 | −0.0015408524 |
| 500 | +0.0017580303 | +0.0017639800 |

All six seed–stream pairs share each cell's sign. Primary weight main effect
+0.0033375344; LR main effect +0.0000386518; interaction −0.0000654041. Effects
follow weights under both tested schedule segments; the LR difference is only
about 3%. No general LR-irrelevance, representation-causation or retrieval-win
claim. Three reused seeds and reporting pool; early diagonal is historical replay.

All 548 original manifested files and 48 endpoint caches were verified before
filtering. 489 retained files include the original backup manifest; 60 inputs
containing captions or binary arrays are omitted. The verifier recomputes every
seed/timepoint factorial contrast and checks historical replay and matching records.
It does not rerun training or prove a remote Drive flush.

```bash
python -m scripts.archive_clip_weight_lr_evidence --verify-only
# Full download: verify caches and export to a new, empty destination.
python -m scripts.archive_clip_weight_lr_evidence \
  --downloads /path/to/downloads --output-root /path/to/new-export
```

The issued protocol and source overlay retain their preparation-time status;
the run is completed as recorded above. Do not rebuild its historical bundle to
change that status. Full-download checks need the original extracted folder or ZIP;
the public export cannot recompute omitted feature caches.
