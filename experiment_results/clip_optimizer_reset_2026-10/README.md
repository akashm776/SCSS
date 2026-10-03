# AdamW optimizer-state reset — completed

Run: `clip_optimizer_reset_20261002T043855_470158Z`.
[Original protocol](../../docs/clip-optimizer-reset.md) · [Completion](completion.json) ·
[Full time course and seed results](comparison.json) · [Export audit](export_audit.json).

Three seeds × two native-only starting ages × two matched streams; retained/reset
AdamW crossed with native/sustained continuation. 3,900 updates and observations
at 0, 1, 5, 10, 25, 50. Retained native/sustained starts, traces and endpoints
reproduce the pinned native-history run exactly.

At horizon 50:

| Starting age | Retained sustained − native loss | Reset sustained − native loss | Reset interaction |
|---|---:|---:|---:|
| 100 | −0.0011155613 | −0.0016122062 | −0.0004966450 |
| 500 | +0.0014907718 | +0.0016371505 | +0.0001463786 |

The primary is the late interaction: **+0.0001463786113727513**. Three of three
seed means and five of six stream pairs are positive. All six early pairs benefit
and all six later pairs worsen under either optimizer condition. No statistical
significance or retrieval-improvement claim.

Reset clears first/second moments and parameter step counters; weights, scheduler,
groups and RNG remain matched. Therefore this is not a momentum-direction-only
test. Persistence of the pattern does not prove representation causation; age-specific
LR, native exposure and weights still differ. Retained replays are not additional
independent evidence, and seeds/streams/report data are reused.

552 original manifested files and 48 endpoint caches verified; 487 retained files
include the original backup manifest, with 66 caption-bearing/binary input files
omitted. Full-input verification rechecks the replay reference, native trunk, paired
invariants, standalone observations and exact summary. Public verification checks
the retained bytes; it cannot recompute omitted NPZ metrics.

The issued protocol's pre-run status is historical. See the
[current scientific report](../../docs/research-status-2026-10-03.md) and
[verification instructions](../../docs/evidence-and-reproduction.md).
