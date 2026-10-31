# Continue the interrupted D0 audit

Use the entire [recovery Colab cell](../colabs/clip_geometry_recovery_drive_one_cell.py)
in a high-RAM A100 runtime with the same pinned torch 2.11.0+cu128 /
torchvision 0.26.0+cu128 stack. This is a separate, explicit recovery entry point.
The original cell's duplicate-run guard and all original issued sources remain
unchanged. No scientific protocol, source rule, pressure or D1 design is changed.

The cell reads the original run already on Drive:
`clip_geometry_audit_20261008T221455_073360Z`.
It verifies the saved inventory, file hashes, source bundle, protocol/plan/freeze,
native starting-state and trunk identities, batch membership and observation
state-preservation records. It reuses only complete authenticated shards. Partial
uncommitted shards are not reused; changed files, conflicting observations or
recorded scientific failures stop recovery. There is no selection by geometry
or outcomes.

For the inspected download, **166 forwards are reusable and 458 remain**.
If more complete shards reached Drive before disconnection, verification may
recover more; the cell prints the actual counts. The original run folder is
read-only. A new `clip_geometry_recovery_<timestamp>` folder holds the combined
complete dataset, freshly rebuilt summaries, source provenance and per-batch
origin records. It does not count reused batches as new computation or evidence.

The old run saved no model checkpoints. Native starting states therefore need
regeneration and authentication: **1,500 native preparation updates in this
attempt**, including repeating the first seed's 500 updates. These are separate
from the audit, which still performs zero optimizer/backward updates. Final
scientific membership is unchanged: 624 fixed-state forwards / 39,936 exposures.
`recovery_completion.json` records reused and newly computed counts separately;
`completion.json` describes the combined audit and current preparation cost.
The earlier 500 preparation updates remain documented in the source run.

Wait for **DRIVE_SYNC_COMPLETE**, then download the new recovery folder and the
`clip_geometry_recovery_logs` log. No D1 is launched. State feasibility failures
are valid observations, not reasons to change k or replace source candidates.

If a recovery is interrupted again, add that new folder name to `RECOVERY_RUNS`
near the top of this same cell, keeping the original name too. All listed folders
are verified; byte-identical duplicates are deduplicated, conflicting shards are
rejected. The runner detects unlisted prior recovery folders and stops rather
than silently discarding their saved work. Do not delete or rename evidence to
bypass the original duplicate-run guard.

The interruption cause is not established by the downloaded log. This recovery
preserves saved work; it does not claim to fix a diagnosed CUDA/RAM/runtime fault.
Original protocol and numerical sources retain their checksums. The new execution
adapter only adds verification, cache reuse, provenance and separate budget counts.

```bash
python -m scripts.build_clip_geometry_recovery_cell
python -m pytest -q tests/test_clip_geometry_recovery.py
```

Tests cover authentication, modified files and identities, unsafe paths,
incomplete commits, conflicting shards, duplicate consumption, separate budget
accounting and end-to-end toy continuation. The reuse pass prohibits new audit
forwards while rebuilding identical scientific summaries.
