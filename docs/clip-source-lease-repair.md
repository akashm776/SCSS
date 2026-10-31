# D1 lease-read recovery

Use the entire [lease-repaired latest-only cell](../colabs/clip_source_lease_repair_drive_one_cell.py)
after the previous worker has stopped. Keep `RESUME_RUN = None` and
`CLEANUP_ONLY = False` to discover the same lineage and continue from its verified
checkpoint. Keep the active Drive run folder intact. No manual creation or
deletion of `resume_lease.json` is needed.

The October 15 failure occurred while reading `resume_lease.json` before a
checkpoint save. It was a `FileNotFoundError`, not a reported numerical replay
failure. The log does not establish why the file was absent. One possible cause
is a transient mounted-Drive lookup during heartbeat replacement: the issued
code reads the lease in the main thread while another thread periodically
replaces it. External deletion or a mount problem could also cause this error.

The repair serializes heartbeat writes and main-thread ownership checks with a
process-local lock, and retries missing/partially visible lease reads six times
at 0.2-second intervals. It does not recreate a missing lease during training or
ignore changed ownership. A persistent absence still stops the run; another
owner is rejected. A new invocation acquires its lease using the existing local
file-lock and remote stale-heartbeat procedure. Use only one notebook per run;
this remains a mounted-filesystem lease, not a distributed locking service.

The source bundle, checkpoint identity and latest-only retention remain unchanged.
The repair is installed by a separate operational launcher; no issued scientific
source is edited. Provenance is recorded in `resume_lease_policy.json` beside
the lease. The earlier dependency-bootstrap fix remains active.

The pasted log shows all three 500-step native trunks and 90 complete branches.
The first branch of the final stream reached at least update 40 in verified
checkpoints before the save failure. If those files remain intact, approximately
860 training updates remain, plus reporting, diagnostics and final Drive sync.
The actual resume offset is authenticated when loading the Drive checkpoint.

Local tests simulate a missing-file gap during heartbeat replacement, transient
reads, persistent deletion, foreign ownership, stale-lease recovery and a second
local writer. The recovery and latest-only storage regression tests are also run.
No live Colab or Drive repair was executed locally.

Rebuild with `python -m scripts.build_clip_source_lease_repair_cell`.
