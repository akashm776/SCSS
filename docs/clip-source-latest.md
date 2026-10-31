# D1: continue with latest state, without checkpoint history

Stop the currently running Colab cell first. Then replace it with the entire
[latest-only cell](../colabs/clip_source_latest_drive_one_cell.py) and run it.
Leave `RESUME_RUN = None` to find the existing run automatically. The cell first
cleans old resume blobs, prints `D1_STORAGE_CLEANUP` with the number of files and
GiB removed, then continues through the usual checks and resume path.

To reclaim space and remain stopped, set `CLEANUP_ONLY = True` in that cell.
No GPU training is started in that mode. Later set it back to `False` and rerun
to continue. Cleanup uses the existing single-writer lease; an active worker
blocks it. A disconnected worker's stale lease may take up to three minutes
to expire.

Storage now retains **one verified generation per required phase**, replacing
the original two-generation retention. It removes older generations and
unreferenced checkpoint blobs/abandoned upload files only inside this run's
`resume_state` directory. It verifies all latest checkpoint hashes and contracts
before deleting any checkpoint history. Results, feature caches, logs, scientific
sources and other run folders are preserved.

There is still a latest checkpoint: it is what makes continuation possible after
a runtime stops. The issued runner also needs saved native starting states at
weights 100 and 500 and per-stream progress. This amendment keeps those required
states; it does not reduce the whole experiment to one model file. Completed
stream checkpoints contain result records rather than an active model state.

The cadence remains every 25 native updates / 10 branch updates and branch
completion. On each save, the previous latest remains until the replacement
blob and pointer verify, then it is deleted. One extra checkpoint's worth of
temporary free space is therefore still needed while publishing a replacement.
If the prior cell stopped between saves, continuation replays the unsaved suffix.

This is an explicitly recorded storage amendment. The original 135-file source
payload and bundle identity remain unchanged so existing checkpoints still match:

`c343fa83103113fe2ad5e16524f086a773c653a048ccd4ee2112082a790f468c`

The added launcher installs a checkpoint-store subclass, which delegates each
save to the issued transaction before pruning retention. It does not modify
training payloads, RNG, objectives, source identities, coefficients or scientific
analysis. The storage amendment and implementation hash are saved as
`resume_state/latest_retention_policy.json`; cleanup events are also logged.

The previous bootstrap fix remains in place: dependencies are installed before
the fresh worker imports the model stack. Do not use the older launch cells for
future resumes if you want to keep latest-only retention.

Local validation covers corrupt-latest rejection before any deletion, contract
and unsafe-path rejection, preservation of result files, interrupted publication,
failed uploads, repeated cleanup, unchanged embedded sources, and exact agreement
between interrupted/resumed and uninterrupted toy-model branch results. Actual
Drive space reclaimed is printed by the Colab cleanup, not estimated locally.

Build with `python -m scripts.build_clip_source_latest_cell`.
