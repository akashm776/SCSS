# D1 Colab dependency-startup repair

Use the entire [repaired one-cell launcher](../colabs/clip_source_resume_bootstrap_drive_one_cell.py)
in the same A100/high-RAM Colab notebook. Replace the previous launch cell, then
run it once. Leave `RESUME_RUN = None` for automatic discovery. Keep the existing
Drive run folders and checkpoints; do not run two notebooks against the same run.

The reported failure occurred at `AutoProcessor.from_pretrained`, before this
attempt reached training. The issued worker imports resume discovery before
running pip. Discovery imports checkpoint helpers, which import the dataset/model
stack and Hugging Face Hub. Pip then replaces installed packages without replacing
modules already held in that worker's memory. The traceback's `httpx` Hub path
after installation of Hub 0.36.2 is consistent with that mixed environment.
The preflight tests ran in a separate fresh process, so their success did not
exercise this startup-order defect.

The repaired launcher installs the same declared dependencies in a bootstrap
process, then uses `exec` to start a fresh Python interpreter for the unchanged
worker. Installation output and worker output share the existing Drive log.
Installation failure stops before the worker starts. The worker retains its
original dependency checks, scientific checks, lease and checkpoint behavior.

This is a delivery repair, not a new scientific source bundle. The embedded
payload, all 135 source files, scientific freeze and resume bundle identity remain
unchanged:

`c343fa83103113fe2ad5e16524f086a773c653a048ccd4ee2112082a790f468c`

Consequently, automatic discovery can select the existing lineage, including
`clip_source_resume_20261015T014921_451960Z`, subject to the usual integrity and
single-writer checks. It reuses whatever authenticated work that lineage contains.
Old unfinished branches without checkpoints still require replay as documented
in the [resume protocol](clip-source-resume.md).

Validation: **18 targeted tests passed**, covering the existing resume suite plus
fresh-interpreter dependency replacement, worker exit propagation, failed-install
handling, and byte-for-byte payload/source preservation. Local import inspection
also confirmed that importing resume discovery loads Hugging Face Hub before pip.
No GPU experiment or live Colab processor download was run during this repair.

Rebuild the repaired delivery with:

```bash
python -m scripts.build_clip_source_resume_bootstrap_cell
```

The builder wraps the issued cell and rejects an unexpected bundle identity;
it does not rebuild the historical source payload.
