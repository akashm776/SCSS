# CPU-only reporting-image influence diagnostic

Question: Is the small gated-versus-random timing advantage distributed across
the 512 reporting image–caption pairs, or driven by a small favorable subset?

Input is the completed `clip_policy_pool_test_20260930T120952_487061Z` full folder
or single ZIP, including all 48 endpoint NPZ caches. Its manifest and bundle
identities are pinned. No training checkpoints, model downloads, new captions,
gradient probes, gate decisions, or optimizer updates are needed. NumPy only.

## Fixed analysis contract

- Verify every source-file hash, each cache hash and row order; reproduce all 48
  original full-pool losses and R@1 values before reporting influence results.
- Decompose symmetric CLIP loss into 512 text-to-image and 512 image-to-text
  query losses, pairing both directions by their source image–caption pair.
- Compute gated-minus-control contributions for random-matched (primary), native
  and sustained controls, retaining all 12 matched sets and the three-seed
  hierarchy. Save directional, per-seed, early/later and per-stream results.
- Define 32 disjoint groups of 16 using SHA256 of image identity and fixed salt
  `image-influence-v1:2026093004`. Group membership is outcome-independent and
  written before analysis. All arms and matched sets use exactly the same groups.
- **Query-only deletion:** omit a group's queries from the mean, retaining all
  512 candidates and original losses. This tests averaging influence.
- **Paired-pool deletion:** remove those 16 image–caption pairs from both axes
  and recompute symmetric loss over 496 pairs. This also changes negative pools.
  Report every group; do not present it as the same estimand as query-only deletion.
- Describe favorable/harmful query counts, the gross-benefit share of the top
  1/5/10/25 favorable pairs, and query-only deletion of each individual pair.
  Top-k is explicitly outcome-ranked: a post-hoc attribution stress test, not a
  prescription to discard examples or a cleaned-data performance estimate.

Negative contrasts favor gating. A consistently negative group-deletion result
would support local robustness of these endpoints, not independent replication.
Neither groups nor images are independent training seeds; no significance test
or confidence interval is inferred from their counts. All data and model seeds
are reused, and the analysis was selected after aggregate results were seen.

## Run

Local CPU:

```sh
.venv/bin/python -m scripts.analyze_clip_image_influence /path/to/completed_pool_run.zip --output /path/to/new_output
```

Colab: paste the entire `colabs/clip_image_influence_one_cell.py` into a CPU
runtime. Place the source folder or single ZIP in
`MyDrive/OTCO/evaluation_transfer` or `/content`. Wait for `DRIVE_SYNC_COMPLETE`.
The cell includes the code and saves small outputs plus source snapshots and a
console log; it does not copy the embeddings into its output. No Git push needed.

The output refuses overwrite/resume, saves its protocol and group membership,
then writes `analysis.json`, `REPORT.md` and a completion hash inventory. A failure
is never labeled completion. Keep the original source ZIP unchanged.

Validation: independent cross-entropy formula checks, synthetic concentrated and
distributed effects, query-only versus candidate-removing deletion, fixed group
membership, cache corruption rejection and exact delivery-bundle source checks.

## Local result — September 30

Ran on the pinned downloaded ZIP. All 221 source files authenticated; all 48
cached endpoint losses and retrieval values reproduced (maximum absolute loss
error 3.6e-15). Zero optimizer updates. Results are in
`experiment_results/clip_image_influence_2026-09/analysis.json` and `REPORT.md`.

The gated-minus-random symmetric mean is −0.0001756094. Of 512 averaged query
pairs, 263 help and 249 hurt. Both directional means favor gating. The net benefit
survives all 512 single-query exclusions, all 32 query-group exclusions, and all
32 paired-pool group exclusions. Each seed's paired-pool group exclusions is also
uniformly favorable. Later-state and first-stream subgroups have occasional flips.

Excluding the five most favorable query contributions flips the query-only mean
to +0.00001201065. Those five account for only 6.19% of gross benefit: this is a
small net difference between larger opposing contributions, not proof that five
outliers explain nearly all improvements. The outcome-ranked exclusion is not a
fixed-group experiment or a reason to delete examples. Native-control later-state
loss remains unfavorable across all 32 paired-pool deletions.

Structural update: local group-deletion robustness is encouraging, while the
small net advantage remains sensitive to selective outcome-ranked attribution.
Next substantive question is replication on new seeds and untouched report data,
not further selection of favorable images, partitions or gate thresholds here.
