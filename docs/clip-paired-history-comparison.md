# Paired treatment-history comparison v1

Status: analysis implemented; the gated-history experiment has not been run here.
This follow-up was selected **after** seeing the native-only results. Its analysis
is frozen before the gated-history results, not before both conditions. It is an
exploratory matched comparison, not independent confirmation or an external test.

## Question

Does prior exposure to the frozen alignment-gated auxiliary treatment change the
response to 50 further sustained auxiliary updates? Native-only history already
showed early improvement and later worsening of full-512 loss. That observation
does not establish the effect of prior auxiliary exposure.

Use the preserved `clip_fresh_split_policy_v1` as the second history condition.
Its training cell, source bundle, parameters, reporting split, seeds, streams and
original policy primary endpoint are unchanged. The native experiment is also
unchanged and **must not be rerun**. Both starting-state histories are native
through update 100; only the parent enables the gate thereafter (refreshes at
100 and 250). This is a history-policy intervention, not a fixed historical dose.

## Frozen paired analysis

At starting update **500**, define each matched seed/stream effect as:

`E(history) = full512_loss(sustained, offset50) - full512_loss(native, offset50)`

The new comparison primary is `E(gated_history) - E(native_history)`, averaging
the two streams within each seed, then the three seeds equally. Positive means
prior gated history makes subsequent treatment less helpful/more harmful;
negative means less harmful/more helpful. Always report both absolute effects.
There are **three paired seed-level units**, not six independent seeds or twelve
independent experiments. No significance test, tuning, image selection or stream
selection is added. The original policy primary remains a separate endpoint.

Report the same interaction for mean bidirectional R@1 separately, both history
effects at update 100, and per-seed early/later interaction. Update 100 is an
**exact matching check**, not another chance to select a favorable primary.
Fail closed on early starting-state, actions, traces, state hashes, feature hashes
or endpoint mismatch; inspect the environment instead of relaxing tolerances.
Probe wall-clock durations and descriptive snapshot metadata need not match.

The analyzer verifies complete Drive manifests and file checksums, pins the known
native run and both source manifests/bundles, verifies protocols and 3,900-update
budgets, checks shared split/probe identities, captions, training plans, RNG/LR
traces, and recomputes all **96** endpoint caches. Full-pool loss arithmetic has
an absolute tolerance of 1e-12 for CPU floating-point summation; state/feature
identity checks are exact. Inputs are read-only; output directories cannot be
overwritten. Incomplete runs or unequal data/environment are not pooled.

Actual historical exposure is reported per seed, including a zero-exposure
condition if the frozen gate never activates. No seeds are removed for this.
This comparison does not separate representation changes from optimizer moments,
nor identify the mechanism of the age contrast. The reporting images remain CUB
images used historically, not a new external or historically untouched test set.

## Run the remaining condition

Paste the entire [paired-history launcher](../colabs/clip_paired_history_drive_one_cell.py)
into an A100 Colab runtime with `torch==2.11.0+cu128` and
`torchvision==0.26.0+cu128` (not cu130). Prefer high RAM; at least 15 GiB runtime
disk is required. It uses the existing experiment's complete preflight tests.

The launcher writes and read-back checks the analysis sources/specification under
`MyDrive/OTCO/evaluation_transfer/clip_paired_history_comparison_v1_plan`, flushes
that plan, and runs the **unchanged** parent cell. An identical frozen plan can be
reused after manual inspection of a preflight failure; a changed plan is rejected.
The parent still blocks blindly repeating an existing run. No automatic retries,
environment replacement, or native-history training are performed.

Budget: **3,900 additional optimizer updates**, not 7,800 new updates. No old
checkpoint ZIP or native-result upload is needed for this training. Do not also
run the original parent cell separately. Outputs retain the parent's prefix:
`clip_fresh_split_policy_<timestamp>` and `clip_fresh_split_policy_logs`.
Wait for `DRIVE_SYNC_COMPLETE`; download the run folder and log. The launcher
explicitly labels the cross-history comparison as still pending CPU analysis.

## Analyze downloaded results

Extract both complete run folders. From the OTCO repository, use:

```sh
.venv/bin/python -m scripts.analyze_clip_paired_history \
  --native /path/to/clip_native_history_policy_20261001T234400_383003Z \
  --gated-history /path/to/clip_fresh_split_policy_TIMESTAMP \
  --output /path/to/new_history_comparison
```

Omit `--gated-history` and `--output` to verify only the native evidence. That mode
never writes a comparison or claims the missing condition completed. The CPU
analysis requires NumPy, not a GPU or model/checkpoint download. Outputs are
`comparison.json` and the frozen `protocol.json`, with source manifest hashes.

Build the launcher with `python -m scripts.build_clip_paired_history_cell`.
The source specification is `configs/clip_paired_history_comparison.json`.
No paired outcome is claimed until both conditions pass all checks.
