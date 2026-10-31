# Appendix D. Reproducibility

[Contents](../../README.md#contents) · [Evidence index](evidence.md) · [Colab guide](../../colabs/README.md)

## D.1 Evaluate the findings without a GPU

Read the [results](results.md), [detailed seed-level tables](detailed-results.md), and [claim-to-evidence index](evidence.md). The public archives retain numerical summaries and provenance; complete checkpoint tensors and raw image/caption inputs are not included.

Regenerate Figure 1 and the D1 tables from the repository root:

```bash
uv run --with matplotlib python scripts/build_research_summary.py
```

This checks table arithmetic and redraws retained results. It is not experimental replication.

## D.2 Verify retained evidence

The [technical reproduction guide](../evidence-and-reproduction.md) provides archive verification commands and explains what each verifies. Hashes identify recorded artifacts; they do not prove independent reproduction or validate a scientific conclusion.

Saved-result checks can authenticate recorded acceptance evidence and recompute ratios or contrasts. Recomputing endpoint loss/R@1 requires the original full result archives; recomputing gradients requires the appropriate model, data, and GPU execution.

## D.3 Reproduce experiments

Use the exact study's [protocol and issued launcher](../../colabs/README.md). The generic local environment is for development, not historical bitwise replay. The D2 Python 3.13.16 amendment is explicit; it does not retroactively change D1's recorded environment.

Reproduction requires the original data/model access, full starting-state archives where specified, pinned dependencies, precision settings, and recorded schedules. Follow the protocol's stage gates: passing a batch-level numerical diagnostic does not authorize full D2 training. The full-population D2-P v2 audit remains pending.

## D.4 What is preserved

Historical launchers, frozen configuration files, and source snapshots remain in their original paths. They record which computation produced each result and should not be interpreted as equally current recommended entry points. The curated Colab guide supplies that distinction.

The [research ledger](../experiment-ledger.md) includes failed and partial runs as well as successful ones. A revised protocol produces new evidence; it does not overwrite a failed version. Operational recovery, acceptance failure, invalid provenance, and scientific utility are reported separately.
