# 6. Discussion

[Previous](results.md) · [Contents](../../README.md#contents) · [Next](limitations.md)

## Utility is conditional, but the mechanism remains unresolved

The strongest recurring result is a difference in short-horizon reporting-loss utility across tested starting weights. Prior auxiliary history and retained AdamW state are not necessary for later harm in the controls performed. A modest change in LR trajectory does not reverse the sign. These observations restrict explanations; they do not uniquely identify representation geometry, saturation, or optimizer dynamics as the cause.

## Relative construction gains and absolute benefit are different

Synthesis can outperform a specified real-negative treatment while both remain worse than native. The late M comparison illustrates this directly. A source-regime winner would also require more than a small numerical ordering among means from reused seeds. The appropriate comparisons remain paired effects, per-seed heterogeneity, and absolute native-relative utility.

## Loss and retrieval measure different behavior

Contrastive loss responds continuously to score changes, whereas top-1 retrieval changes only when the winning item changes. This distinction provides a possible explanation for discrepant metrics, but the present experiments do not establish why each observed discrepancy occurs. Both endpoints should remain visible. “Improves training” is too broad when reporting loss improves and R@1 worsens.

## Why a placement experiment follows

D0 establishes that source selection does not fix generated ranking position. D1 establishes an early synthesis advantage over the specified Select controls but no consistent side curriculum. Together these motivate holding source identities fixed while prescribing candidate margins.

The proposed D2 comparison remains a comparison of **state-conditioned placement policies under a common surrogate backward convention and matched initial raw-gradient pressure**. It cannot separately identify boundary specialness, denominator-based weighting, surrogate-gradient effects, and optimizer-mediated effects. Calibration is part of that treatment, not proof that all downstream updates have equal strength.

## Longer-term implications

The findings motivate testing when to supply or withhold generated supervision, rather than presuming that more negative pressure always helps. Establishing a useful system would require new seeds and untouched evaluation, broader tasks, an explicit selection policy, and equal-budget comparisons. Storage savings and learned generator libraries remain research goals, described in the [separate agenda](../research-program.md).

---

[Previous](results.md) · [Contents](../../README.md#contents) · [Next](limitations.md)
