# 7. Limitations

[Previous](discussion.md) · [Contents](../../README.md#contents) · [Next](conclusion.md)

**Replication and selection.** The recent studies reuse three development seeds and two nested continuation streams. The reporting pool has influenced subsequent research decisions. It is not an untouched final test, and streams, checkpoints, replayed branches, or query exposures are not independent training-seed replications.

**Effect size and uncertainty.** Many loss differences are small and heterogeneous. We show individual-seed results without treating mean signs as significance tests. No equivalence claim follows from a null or near-zero result. Secondary horizons and comparisons are not substitutes for the frozen primary endpoint.

**Population restriction.** D1 treats approximately 55.56% of scheduled continuation exposures under a common cross-state eligibility mask. Its conclusions concern that masked intervention measured on the full reporting pool. They need not extend to all queries, larger candidate pools, or adaptive source selection.

**Setting and horizon.** Evidence is concentrated in a pretrained CLIP/CUB fine-tuning setting, restricted trainable parameters, and short continuations. Early/late labels describe this schedule. There is no established transfer to other models, tasks, or training scales, and short-horizon loss gains need not survive longer training.

**Task utility.** Aggregate early D1 R@1 changes are negative. A reliable retrieval improvement, a general curriculum, and a benefit over strong methods at matched total compute have not been demonstrated.

**Mechanistic identifiability.** Weight states encode history; source identities differ across starting states; calibrated coefficients differ across arms. Raw-gradient matching does not match clipping or AdamW displacement. These controls narrow hypotheses without fully isolating a causal mechanism.

**Numerics and reproducibility.** D2's FP32 head is a numerical regime change from D1. Its initial preflight PASS statuses do not certify the whole population or future trajectories. Saved-artifact checks verify recorded evidence and selected arithmetic, not an independent GPU replication. Full checkpoint and input archives are required for stronger reproduction.

**Data and pretrained knowledge.** The experiments inherit the data and pretraining history of CLIP and the CUB caption corpus. They do not establish independence from real examples, reduced upstream data cost, or absence of pretraining overlap. Dataset usage conditions and pretrained-model provenance remain relevant to reuse.

**Publication status.** This is a living research report. A paper-style structure does not imply peer review, conference submission, or acceptance. Related-work coverage is focused, and no comprehensive novelty or benchmark-superiority claim is made.

---

[Previous](discussion.md) · [Contents](../../README.md#contents) · [Next](conclusion.md)
