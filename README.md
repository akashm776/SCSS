# State-Conditioned Synthetic Supervision

### When does a generated training signal help a learner?

**A living research report on synthetic-negative supervision in CLIP.**

[Read the report](docs/paper/introduction.md) · [Current progress](docs/current-progress.md) · [Research ledger](docs/experiment-ledger.md) · [Reproduce the evidence](docs/paper/reproducibility.md)

## Abstract

Synthetic supervision is useful only if it improves the learner receiving it. We study this dependence through matched continuations of a pretrained CLIP model on CUB image–text data. History, optimizer-reset, learning-rate, difficulty, and pressure controls show that additional negative supervision can improve early reporting loss while worsening it at later starting states. In the completed source-controlled study, all eight interventions improve early aggregate loss and worsen it late; two-source synthesis beats its corresponding real-negative control early in all four source regimes. However, the proposed source-side curriculum is inconsistent, and early aggregate retrieval changes are negative. These findings motivate direct tests of candidate placement, whose numerical, gradient, and calibration stages have passed but whose full-population audit and training experiment remain pending. The evidence supports conditional loss utility in this development setting, not a general retrieval improvement or a validated curriculum.

![D1: reporting-loss and retrieval effects after 50 updates](docs/figures/d1_research_summary.svg)

**Figure 1.** Effects relative to native training at early and late starting states. Diamonds are equal-seed means; circles are individual seed means, each averaged over two continuation streams. Lower loss and higher R@1 are preferred. Loss improvements need not imply retrieval improvements. [Results and interpretation](docs/paper/results.md) · [Exact values](docs/paper/detailed-results.md).

## Contents

| Section | What it covers |
|---|---|
| [1. Introduction](docs/paper/introduction.md) | Motivation, questions, and the scope of the contribution |
| [2. Related Work](docs/paper/related-work.md) | Recent negative-selection, curriculum, synthesis, and data-utility literature |
| [3. Problem Formulation](docs/paper/formulation.md) | Conditional utility, construction rules, and placement policies |
| [4. Experimental Setup](docs/paper/setup.md) | Data, matched controls, calibration, metrics, and replication units |
| [5. Results](docs/paper/results.md) | Completed findings, including negative results and retrieval discrepancies |
| [6. Discussion](docs/paper/discussion.md) | What the controls resolve and what remains unexplained |
| [7. Limitations](docs/paper/limitations.md) | Reused development data, effect sizes, generalization, and identification |
| [8. Conclusion](docs/paper/conclusion.md) | Supported conclusions and outstanding questions |
| [References](docs/paper/references.md) | Primary sources, emphasizing relevant 2024–2026 work |

### Appendices

[A. Detailed results](docs/paper/detailed-results.md) · [B. Placement validation](docs/paper/numerical-validation.md) · [C. Claim-to-evidence index](docs/paper/evidence.md) · [D. Reproducibility](docs/paper/reproducibility.md)

### Research workspace

[Current progress](docs/current-progress.md) summarizes completed and pending work. The [research ledger](docs/experiment-ledger.md) preserves the study history; the [research agenda](docs/research-program.md) describes the longer-term objective. The [Colab guide](colabs/README.md) separates current validation entry points from historical launchers.

| Location | Purpose |
|---|---|
| [Source](src/) and [CLIP training](model/clip_training.py) | Objectives, constructions, and controlled experiments |
| [Configurations](configs/) and [tests](tests/) | Protocols, references, and implementation checks |
| [Evidence archives](experiment_results/README.md) | Numerical results, audits, and historical snapshots |
| [Scripts](scripts/) | Evidence analysis and reproducible figure generation |

This is ongoing research, not a claim of conference publication. The project was formerly **OTCO**; historical paths retain that name. Optimal-transport weighting was an initial hypothesis, not an established advantage. Technical provenance and hashes are retained in the linked evidence archives.
