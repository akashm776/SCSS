# 2. Related Work

[Previous](introduction.md) · [Contents](../../README.md#contents) · [Next](formulation.md)

This focused review emphasizes relevant 2024–2026 work, with earlier references retained for the model, construction, and optimization foundations. These are literature comparisons, not methods independently rerun in this repository. The experimental evidence does not establish benchmark superiority over them.

## 2.1 Negative selection and false negatives

**FALCON** learns a minibatch scheduler that selects negative hardness per anchor using a proxy for cross-modal alignment improvement. It explicitly addresses the tension between hard and false negatives. This is close to our motivation for state-dependent supervision, but our completed studies use frozen construction rules and matched continuations rather than a learned batch scheduler. [Kim et al., CVPR 2026; author manuscript v5](https://arxiv.org/abs/2505.11192v5).

**HOBIT** optimizes minibatch composition for informative in-batch negatives, with a submodular objective and greedy construction. Its intervention changes which real examples co-occur. Our source study keeps continuation batches fixed and adds a masked auxiliary negative. Thus it asks a different controlled question; it does not establish that auxiliary synthesis is preferable to improved batching. [Dutta et al., ICML 2026](https://proceedings.mlr.press/v306/dutta26a.html).

**MAFA** combines connection mining and contrastive label smoothing to address missing positive connections in vision–language data. This highlights an unresolved issue in our setting: a negative's ranking margin does not certify that it is semantically negative. Our geometry audit measures ranking, not false-negative labels. [Byun et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Byun_MAFA_Managing_False_Negatives_for_Vision-Language_Pre-training_CVPR_2024_paper.html).

## 2.2 Boundary-aware curricula and synthetic geometry

**BACL** combines progressively harder boundary-aware negative sampling with a local-attention contrastive loss. It is directly relevant to the proposed placement study, and prevents a broad novelty claim that boundary-sensitive multimodal curricula are unexplored. Our planned D2 manipulation is narrower: prescribe three pairwise cosine margins for the same frozen source identities under a common straight-through backward convention. D2 currently has validation evidence, not continuation results. [Ye et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8d00d465737c3bee5448f983dea88fd2-Abstract-Conference.html).

**Geodesic Multi-Modal Mixup** constructs mixed image–text embeddings on the hypersphere for robust fine-tuning. It is a closer multimodal construction reference than generic image augmentation. D1 instead mixes two image-source embeddings and compares that construction with a specified real-negative control at two starting states. [Oh et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a45296e83b19f656392e0130d9e53cb1-Abstract-Conference.html).

Earlier foundations include feature-level hard-negative mixing in [MoCHi (Kalantidis et al., 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f7cade80b7cc92b991cf4d2806d6bd78-Abstract.html), controllable hard-negative sampling in [Robinson et al. (2021)](https://openreview.net/pdf?id=CR1XOQ0UTh-), and example ordering in [curriculum learning (Bengio et al., 2009)](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf). Our evidence does not support a universal easy-to-hard, hard-to-easy, or source-side curriculum.

## 2.3 Data utility, learner state, and optimization

**LESS** uses optimizer-aware gradient features for targeted instruction-data selection, adapting influence estimation to Adam. It motivates evaluating useful training signals in relation to the optimizer and target task, rather than equating raw gradient alignment with realized utility. Its language-model instruction-tuning results do not validate our contrastive generator or our prediction rules. [Xia et al., ICML 2024](https://proceedings.mlr.press/v235/xia24c.html).

**BLISS** learns data scores through a bilevel objective and multiple proxy-model updates, addressing evolving data preference during language-model pretraining. It is relevant to the broader SCSS agenda of conditional supervision utility. Our current experiments neither learn a scoring model nor demonstrate transfer from a proxy; they measure realized effects of fixed interventions. [Hao et al., ICML 2026](https://proceedings.mlr.press/v306/hao26b.html).

[AdamW (Loshchilov and Hutter, 2019)](https://arxiv.org/abs/1711.05101) provides the optimizer used in the continuation studies. Our reset controls and displacement probes distinguish initial raw-gradient pressure from the resulting adaptive update. [Straight-through estimators (Bengio, Léonard, and Courville, 2013)](https://arxiv.org/abs/1308.3432) provide context for D2's surrogate derivative, not evidence that it approximates our constrained generator's true derivative or improves training.

## 2.4 What this report contributes to the comparison

| Literature direction | Main intervention | Question addressed here |
|---|---|---|
| FALCON / HOBIT / MAFA | Negative selection, batch construction, or treatment of false negatives | When does added negative pressure help or harm under matched continuations? |
| BACL | Boundary-aware sampling plus a curriculum and local loss | Can fixed-source pairwise-margin policies differ in utility? **Pending D2.** |
| Geodesic Mixup / MoCHi | Feature-space construction | Does two-source synthesis beat its specified real-source control at early and late states? |
| LESS / BLISS | Estimated utility for data selection | How do measured intervention effects depend on state, horizon, and optimizer behavior? |

These distinctions locate the study; they do not establish priority or isolate a new general mechanism. The clearest completed contribution is a set of controlled empirical findings, including adverse and inconsistent outcomes. Stronger comparisons would require implementing relevant contemporary baselines at matched data and compute budgets.

## 2.5 Model and data foundations

The test bed uses pretrained [CLIP (Radford et al., 2021)](https://proceedings.mlr.press/v139/radford21a.html), [CUB-200-2011 (Wah et al., 2011)](https://data.caltech.edu/records/65de6-vp158), and the fine-grained visual descriptions associated with [Reed et al. (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf). Restricted fine-tuning and small-pool instance retrieval do not reproduce large-scale CLIP pretraining or establish broad transfer gains.

*Literature checked October 31, 2026. Shared research discussions and local literature-review PDFs informed discovery; bibliographic entries and method descriptions link to original papers or proceedings. Internal reviews are not treated as independent scholarly evidence.*

---

[Previous](introduction.md) · [Contents](../../README.md#contents) · [Next](formulation.md)
