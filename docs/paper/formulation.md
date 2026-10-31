# 3. Problem Formulation

[Previous](related-work.md) · [Contents](../../README.md#contents) · [Next](setup.md)

## Conditional intervention utility

Let a learner state $s$ include model parameters, optimizer state, schedule position, and relevant training history. An intervention $a$ specifies the source rule, candidate construction, strength, and exposure policy. For a continuation horizon $H$, define

$$D_H(s,a)=J(\theta_H^{s,a})-J(\theta_H^{s,N}),$$

where $N$ denotes matched native training and $J$ is the frozen reporting loss. Negative $D_H$ favors the intervention. This is a finite-horizon policy comparison, not a statement about asymptotic performance.

## Native and auxiliary objectives

For normalized text and image embeddings $q_i,v_j$ and positive logit scale $\tau$, write $\ell_{ij}=\tau q_i^\top v_j$. Native training minimizes the mean of row-wise and column-wise cross-entropies on the full batch.

For an auxiliary candidate $z_i$, define

$$e_i=\tau q_i^\top z_i,\qquad A_i=\log\sum_j\exp(\tau q_i^\top v_j).$$

With a frozen eligible-query set $E$, the auxiliary objective is

$$L_A=\frac{1}{|E|}\sum_{i\in E}\operatorname{softplus}(e_i-A_i),\qquad L=L_N+\alpha L_A.$$

All native pairs and all denominator columns remain present. The auxiliary scale is detached from the learned scale parameter; candidate and native-partition features retain their gradient paths. Empty eligibility, where allowed by the protocol, gives zero auxiliary loss.

## Source selection and synthesis

D1 freezes two source identities per eligible exposure at each starting state. Select applies extra pressure to the first specified real source. Synth normalizes the equally weighted sum of the two current source embeddings. Embeddings evolve during continuation; source identities and the eligibility mask do not.

| Regime | Starting-state source rule |
|---|---|
| H | Two highest-similarity negatives |
| M | Negative ranks 16–17 by similarity |
| C | Two smallest strictly positive positive-minus-negative margins |
| I | Two smallest absolute nonpositive margins, including ties |

C and I describe initial **source** margins. They do not guarantee the side on which the generated candidate lies, or the sources' side after learning.

## Initial pressure calibration

For four scheduled training-only calibration batches, the coefficient is

$$\alpha_a=\rho\left(\frac14\sum_{b=1}^4\frac{\|g_{A,a}^{(b)}\|}{\|g_N^{(b)}\|}\right)^{-1}.$$

D1 uses $\rho=0.02$. The coefficient is verified with weighted gradients, then fixed during continuation and shared across streams within seed/state/arm. This matches an empirical initial mean raw-gradient ratio, not per-query pressure, future gradients, clipping, or AdamW displacement.

## Prespecified D1 contrast

At $H=50$, let $Q(w)=D(C\text{-Select},w)-D(I\text{-Select},w)$. The primary contrast is

$$\Delta_{\mathrm{side}}=Q(500)-Q(100).$$

A negative value indicates a relative shift toward C. A directional I-early/C-late crossover additionally requires $Q(100)>0$ and $Q(500)<0$. Absolute benefit still requires $D<0$. Synth-minus-Select comparisons within regimes are secondary.

## Placement follow-up: not yet a utility result

D2 proposes prescribed pairwise margins $r=q^\top v^+-q^\top z\in\{-\delta,0,+\delta\}$. It uses a detached geometric target and $z=\operatorname{sg}(z^\star)+(z_0-\operatorname{sg}(z_0))$, giving an identity local Jacobian to the live carrier $z_0$.

This holds a surrogate backward convention fixed, not the full training gradient. Placement changes competitiveness, loss weighting, query gradients, and subsequent trajectories. A zero-margin advantage would not by itself establish a uniquely special boundary mechanism. The [current progress page](../current-progress.md) distinguishes passed validation stages from the unperformed utility experiment.

---

[Previous](related-work.md) · [Contents](../../README.md#contents) · [Next](setup.md)
