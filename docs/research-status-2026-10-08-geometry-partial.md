# D0: verified partial download

The available D0 download is **incomplete**: 166/624 fixed-state batch forwards
(10,624/39,936 query exposures). Only seed 271828 is present. Stream 2026100101
has all 50 batches at both weights; stream 2026100102 has 33/50. The calibration
population and the other two seeds are absent. The user confirms that the runtime stopped or disconnected. The run manifest
remains `running`, and no completion, failure record or error traceback is present.
The run is interrupted; the saved artifacts do not identify the interruption cause.

All 629 downloaded inventory files and 111 source-overlay members pass checksum
verification. Protocol/plan/freeze and native checkpoint/trunk identities match
the frozen references. All 166 saved batch audits preserve state and record zero
training updates/backward passes. Native checkpoint preparation used 500 updates
for this seed. Across saved query records, no invalid mixtures or applicable
I-Synth invariant violations were found. This is artifact verification, not a
rerun of the encoder or independently reconstructed embeddings.

The following descriptive results use **only the complete first stream**:
seed 271828, stream 2026100101, 3,200 matched query exposures at each state.
They are not the completed experiment's seed-balanced estimates.

| Availability | Weights 100 | Weights 500 |
|---|---:|---:|
| No misranked-or-tied candidate | 18.94% (606/3,200) | 26.03% (833/3,200) |
| At least two misranked-or-tied candidates | 69.53% (2,225/3,200) | 58.72% (1,879/3,200) |
| I support at k=8 | 35.09% (1,123/3,200) | 24.72% (791/3,200) |
| C support at k=8 | 99.44% (3,182/3,200) | 99.81% (3,194/3,200) |
| Common C/I eligibility at k=8 | 34.53% (1,105/3,200) | 24.53% (785/3,200) |

Common k=8 C/I eligibility at **both** states is 21.41% (685/3,200).
Transitions are 420 early-only, 100 late-only and 1,995 neither. These are
matched query exposures, not distinct images or independent replicates.

Already-observed queries with no nonpositive candidates rule out the proposed
**all-query in-batch I treatment** on these intended exposures for every k>=1.
A smaller k cannot fix zero availability. This is a structural counterexample;
it does not require the remaining seeds to establish that strict all-query
availability fails. The full audit is still needed to characterize coverage
and freeze an alternative population or candidate-pool design.

Uniform synthesis also substantially changes the side and distance of the
resulting candidate. These are ideal FP32 medians on each regime's own feasible
query population at k=8; C and I rows must not be treated as common-population
comparisons.

| Regime and candidate | Weights 100 | Weights 500 |
|---|---:|---:|
| C-Select margin | +0.001440 (n=3,182) | +0.002105 (n=3,194) |
| C uniform-synthetic margin | −0.026047 (n=3,182) | −0.016443 (n=3,194) |
| I-Select margin | −0.000497 (n=1,123) | −0.000667 (n=791) |
| I uniform-synthetic margin | −0.038125 (n=1,123) | −0.031479 (n=791) |

Although every C source has a positive margin, **96.23% early and 88.29% late**
of its valid synthetic candidates have nonpositive margins. Thus C source
selection does not preserve the correct side in the generated candidate.
I synthesis also moves away from the boundary relative to I-Select, consistent
with the frozen invariant. These constructions cannot be described as delivering
uniformly correct-side or boundary-targeted synthetic supervision.

H and I are not interchangeable on the available I population: mean support
overlap is 28.67% early (n=1,123) and 36.41% late (n=791). H/C overlap is 46.89%
and 56.96%, respectively, on C-feasible queries. Overlap does not establish
an effect on learning.

Same-species candidates are enriched among nonpositive margins but do not dominate
that candidate population: 1.41% early and 1.81% late, versus 0.37% and 0.38%
among positive margins. These pooled candidate-member rates describe association,
not verified semantic false negatives.

**D1 remains unfrozen.** The partial data identify concrete design problems:
all-query I is infeasible, common eligibility changes with state, and uniform
mixing does not maintain the selected sources' boundary geometry. They do not
show which negative helps training, whether synthesis beats selection, or whether
the preferred regime changes with learner state. No reporting loss or retrieval
outcome was measured.

[Compact evidence and reproduction instructions](../experiment_results/clip_geometry_audit_partial_2026-10/README.md).

An [explicit recovery cell](clip-geometry-recovery.md) now verifies and reuses
saved shards in a new folder. It does not alter this partial evidence or the
frozen numerical protocol. Full recovery has not yet been run.
