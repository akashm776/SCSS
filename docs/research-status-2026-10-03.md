# Evidence status — October 3, 2026

Current program: [State-Conditioned Synthetic Supervision](research-program.md).
This report supersedes historical “pending” wording in issued protocols without
rewriting their source snapshots. All studies below are completed; no new GPU run
was launched to prepare this update.

## Native history, prior treatment history and optimizer reset

The recent design uses CLIP/CUB-200, three training seeds (271828, 161803, 141421),
two matched continuation streams, starting updates 100/500 and horizon 50. Each
GPU run uses 1,500 trunk plus 2,400 branch updates. The reporting pool contains
512 canonical-caption image pairs excluded from that run's training and probes,
but is not historically untouched external evaluation.

The treatment is sustained uniform-top-8 auxiliary pressure with coefficient
0.06005351588542947. “Native” means the native training objective without that
auxiliary. Loss contrasts below are treatment minus matched native; negative helps.

| Starting update | Native history / retained optimizer | Gated history / retained optimizer | Native history / reset optimizer |
|---|---:|---:|---:|
| 100 | −0.0011155613 | −0.0011155613 | −0.0016122062 |
| 500 | +0.0014907718 | +0.0014949189 | +0.0016371505 |

At horizon 50, all six early pairs benefit and all six later pairs worsen under
each condition. These are repeated, matched conditions—not 18 independent seeds.
The retained reset-study branches replay the native-history branches exactly.

### Prior treatment history

[Comparison results](../experiment_results/clip_paired_history_comparison_2026-10/comparison.json)
compare the native-history run against the preserved gated-history replication.
The comparison plan was saved at **2026-10-02 02:44:04 UTC**, before the gated run
started at 02:49:02 UTC, but after observing the native-history results.

The primary update-500 history interaction is **+0.0000041470**:
gated-history treatment effect minus native-history treatment effect.
Per-seed interactions are +0.0001426526, +0.0001065280, −0.0002367395.
The small mean and mixed signs do **not** establish equivalence. The stronger
observation is that later harm also occurs without prior auxiliary exposure.
The gated trunks applied 150 auxiliary updates per seed; the native trunks applied
none. At update 100, starts and subsequent branches match exactly.

Mean bidirectional reporting-pool R@1 effects at update 500 are +0.048828125 pp
under native history and −0.016276042 pp under gated history; the interaction is
−0.065104167 pp. Loss and retrieval conclusions are distinct.

### AdamW reset

[Reset results](../experiment_results/clip_optimizer_reset_2026-10/comparison.json)
cross retained/reset optimizer with native/sustained continuation. Reset clears
first/second moments and all 35 parameter-state step counters. Weights, parameter
groups, RNG and the age-specific external LR trajectory remain matched.
This is not a momentum-direction-only intervention.

The primary update-500/offset-50 reset interaction is **+0.0001463786**:
reset treatment effect minus retained treatment effect. Positive means more harm
or less benefit. All three seed means are positive; five of six individual
seed-stream interactions are positive. There is no population-significance claim.

| Training seed | Primary reset interaction |
|---|---:|
| 271828 | +0.0001771129 |
| 161803 | +0.0000503088 |
| 141421 | +0.0002117142 |

Fixed time course (seed means after averaging matched streams):

| Starting update | Continuation offset | Retained loss effect | Reset loss effect | Reset − retained |
|---|---:|---:|---:|---:|
| 100 | 1 | −0.0000263754 | −0.0002623125 | −0.0002359371 |
| 100 | 5 | −0.0002805309 | −0.0010759117 | −0.0007953808 |
| 100 | 10 | −0.0005711364 | −0.0012791016 | −0.0007079652 |
| 100 | 25 | −0.0009282301 | −0.0017153875 | −0.0007871575 |
| 100 | 50 | −0.0011155613 | −0.0016122062 | −0.0004966450 |
| 500 | 1 | +0.0000090608 | +0.0000642475 | +0.0000551867 |
| 500 | 5 | +0.0001285381 | +0.0003836386 | +0.0002551005 |
| 500 | 10 | +0.0003683525 | +0.0006022612 | +0.0002339087 |
| 500 | 25 | +0.0011740703 | +0.0012185583 | +0.0000444880 |
| 500 | 50 | +0.0014907718 | +0.0016371505 | +0.0001463786 |

At horizon 50, mean R@1 effects are zero for both early optimizer conditions;
later they are +0.048828125 pp retained and −0.016276042 pp reset. These small,
mixed metric changes do not establish improved downstream performance.

### What follows—and what does not

In this setting, inherited auxiliary exposure and inherited AdamW state are each
unnecessary for the later harmful loss effect. Reset does not eliminate the
early/later contrast. It does not follow that representations are the dominant
cause: starting weights, native exposure and LR trajectories still co-vary by age.
The two ages are not a complete map of learning states.

This supports the broader program's premise that generated supervision must be
evaluated conditionally. It does not show that a generator can replace stored data,
that a selector can predict utility reliably, or that any resource is saved.

## Policy and evaluation robustness

The native-history gate-minus-random full512 loss average is about −0.00017777;
the gated-history replication is about −0.00013686. Both favor gating in two of
three seed means. Later gating still worsens loss versus native training: four
of six pairs under native history and all six under gated history. A favorable
comparison with random timing is not the same as outperforming no intervention.

The earlier [fixed-endpoint pool replay](../experiment_results/clip_policy_pool_test_2026-09/README.md)
has gate-minus-random full512 loss −0.0001756094, with 30/32 B64 partitions favorable.
It replays 48 existing endpoints rather than supplying new independent seeds.
The [image-influence analysis](../experiment_results/clip_image_influence_2026-09/REPORT.md)
preserves the sign under every fixed-group deletion and single-query deletion;
removing the five most favorable query contributions flips its query-only mean.
Those five account for 6.19% of gross benefit, illustrating cancellation sensitivity.
No images were discarded to report a cleaned performance score.

## Verification performed for this release

| Downloaded run | Original manifested files checked | Endpoint caches recomputed | Optimizer updates in original run |
|---|---:|---:|---:|
| Fixed-endpoint pool replay | 221 | 48 | 2,400 |
| Native-history policy | 377 | 48 | 3,900 |
| Gated-history policy | 374 | 48 | 3,900 |
| Optimizer reset | 552 | 48 | 3,900 |

Counts exclude each backup manifest itself. Every original file's SHA256 and size
were checked against its pinned manifest. Cache checks recomputed full512 loss
and both retrieval directions. Paired-history analysis checked matching controls;
reset analysis checked native replay references, traces, starting invariants,
standalone observation records and full summary arithmetic.

Small exports retain numerical results, original manifests and source snapshots;
caption-bearing inputs and binary caches are omitted. Local file verification is
not proof of a remote Drive flush or a new GPU replication. See
[reproduction details](evidence-and-reproduction.md).

## Next proposed step

Cross native early/later weights with controlled LR trajectories under fresh
AdamW, maintaining within-cell native/treated contrasts and fixed timepoints.
This is **not implemented or run** in this release. It is a bounded mechanism
diagnostic; generator-library, data-efficiency and cross-task claims require the
additional experiments in the [research program](research-program.md).
