# Native-history policy — completed

Run: `clip_native_history_policy_20261001T234400_383003Z`.
[Original protocol](../../docs/clip-native-history-policy.md) · [Completion](completion.json) ·
[Comparison](comparison.json) · [Export audit](export_audit.json).

Three seeds × two native-only starting ages (100/500) × two continuation streams;
four arms (native, sustained, gated, dose-matched random timing), 3,900 total updates.
At horizon 50 sustained-minus-native loss is −0.0011155613 early and +0.0014907718
later, with 6/6 corresponding pair signs at each age. Prior auxiliary exposure is
not necessary for later harm in this setting.

Gate-minus-random primary is approximately −0.00017777, favorable in two of three
seed means. Later gate-minus-native loss worsens in four of six pairs. A small loss
contrast against random timing does not establish a retrieval-improving policy.
Reporting images are excluded from this run's training/probes, not historically
untouched external evaluation. Three seeds, not twelve independent replications.

377 original manifested files and 48 endpoint caches verified; 312 files retained,
66 caption-bearing/binary input files omitted. Issued protocol's “not run” wording
is historical; this README and completion marker record the completed outcome.
[Latest interpretation](../../docs/research-status-2026-10-03.md).
