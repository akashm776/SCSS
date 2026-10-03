# Preserved gated-history policy replication — completed

Run: `clip_fresh_split_policy_20261002T024902_873109Z`.
[Original protocol](../../docs/clip-fresh-split-policy.md) · [Completion](completion.json) ·
[Comparison](comparison.json) · [Export audit](export_audit.json).

Uses the same three seeds and matched data/streams as the native-history study,
but generates starting states under the frozen gated-history policy (150 auxiliary
trunk updates per seed). Four continuation arms, starting updates 100/500,
horizon 50, 3,900 total updates.

Sustained-minus-native full512 loss is −0.0011155613 early and +0.0014949189 later;
6/6 early pairs help and 6/6 later pairs worsen. Gate-minus-random primary is about
−0.00013686 (two of three seed means favorable); later gate vs native harms all six
pairs. Newly excluded reporting images are not historically untouched data.

The original experiment and cell were preserved, not replaced by native-history
training. The [paired-history analysis](../clip_paired_history_comparison_2026-10/README.md)
was frozen before this outcome and after the native outcome. It is exploratory.

374 original manifested files and 48 endpoint caches verified; 309 files retained,
66 caption-bearing/binary input files omitted. Old pre-run status in the issued
protocol is historical; the run is now complete.
