# Fixed-endpoint reporting-pool replay — completed

Run: `clip_policy_pool_test_20260930T120952_487061Z`.
[Protocol](../../docs/clip-policy-pool-test.md) · [Completion](completion.json) ·
[Comparison](comparison.json) · [Export audit](export_audit.json).

Reproduces 48 existing policy endpoints with 2,400 optimizer updates, then evaluates
the full 512-image pool and 32 B64 partitions. Gate-minus-random full-pool loss is
−0.00017560938547234173; 30/32 partition averages favor gating. All three seed means
and both stream means favor gating, but later gating still worsens loss versus native.
This is evaluation robustness on reused endpoints, not independent training replication.

All 221 original manifested files were verified, including 48 NPZ endpoint caches
whose loss and R@1 were recomputed. Public export retains 174 files including the
backup manifest; 48 binary caches are omitted. See [verification scope](../../docs/evidence-and-reproduction.md).
