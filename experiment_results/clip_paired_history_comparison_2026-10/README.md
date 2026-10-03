# Paired-history comparison — completed CPU analysis

[Comparison](comparison.json) · [Protocol](protocol.json) ·
[Original frozen plan](frozen_plan/plan_manifest.json) ·
[Analyzer](../../scripts/analyze_clip_paired_history.py).

The plan was frozen at 2026-10-02 02:44:04 UTC, after native-history results and
before the gated-history run. Seven pinned source files accompany the plan.
Analysis checks 377 native + 374 gated original manifested files, 96 endpoint
caches and the paired controls. No additional optimizer updates.

Primary at start 500 / horizon 50:
`(sustained - native under gated history) - (sustained - native under native history)`
is **+0.000004147033718204796** in full512 loss. Seed interactions are
+0.0001426525735, +0.0001065279897 and −0.0002367394620. The small mean does not
establish equivalence. All six later pairs worsen under either history. The
update-100 comparison is an exact matching control, not independent replication.

The reporting-pool R@1 interaction is −0.065104167 percentage points. This history
comparison does not isolate representation, optimizer or learning-rate mechanisms.
Three paired training seeds; no external test set, policy refitting or new GPU run.

`archive_clip_october_evidence --verify-only` recomputes this comparison from the
public numerical rows and checks the frozen source plan. Full cache verification
requires the original downloaded inputs. [Detailed report](../../docs/research-status-2026-10-03.md).
