# Image-influence diagnostic

Question: Is the fixed-endpoint timing advantage concentrated in a few report pairs?

Hypothesis: A small number of influential queries may account for the net loss advantage.

Experiment: Reuse all 48 verified caches; fixed 32 groups of 16 pairs; no updates or refitting.

Result: gated-minus-random loss -0.000175609385472; favorable/harmful queries 263/249. Paired-pool group deletion: 32/32 favorable; range [-0.000245660029626, -6.97727357393e-05].

Top-k attribution (outcome-ranked, query-only; original candidates retained):

- k=1: gross-benefit share=0.013458195009661434; remaining mean=-0.000135094587435
- k=5: gross-benefit share=0.061881601957659675; remaining mean=1.20106486842e-05
- k=10: gross-benefit share=0.11443422156212867; remaining mean=0.00017453800429
- k=25: gross-benefit share=0.23463755449824236; remaining mean=0.000562829982242

Structural update: Read loss concentration and negative-pool sensitivity separately. Even a stable deletion sign establishes robustness only on these reused endpoints and images.

Next question: Does any timing advantage replicate on genuinely new seeds and untouched reporting data?

Gated minus control. Negative helps. Same three seeds and 512 report pairs. Query-only deletion keeps 512 candidates; paired-pool deletion removes 16 matched pairs from both axes and re-evaluates 496-pair loss. These answer different sensitivity questions. Groups/images are not independent training replications, confidence intervals or significance tests.
