# Research program: State-Conditioned Synthetic Supervision

**SCSS — reusable generators of training signals, evaluated by when they help a learner.**
Updated October 3, 2026. Historical name: OTCO. Repository URL unchanged.

## Motivation: store a way to generate supervision

The motivating idea is to store reusable **functions for producing training
signals**, rather than relying exclusively on a fixed collection of stored examples.
A generator may create examples, embeddings, targets, comparisons or auxiliary
loss terms. Its usefulness is conditional on the learner that receives them.

This is a research objective, not a capability the repository already establishes.
“Less dependent on stored examples” does not mean data-independent: a generator
can depend on real batches, pretrained representations, retained examples or
parameters learned from substantial data. Its provenance and those costs must be
counted. Nor does a compact function necessarily contain new information.

The intended reusable object is the **generation rule**, potentially with learned
parameters, not merely a cache of its outputs. A rule can be evaluated in many
states; the selector or controller decides whether and how to use it. The generator
itself may also be conditioned on state. These are separate design decisions.
State-dependent negative pressure is the first experimental line, not the limit
of the longer-term program.

## Operational question

Write a possible future generator as `z = G_phi(c, s, epsilon)`: context `c`, learner
state `s`, and optional randomness `epsilon` produce a training signal `z`.
Context can include a real batch. `phi` denotes a stored rule or learned parameters;
the notation does not imply that the current experiments train a generator.

Let `a` specify a generator, strength, timing and duration, and let `H` be a fixed
continuation horizon. For a reporting objective that is minimized, define

```text
Delta J_H(s, a) = J_report(theta_H under intervention a from s)
               - J_report(theta_H under matched native continuation from s).
```

Negative is helpful. The contrast depends on the reporting population, evaluation
pool, optimizer and continuation stream as well as `s`, `a` and `H`. Averaging over
specified matched streams estimates a design-specific average, not a universal
property of the generated sample. Retrieval is reported separately from loss.

The measured state includes model weights, optimizer moments/counters, scheduler
position and training history. Training step is a coarse proxy, not a causal
explanation. A future policy `pi(s, meta)` can select an intervention or abstain,
but must not access the reporting outcomes it will subsequently be scored on.

## What exists today

The current instantiation mixes selected mismatched image embeddings into a
synthetic negative for a text query and applies auxiliary contrastive pressure.
The wider codebase compares OT weighting, uniform mixtures and real-negative
controls. Recent diagnostics hold the uniform-top-8 construction and coefficient
fixed to study usefulness across states, horizons and histories.

There is no deployed bank of learned generators, no demonstrated replacement of
a dataset, and no established OT advantage. There is a matched-counterfactual
test harness: restore starting state, use matched data/RNG/LR streams, change the
specified intervention, and evaluate at predetermined offsets. This is the current
contribution under investigation, not a claim of priority over related research.

Five questions must stay separate:

| Question | Evidence needed |
|---|---|
| Does a construction make a negative geometrically harder? | Similarity/margin measurements |
| Is its direction compatible with a specified objective locally? | Appropriate gradient or optimizer-aware probes |
| Does the actual optimizer update help? | Matched realized update contrasts |
| Does sustained use help at a chosen horizon? | Matched continuation trajectories |
| Does a selectable generator improve a useful task or reduce data needs? | Frozen policies, fresh reporting, equal-budget task/data-efficiency comparisons |

Success at an earlier question does not answer a later one. Our loss and retrieval
results already demonstrate why that distinction matters.

## Evidence and limits

The [experiment ledger](experiment-ledger.md) is the complete chronological map;
the [October 3 report](research-status-2026-10-03.md) gives the latest numbers.

Recent native-history continuations reproduce early reporting-loss benefit and
later harm for the same auxiliary rule. Prior gated exposure is not necessary for
that pattern. Clearing AdamW moments and parameter step counters also leaves the
pattern intact. These findings narrow two explanations but do not establish that
representation geometry causes the contrast, or that the same contrast holds for
other generators, objectives, models or datasets.

Historical intermediate checkpoints were non-monotonic and seed-dependent.
Therefore “early good, late bad” is not the proposed universal law or a proven
curriculum. The program asks which state variables matter, and whether usefulness
can be predicted prospectively well enough to improve a meaningful outcome.

## Staged research agenda

The following are **proposals**, not implemented experiments or completed results.
Each requires a separately versioned protocol, fixed budget and acceptance criteria.

1. **Separate state-associated mechanisms.** Cross native-only weights from
   updates 100/500 with common or crossed external LR trajectories, resetting
   AdamW consistently. Match continuation streams; measure offsets 1, 5, 10, 25,
   50. This tests LR/weight-state interactions, not pure “geometry”: the weights
   also encode exposure and other changes. Compare native versus treated within
   every cell rather than ranking raw endpoint losses across ages.
2. **Test a generator family.** Compare synthetic constructions with real-negative,
   random-mixture and no-intervention controls. Freeze candidate rules and strengths
   on development/meta data. The result must distinguish a construction-specific
   artifact from conditional usefulness shared across generation rules.
3. **Predict utility out of sample.** Fit or select state descriptors on meta data,
   then freeze them before evaluating new seeds, streams and reporting examples.
   Include step-only, static best-rule, random timing and abstention baselines.
   Report uncertainty/calibration and harmful selections, not only sign accuracy.
4. **Test reusable generators as a resource.** Store the generator rule/parameters;
   regenerate signals in unseen learning states. Compare against storing generated
   samples and real-example replay at matched bytes, source-data access and compute.
   Include generator training, controller probes and online generation costs.
   State exactly which resource, if any, is reduced.
5. **Test downstream transfer.** Use a new task/model or external dataset, an
   untouched final reporting set, and a fixed task metric. Demonstrate a benefit
   against strong native and data-selection baselines before claiming a generally
   useful training system.

The immediate mechanism study supports the larger program, but cannot substitute
for the resource and transfer tests in steps 4–5.

## What would weaken the thesis?

- If a fixed generator/timing rule matches a state-conditioned selector at equal
  total cost, there is no demonstrated benefit from state conditioning.
- If stored examples outperform generators at matched storage, data access and
  compute, the proposed resource advantage is unsupported.
- If meta-predicted gains disappear on untouched reporting data or new tasks,
  usefulness prediction has not generalized.
- If improvements are confined to the measured contrastive loss without task
  benefit, the system has not demonstrated the intended downstream value.

Negative findings belong in the record. Three training seeds, multiple continuations
and repeated observations are a nested design, not dozens of independent replications.
Report data used to choose a generator becomes development/meta data for that choice;
calling it “held out” does not make it an untouched final test.

## Naming and compatibility

SCSS names a research program, not a finished algorithm or a proven data-compression
method. `OTCO` remains in issued bundle IDs, code identifiers, historical proposals,
Drive paths and original source snapshots. Those records are preserved to make the
scientific evolution explicit. The public repository URL and Python distribution
identifier remain `otco` for compatibility; the landing page and current research
documents use the new name.
