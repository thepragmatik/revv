# Preregistration: value-of-refresh mechanism screen

**Registered before scoring:** 28 September 2026 (Australia/Sydney)  
**Status:** exploratory, model-free CPU screen; no results yet  
**Navigation:** [benchmark selection](2026-09-27-edit-benchmark-selection.md) · [generator integrity probe](2026-09-27-intervention-mask-probe.md) · [novelty audit](../../docs/novelty-audit.md) · [next steps](../../docs/next-steps.md) · [handoff](../../codex/RESUME.md)

## Question

On a controlled paired-edit workload, can cheap field-level signals rank which **stale cached outputs are wrong after an edit**, so an exact refresher can correct more errors under the same number of verifier calls than random selection?

This is a mechanism screen. It does not test a learned language model, natural language transfer, calibrated expected benefit, measured runtime savings, memory, or architecture novelty.

## Why this is the next cheap test

The existing [intervention-mask fixture](2026-09-27-intervention-mask-probe.md) already supplies original synthetic rule worlds, one-fact edits, exact post-edit labels, proof traces, nested Q=1/5/20 bundles, and composition-held-out groups. Reuse that generator. Do not make a duplicate dataset or use third-party data.

The prior frozen Kaggle screen predicted label flips and never reached inference. This CPU test asks a sharper question using a transparent two-stage proxy: one rule-propagation round supplies a deliberately limited cached predictor; full forward chaining supplies gold and the idealized refresher.

## Frozen method

- Generate 256 states with seed 20260928, eight state variations per rule-composition group, using the existing generator and its stable seen/held-out composition split.
- Evaluate both edit strata separately: **uniform_candidate_sample** and **effective_edit_sample**. The latter is conditioned on a label change and is never used to estimate prevalence.
- Reconstruct each edited state. The shallow predictor sees only direct facts plus one simultaneous rule-propagation round. The cache is its prediction on the original state. Its one-round post-edit output is available as a selector signal.
- Gold output is the existing generator's exact post-edit label. The simulated refresher returns that exact label for a selected field. Therefore it is an **optimistic perfect-verifier ceiling**; it cannot measure real verifier errors.
- Evaluate nested question bundles Q=1, 5, 20 independently. Budgets are k in {0, ceil(Q/4), ceil(Q/2), ceil(3Q/4), Q}, deduplicated.
- Rank fields with:
  1. **cheap_flip:** shallow pre-edit and post-edit labels differ;
  2. **unproved_after:** post-edit shallow output is unknown, used as an uncertainty proxy;
  3. **edit_query_jaccard:** token Jaccard between the edit sentence and question, with only “is” removed;
  4. **random_expected:** analytical expected saved errors kE/Q, where E is the stale error count in that bundle;
  5. **oracle_error:** whether the stale output is wrong against the post-edit gold. This is an upper bound, never a deployable policy.
- Ties use stable field ID order; selectors may read only their named signal. Do not fit thresholds on the held-out groups.
- Primary metric: exact stale errors avoided at each equal refresh-call budget, separately on seen and held-out compositions and by edit stratum and bundle size. Also report baseline stale errors, error fraction avoided, and remaining errors.
- Every selected exact refresh has unit cost. This is a matched-call comparison only; it is not wall-clock timing and ignores shared verifier setup.

For field q, with stale output y0, post-edit gold y1, and perfect refreshed output y1, the simulated signed benefit under zero-one loss is bq = 1[y0 != y1]. This equals the oracle error indicator because the refresher is perfect. That makes the oracle bound exact but makes the experiment optimistic by construction.

## Integrity checks

Before aggregating policy results, verify for every generated state and edit:

- generated records and scores are deterministic for the fixed seed;
- the one-round inference is a subset of full closure and agrees with all direct fact labels;
- pre-edit cache and post-edit gold field IDs match exactly;
- post-edit exact labels match the generator's intervention record;
- all affected-field masks equal exact before/after label differences;
- no parent state or composition group crosses seen/held-out partitions;
- no policy selects more than k fields and oracle saved errors upper-bound every signal policy at the same k.

Save the seed, source hashes, run configuration, aggregate JSON, field-level score JSONL, tests and a friendly methods/results note. Use standard-library Python only; no downloads, model fitting, GPU or paid compute. Local execution budget: five minutes.

## Falsifiers and limits

- If held-out stale errors are absent or negligible, the benchmark has no useful repair headroom; stop this discriminator.
- If simple signals do not beat analytical random at equal calls on held-out compositions, reject them as evidence for selective refresh.
- If a signal only works on the conditioned effective-edit stratum or seen compositions, do not generalize it.
- If the oracle ceiling is small, the repair-value path cannot justify model work on this task distribution.
- A positive result supports only that the signal ranks errors in this synthetic symbolic setup with a perfect verifier. It does not support natural-language quality, model value, latency or novelty.
- Do not launch neural training or another Kaggle kernel from this result alone. A later step needs a distinct imperfect compact predictor/refresher and a real-text paired-edit task with verified rights and labels.
