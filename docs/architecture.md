# Architecture hypotheses

**[Home](../README.md) · [Plain explanation](start-here.md) · [Research](research.md) · [Ideas](ideas.md) · [Evaluation](evaluation.md)**

The original mission is to find a genuinely useful new local decision model, not to relabel familiar parts as a new architecture. The strongest remaining technical question is whether a set of independent typed questions about one state can use different exit depths while paying to encode that state only once. The exact novelty gap is unverified.

## Candidate: question exits over one shared state pass

~~~~mermaid
flowchart TD
    D["Document state"] --> E["Advance shared layers once"]
    E --> C["Cached state at layer l"]
    C --> Q["Score active typed questions"]
    Q --> X{"Exit rule passes?"}
    X -->|Yes| F["Freeze this answer"]
    X -->|No| E
    F --> A{"Any active question?"}
    A -->|Yes| E
    A -->|No| P["Return distributions"]
~~~~

For each depth l, compute a state representation or prefix cache $D^{(l)}$ once. Every unfinished question q reads that representation and its own options through a contextual scoring head, producing $p_q^{(l)}$. A question branch stops when its registered output-stability rule passes. Questions that stop no longer consume branch computation. The shared state advances only to serve questions that remain unresolved.

A causal prefix model naturally reuses layerwise state key/value caches. A compact encoder with a separate state path and question cross-attention is another implementation. Choose one only after reading the exact competitor code and measuring an operation-count prototype; do not mix both in the first experiment.

### Stability and calibration

For calibration bundle i, define:

$Z_i = max_{l in E, q in Q_i, c in C_{iq}} |p_{iqc}^{(l)} - p_{iqc}^{(L)}|$

E is the finite set of allowed exit layers; L is the final layer. A conformal quantile of these whole-bundle maxima gives a marginal bound under exchangeability of calibration and deployment bundles. Use the same held-out calibration set for all question types and option counts only if the bundle construction reflects deployment; otherwise stratify or report coverage failures.

At inference, a categorical branch can exit when its current top-two probability margin is greater than twice the calibrated coordinate-drift bound. On the event that every coordinate changes by at most delta, the top class cannot change:

$p_{top}^{(L)} - p_{runner}^{(L)} \geq (p_{top}^{(l)} - p_{runner}^{(l)}) - 2 delta$

This preserves the full model's argmax under the stated event. It does not prove the full model's answer is true, and it does not prove early probability vectors are calibrated. Report both reference agreement and task correctness. For ordinal decisions, bound expected score drift directly; a top-level category guarantee alone is insufficient.

Use intermediate heads trained with a small combination of gold-label loss and distillation from the same model's full-depth outputs. Distillation only teaches the student to approximate its teacher. Keep human/exact labels in the objective where available. Counterfactual fact edits and criterion changes are useful checks, but Nimble already publishes a related data-curation method, so they are not a novelty claim.

## Compute model

Let A(l) be state encoding cost through layer l, B_q(l) question and option cost through l, and H all control, calibration and serving overhead:

$T_{fixed}=A(L)+sum_q B_q(L)$

$T_{adaptive}=A(max_q d_q)+sum_q B_q(d_q)+H$

$T_{separate}=sum_q [A(d_q)+B_q(d_q)]$

The first possible benefit is avoiding repeated state encoding compared with separately running each question. The second is dropping easy question branches before the deepest active layer. The cost of per-layer checks and cache management must be included. If one query requires the last layer, A(max d_q) still reaches L; query-branch savings may remain, but state-layer savings do not. For a single short question, a direct model can remain faster.

Measure tokens-to-output wall time on CPU for state lengths 128/512/2,048, Q=1/5/20 questions and K=2/5/20 options. Record p50/p95/p99, cold and warm start, per-state and per-decision latency, throughput, peak process RSS, and the exit-depth distribution. Test batch 1 and the registered concurrency separately.

## Smallest experiment ladder

| Variant | What it answers | Disposition |
| --- | --- | --- |
| Full-depth typed baseline | Quality and speed reference | Required. |
| Static shallow tap | Does a fixed cut already give most of the speed? | Required cheap control. |
| Per-question early exit without state reuse | Value of adaptive depth alone | Required control. |
| Shared state, fixed depth | Value of state reuse | Required control. |
| Shared state, question exits | Value of the proposed interaction | Run only after a layerwise probe shows headroom. |
| Criterion/evidence factorization | Potential quality gain and compositional transfer | Secondary ablation, after direct overlap checks. |
| Quantized runtime | Compact-device behavior and quality drift | Separate final profile. |

The primary memory gate is full process RSS below 8 GiB. A separate quantized profile must be measured below 4 GiB. Do not equate parameter count, GPU memory or model file size with deployed process memory.

Python/PyTorch remains the research and training path. Start CPU inference comparisons with the checkpoint's supported runtime; use ONNX Runtime for an encoder only when an export has equivalent outputs. A Rust wrapper is warranted only if profiling finds Python overhead large enough to change the end-to-end result.
