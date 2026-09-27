# Hypothesis ledger: fast local typed decisions

**[Home](../README.md) · [Plain explanation](start-here.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md) · [Research charter](../codex/MISSION.md) · [Original prompt](../codex/ORIGINAL_PROMPT.md)**

This ledger follows the original mission: look for a real architectural contribution, reason about it before implementation, try to falsify it cheaply, and keep the audience split between friendly explanations and expert evidence. The original brief remains verbatim in ORIGINAL_PROMPT.md. The primary process target is below 8 GiB RSS; a separately measured quantized build must fit below 4 GiB.

## Current verdict

**We have not proved a novel architecture exists. We have also not exhausted the search.** A new literature pass found direct typed-decision models and close overlaps with the earlier rubric/evidence proposal. The best remaining experiment candidate is a narrower efficiency question: can each question about one shared state stop at its own model depth, with a calibrated rule for when its answer matches the full-depth model?

That is a falsifiable candidate, not a novelty claim. We should discard it if direct prior art covers the same mechanism, if a small probe shows little early-depth headroom, or if exit checks cost more than the saved inference.

## What the adversarial review removed

| Earlier claim | New direct overlap | Disposition |
| --- | --- | --- |
| Shared state encoding and typed candidate scoring | Laya, Kev, Rev, Typical, Decider, RSI-Jev and Primus Decision all cover substantial parts of this path. Typical uses a shared state cache and contextual candidate scores. | Required baseline, not a novelty claim. |
| Counterfactual evidence pairs teach rule dependence | Nimble explicitly curates near-identical examples where one evidence fact changes and the correct typed decision flips. | Use as a training-data baseline or ablation. |
| Compile rubrics, verify evidence and calibrate typed scores | RULERS studies executable rubric bundles, deterministic evidence checks and post-hoc calibration. | Not a new contribution by itself. |
| Tiny CPU typed decision model | Primus Decision reports a 3.7M-parameter S4D/GRU plus LSA ensemble, fixed supported schemas, and a measured CPU profile. | Strong direct baseline; its schema and state-format limits define one useful boundary. |
| Reusable multi-query memory | KVzip, CacheNotes and ContrastKV study query-independent or task-aware cache/memory reuse. | Generic reusable-memory compression is not a novelty claim. |
| Early exit | CALM, CATs, LEAP and AdaMTL study adaptive depth, early exit or task-aware compute. | Early exit alone is not novel. |

## Candidate to test: independent question exits on one shared state pass

**Plain English.** Read a document once. Ask several typed questions about it. Easy questions may be answered from an earlier layer; harder ones keep using deeper layers. The expensive state is advanced only as far as the hardest unanswered question needs. A calibration rule controls when an early answer is close enough to the full model's answer.

**Technical sketch.** Let D be a state and let q=1…Q be independent questions with candidate sets C_q and output types. At layer l, encode the shared state once and score each active question against that layer's state representation:

$p_q^{(l)} = f_l(D^{(l)}, q, C_q)$

Each question gets an exit depth d_q. Once its exit condition passes, its question branch is removed. The shared state pass continues only while at least one question remains active, stopping at $d_{max}=max_q d_q$. Use question-aware candidate scoring; a frozen label-text embedding is not an acceptable final readout.

~~~~mermaid
flowchart TD
    D["One state"] --> L["Shared layers advance"]
    L --> Q["Score each active question"]
    Q --> G{"Stable at this depth?"}
    G -->|Yes| F["Finalize that question"]
    G -->|No| L
    F --> S{"Any question active?"}
    S -->|Yes| L
    S -->|No| O["Return typed probabilities"]
~~~~

### A testable stability rule

For a calibration set grouped by whole state and its question bundle, record the largest coordinate difference between every intermediate distribution and the full-depth distribution:

$Z_i = max_{l,q,c} |p_{iqc}^{(l)} - p_{iqc}^{(L)}|$

Choose a finite-sample conformal quantile delta at the registered coverage level. If future state bundles are exchangeable with calibration bundles, then with the stated marginal coverage, all tested exit layers and all questions in a bundle have coordinate drift at most delta. For a categorical answer, if the current top-versus-runner-up margin exceeds $2 delta$, its argmax must equal the full-depth argmax:

$p_{top}^{(L)} - p_{runner}^{(L)} \geq margin^{(l)} - 2 delta > 0$

The inequality is exact conditional on the drift bound. The coverage assumption is empirical and distribution-specific; it does not guarantee that the full model is correct. Calibrate separately by question count, type and option count when the workload shifts. For score questions, calibrate expected-score drift as well as per-level probabilities. Probability quality still needs NLL, Brier and reliability tests at each exit depth.

### Why the cost could work, and how it could fail

Let A(l) be the cost of advancing the shared state through l layers and B_q(l) the cost of question q through those layers. Then:

$T_{fixed}=A(L)+sum_q B_q(L)$

$T_{adaptive}=A(max_q d_q)+sum_q B_q(d_q)+H$

$T_{separate}=sum_q [A(d_q)+B_q(d_q)]$

H is the measured overhead of intermediate scoring, calibration and branch management. Relative to separate per-question encoding, reuse saves repeated state work when multiple questions share D. Relative to a fixed-depth shared pass, exits save question-branch work and possibly state layers if every question exits early. With one question, or if one hard question runs to L, the gain may disappear. FLOPs alone are not a speed claim: measure end-to-end CPU time.

## The evidence-factorization idea is now secondary

A compact fact memory with caller-provided criterion atoms and contrastive criterion-to-evidence alignment remains a possible quality ablation. However, compositional rule execution, rubric evidence matching, counterfactual evidence pairs, typed heads and reusable memories all have substantial prior art. Our ContractNLI screens also failed the registered evidence-use gate. Do not lead with this idea or begin training it until the direct competitor and exit-headroom checks are complete.

## Falsifiers and required controls

| Question | Control | Stop condition |
| --- | --- | --- |
| Is the exact exit rule already published? | Search and read early-exit, multi-task adaptive inference, multi-query cache and typed-decision papers and code. | A materially identical method exists; credit it and reframe the project as evaluation or replication. |
| Are predictions available early enough? | Extract per-layer typed scores on train/dev; compare intermediate heads with the final head by question bundle size. | Most questions exit only at the final layer, or error/margin bounds remain too loose. |
| Does calibration generalize to whole bundles? | Hold out source, state template and question composition; calibrate the maximum drift over all questions and exit layers. | Coverage fails by source/type/Q-count, or requires a bound so conservative that no early exits occur. |
| Does adaptive execution save CPU time? | Compare static full depth, fixed intermediate tap, independently processed questions and shared adaptive depth. Include all checks. | No paired p95 improvement at matched quality, or added overhead erases savings. |
| Are probabilities still useful? | Compare per-depth NLL, Brier, calibration plots, score MAE and risk-coverage. | Early distributions fail the pre-registered quality margin even when the top answer is stable. |
| Does it generalize across decision types? | Choice, yes/no, ordinal score, none/abstain, negation, evidence removal, distractor changes and option permutation. | Gains appear only on one synthetic split or fixed schema. |
| Does it meet deployment limits? | Count weights, tokenizer, runtime, state cache, buffers and concurrency in peak process RSS; then measure a separate quantized export. | Full-precision process exceeds 8 GiB or quantized process exceeds 4 GiB. |

## Candidate disposition

- **Test first:** question-specific exits while one shared state representation advances; novelty unresolved.
- **Use as baselines:** fixed shared-state scoring, existing early-exit methods, tiny structured-input models, contextual candidate readout and full-depth inference.
- **Ablate later:** criterion-to-evidence slots, counterfactual rubric pairs, contrastive alignment and set-context scoring.
- **Reject as standalone novelty:** generic shared state, pointer heads, cache compression, contrastive learning, abstention, rule decomposition or early exit.
- **Keep paused:** the earlier ContractNLI coverage-triggered architecture. Its evidence-use gates did not pass.

The concrete execution gates are in [next steps](next-steps.md); the equations and system paths are in [architecture](architecture.md); measured negative results remain in [research](research.md).
