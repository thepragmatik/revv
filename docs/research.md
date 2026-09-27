# Research landscape and adversarial reading

**[Home](../README.md) · [Plain explanation](start-here.md) · [Idea ledger](ideas.md) · [Architecture](architecture.md) · [Sources](sources.md) · [Next steps](next-steps.md)**

Research snapshot: 27 September 2026. This review follows the original brief's standard: look for new ideas, compare them to primary sources, reason about failure cases, and use a cheap discriminator before spending training compute. The target is local typed text decisions under 8 GiB peak process RSS, plus a separately measured quantized profile under 4 GiB.

## The second overlap review changed the lead

The earlier retained idea—factor caller criteria into atoms, align them to evidence, then compose typed probabilities—was too close to recently found direct work to lead as a novelty claim:

| Newly checked system | Relevant published design or result | Limit that remains useful for our comparison |
| --- | --- | --- |
| [Primus Decision 0.1](sources.md#s46) | A 3.7M-parameter S4D/GRU ensemble plus 142.8 MB of LSA feature tables; CPU-only typed choice, yes/no and score distributions; author-reported 75.1% on the 2,000-decision Typed Decisions test. It reports roughly 0.5–0.6 GB resident and 1.4–1.6 GB peak on a 4-vCPU Xeon. | Supports 20 fixed question schemas across four synthetic workflows, uses teacher-generated labels and accepts structured state. It is a must-run CPU baseline, not proof of arbitrary natural-language rule generalization. |
| [RSI-Jev 1.0](sources.md#s47) | Open 0.8B and 2B Qwen3.5 typed scorers; a learned option cross-attention head jointly sees the option set. The model card reports 0.614 and 0.662 pooled top-1 on Typed Decisions, versus its cited Jev result of 0.727. | Teacher-generated data and a small synthetic benchmark; CPU latency/RSS still need independent matched measurement. |
| [Bespoke Nimble](sources.md#s48) | Uses near-identical typed decision pairs where one relevant evidence fact changes and the answer changes; includes counterfactual criterion/policy use, provenance checks and evidence-removal checks. | The released model is a LoRA adapter on a 9B backbone; the reported train/evaluation pairs are synthetic and model-checked. This is direct prior art for the earlier evidence-pair objective, not a small CPU baseline. |
| [RULERS](sources.md#s49) | Compiles natural-language rubrics into fixed executable criteria, checks evidence deterministically and calibrates scores without updating model parameters. | It is a framework around judge models, not the same compact neural typed readout. |
| [Typical](sources.md#s29) | Reuses a causal state cache, scores contextualized question/options, returns typed probabilities without generation, and fixes a mid-stack readout at about 71% of the backbone. Its own report finds weak gain on one hard unseen rule composition. | Strong direct baseline and a useful full-depth reference for the speed target; its published latency is not our CPU result. |

This makes generic rule decomposition, contrastive evidence pairs, state reuse, candidate pointers, typed heads, cache compression and abstention unavailable as stand-alone novelty claims.

## The remaining experiment candidate

A narrower, possibly useful question survives: **when one state has several independent typed questions, can each question stop at a different layer while all questions share the state computation, with a bundle-calibrated rule that preserves each stopped answer relative to the same full-depth model?**

The idea combines three established lines, so novelty is still uncertain:

- Typed models such as Typical reuse a single state representation across multiple question suffixes.
- Early-exit methods such as CATs, CALM and LEAP allocate different depth by input or token; AdaMTL learns task-aware compute policies on a shared backbone.
- Conformal methods can calibrate a high-confidence consistency or error bound, but the chosen guarantee applies to the model comparison and workload distribution, not to external truth.

The potential contribution is the *specific measured interaction*: variable exit depth per request-time question, shared state computation, and a simultaneous calibration bound over every question and exit layer in one request bundle. A targeted search found close components but no exact paper in this combination. Search coverage is not exhaustive, so this is a search target—not a novelty result.

The mechanism could reduce repeated state-prefill cost relative to processing each question independently, and skip later question-branch work relative to a fixed-depth model. It cannot correct a full-depth model error. If one hard question forces the state to full depth, only the easier branches' work may be saved. A single-question workload has little or no amortization.

## Mathematical screen and its limits

For a state/question bundle i, let $p_{iqc}^{(l)}$ be the probability for candidate c at layer l. On a calibration set, compute the maximum coordinate drift over allowed exit layers, all questions and all candidates. A finite-sample quantile can bound that maximum on an exchangeable deployment bundle. For a categorical decision, a current top-two margin greater than twice the drift bound guarantees that the intermediate argmax equals the full-depth argmax on that event.

This proof is exact *conditional on the calibrated drift event*. It does not guarantee semantic correctness, probability calibration against ground truth, coverage under source shift, or stable expected score unless those outputs are separately covered. Stratify or re-calibrate by question count/type and option count when needed. If the bound is too wide, the system falls back to deeper inference and the speed hypothesis loses.

Compute the candidate against three controls:

$T_{fixed}=A(L)+sum_q B_q(L)$

$T_{adaptive}=A(max_q d_q)+sum_q B_q(d_q)+H$

$T_{separate}=sum_q[A(d_q)+B_q(d_q)]$

A is state work through depth, B_q is question/option work, H is exit and cache-control overhead. Measure full tokenization-to-output CPU time; this equation is a screening model, not a latency result.

## What our own experiments do and do not show

- CLINC prototype screen: a frozen MiniLM with labeled intent prototypes reached 91.12% known closed accuracy and 86% OOS recall on a public development fold. It is neither a typed multi-question benchmark nor a matched local CPU comparison.
- ContractNLI evidence prototype: training evidence retrieval improved positive any-gold@5 to 95.11%, but all marked spans were retrieved at five for only 63.52%. Improved retrieval did not produce a clear stance gain.
- Matched ContractNLI heads: pooled evidence reached 70.20% accuracy versus 68.08% for a hypothesis-only prior, with the paired interval including no gain. A frozen NLI scorer did not beat the matched lexical result.
- Evidence removal and shuffle: the pre-registered evidence-use effect-size gates failed or remained inconclusive. The gold-span diagnostic was not deployable and did not establish a sufficient stance ceiling.
- Synthetic T4 forward pilot: shared execution won for four of five tested shapes and direct scoring won for one. This was not real model quality or CPU latency.

These are useful negative results. They do not support another evidence-retrieval training run. Details, hashes and limitations are preserved in the [Codex handoff](../codex/RESUME.md) and linked run records.

## Candidate board

| Idea | Prior-art distance | Evidence | Decision |
| --- | --- | --- | --- |
| Fixed shared state plus typed options | Low | Multiple direct competitors | Baseline. |
| Tiny non-transformer structured decision model | Low | Primus Decision directly matches the size/CPU goal, with fixed schema limitations | Run as a baseline; learn from its weak spots. |
| Counterfactual evidence/criterion training | Low | Nimble directly describes this curation | Reproduce or ablate, never claim as new. |
| Fact–criterion evidence memory | Low to unclear | Adjacent rubric, evidence, compositional and retrieval work; project evidence-use gates failed | Secondary quality ablation only. |
| Per-question adaptive depth with one shared state pass and bundle-level calibration | Unclear / potentially narrow | Components are prior art; exact typed multi-question intersection not yet exhaustively searched or measured | Cheapest remaining architecture question to test. |

## What happens next

1. Finish the exact prior-art and code audit, especially Primus, RSI-Jev, Typical, Nimble, CATs/CALM/LEAP and AdaMTL.
2. Pin one direct local CPU comparison path and audit typed, composition and natural-text data without touching locked labels.
3. Run a no-training layerwise probe: measure how many questions could exit early, how the bundle drift grows with Q, and whether probabilities remain useful.
4. Implement shared adaptive execution only if the probe has non-vacuous calibrated exit headroom.
5. Run matched CPU baselines; unlock one short free-Kaggle pilot only if a trainable gap remains.
6. Quantize and red-team only after a full-precision candidate passes.

The detailed stop gates are in [next steps](next-steps.md); the comparison metrics and memory gates are in [evaluation](evaluation.md).
