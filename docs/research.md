# Research landscape and adversarial reading

**[Home](../README.md) · [Plain-English view](start-here.md) · [Idea ledger](ideas.md) · [Architecture](architecture.md) · [Novelty audit](novelty-audit.md) · [Sources](sources.md) · [Next steps](next-steps.md)**

Research snapshot: 27 September 2026. The charter is to seek a meaningful new combination, check nearest prior art, derive its cost, and red-team it before expensive experiments. The target is local typed text decisions under 8 GiB peak process RSS and a separately measured quantized profile under 4 GiB.

## Third overlap pass: the lead narrows again

The previous idea—independent question exits on one shared state pass with a bundle stability check—looked narrow after review of typed models and generic exits. A second and third targeted pass found direct shared-context, task-aware routing, QA-cascade and counterfactual work:

| Work | Overlap found | Boundary that remains |
| --- | --- | --- |
| [Wu et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.244/) | Individual passage exits plus a reinforcement-learned global allocation policy; reports 4.3× lower compute at 95% full-model performance on SQuAD-Open. | Schedules passages to answer one query, not typed branches over one shared state; direct precedent for compute allocation. |
| [IPPD, 2026](https://arxiv.org/abs/2609.05707) | Packs multiple questions about common contexts into one autoregressive prompt and shares attention/memory work; reports GPU throughput gains without changing model architecture. | Generative multi-token execution differs from one-shot typed scoring and local CPU cost. Rules out broad novelty for common-context question sharing. |
| [Deployment-specific exits, 2026 preprint](https://arxiv.org/abs/2609.14144) | Adds intermediate readouts and tunes thresholds for deployment traffic; reports task-fidelity failure when judged only by token agreement. | Generative tokens rather than structured decisions across one request bundle; reinforces gold-label risk as the objective. |
| [TR-BERT](https://aclanthology.org/2021.naacl-main.463/) and [CoLT5](https://aclanthology.org/2023.emnlp-main.309/) | Adaptively allocate transformer depth or heavier computation to tokens. | Token routing is established; do not add it without a measured token-compute bottleneck. |
| Nimble | A schema can contain multiple typed fields; MLX processes shared context once and scores fields in parallel. The project also publishes minimal fact edits that change a typed decision. | Direct overlap on shared typed scoring and basic counterfactual curation. No task-risk selective verifier is described in the reviewed README; code-level review remains required. The 9B checkpoint is outside the memory target. |
| AdaMTL (CVPRW 2023) | Task-specific block/token controllers route tasks over a shared vision encoder; task masks are unioned so shared computation serves all requested outputs. | Closes the broad “different output, different compute on shared state” gap. It is dense vision with an active-compute target, but our remaining idea must distinguish by typed-text risk, intervention masks and evidence-verification action. |
| Cascade Transformer and EEE-QA | Progressive QA rankers, shared partial encodings, candidate pruning and joint question-answer interaction. | Strong overlap for a cheap evidence/option score followed by expensive verification. |
| PairCFR, minimally edited QA and logic-guided consistency | Counterfactual pairs plus contrastive/global alignment, query-side contrast consistency, and relation-guided QA consistency. | Strong overlap for training on fact edits. Possible residual is a verified sparse affected-field mask coupled to stage-specific task-risk routing, not contrastive learning itself. |
| Primus Decision, RSI-Jev, Typical and RULERS | Tiny CPU typed models, contextual scoring, intermediate taps and compiled rubric/evidence checks. | Direct baselines or novelty limits; see the [source ledger](sources.md). |

This does not establish that the refined weave exists or is new. It establishes that both earlier broad claims were too close to known mechanisms. The candidate must show that intervention-mask supervision improves which typed fields receive costly evidence verification, and that this improves measured local CPU risk/latency.

## Refined candidate and derivation

The candidate reuses one state encoding, gives each typed field a cheap score, then selectively pays for deeper evidence verification where predicted gold-task error warrants it. Minimal fact edits supply a verified set of fields expected to change. The shared-cost depth solver applies only if the architecture truly has shared-prefix and per-field branch costs. For additive expected gold-task loss:

$J_x(\mathbf d)=\sum_qr_q(x,d_q)+\lambda[A_x(\max_qd_q)+\sum_qB_q(x,d_q)].$

For a given request $x$ and fixed maximum depth $m$, each question minimizes $g_q(x,d)=r_q(x,d)+\lambda B_q(x,d)$ over $d\le m$, but one question must reach $m$. Therefore:

$J_{x,m}=\lambda A_x(m)+\sum_q\min_{d\le m}g_q(x,d)+\min_q[g_q(x,m)-\min_{d\le m}g_q(x,d)].$

A prefix-minimum scan is exact in $O(QL)$. We checked it against exhaustive selection on 3,000 random small problems; worst discrepancy was floating-point round-off. The [checker](../analysis/check_shared_cost_solver.py) verifies the formula, not model outputs or measured latency.

The exact reduction assumes losses add across questions, cumulative layer costs are profiled, and execution overhead introduces no unmodeled joint term. It does not solve a hard constraint on the probability that **any** answer in a bundle is wrong. Evaluate that risk on independent held-out bundles or justify a separate risk-allocation method.

## What could still be new

A more distinctive weave to test could combine:

- one cached text representation of a long state;
- CLIP-style question/option-to-evidence scores for cheap typed decisions, conditional on reliable support spans;
- verified minimal fact edits with a sparse affected-field mask;
- a per-field risk head trained against gold-task error at cheap and deep verifier stages;
- measured CPU allocation of expensive cross-attention only to fields expected to benefit.

Nimble, AdaMTL, cascade QA, PairCFR and QA consistency work each cover close parts. The possible distinction is the verified intervention mask shaping which field gets deeper decision evidence, with gold-task risk and CPU cost controlling escalation. This remains a research hypothesis: code review may find an equivalent method, retrieval may lack recall, or verifier cost may be too small to route.

## First-order and second-order screen

The first-order benefit is avoided question-branch computation for easy decisions, and possibly avoided shared layers if every branch stops early. The second-order effect is that the deepest question controls shared-state depth. If individual exit depths were independent with CDF $F$, the chance all $Q$ questions stop by layer $m$ is $F(m)^Q$; at $F=0.8$, it is 33% for five questions and 1.2% for twenty. This is a hypothetical calculation. Real question difficulty is correlated, so measure request-level maxima.

Other costs include head evaluation, risk prediction, dynamic compaction, cache lifetime, memory for intermediate states, batch padding and p95 variance. Dense masking may retain the full branch compute. CPU latency, not FLOPs, decides whether the method helps.

## What our own experiments do and do not show

- CLINC prototype screen: a frozen MiniLM with labeled intent prototypes reached 91.12% known closed accuracy and 86% OOS recall on a public development fold. It is neither a typed multi-question benchmark nor a matched local CPU comparison.
- ContractNLI evidence prototype: training evidence retrieval improved positive any-gold@5 to 95.11%, but all marked spans were retrieved at five for only 63.52%. Improved retrieval did not produce a clear stance gain.
- Matched ContractNLI heads: pooled evidence reached 70.20% accuracy versus a 68.08% hypothesis-only prior, with the paired interval including no gain. A frozen NLI scorer did not beat the matched lexical result.
- Evidence removal and shuffle: pre-registered evidence-use gates failed or remained inconclusive. The gold-span diagnostic was not deployable and did not establish a sufficient stance ceiling.
- Synthetic T4 forward pilot: shared execution won for four of five tested shapes and direct scoring won for one. This was not real model quality or CPU latency.
- Shared-cost optimizer check: 3,000 randomized small problems matched exhaustive search to floating-point precision. This verifies the formula only.

These results do not justify another evidence-retrieval training run or an adaptive model claim. See the [Codex handoff](../codex/RESUME.md) for hashes, run links and limitations.

## Training signal: sparse intervention locality

A paired state edit can carry an exact affected-field set. Train the model to answer both worlds correctly while penalizing changes to output distributions for unaffected fields. Formally, for edit $e$ from $x$ to $x^e$ and affected fields $A(e)$:

$$L_{pair}=\sum_q CE(p_q(x),y_q)+\sum_q CE(p_q(x^e),y_q^e)+\beta\sum_{q\notin A(e)}JS(p_q(x),p_q(x^e)).$$

Nimble already changes one relevant fact to change a typed answer; PairCFR combines counterfactual edits with contrastive learning; Zhang et al. train query-side contrast consistency; Asai and Hajishirzi use logic-guided QA consistency. The possible remaining distinction is a verified sparse affected-field mask that supervises every output stage and the risk controller across a typed bundle. Test target flips, unaffected-field probability spillover, evidence recall and CPU branch cost before any training. Train only if this shows a repeatable failure and improvement opportunity.

## Candidate board

| Idea | Prior-art distance | Evidence | Decision |
| --- | --- | --- | --- |
| Fixed shared state plus typed options | Low | Multiple direct competitors | Baseline. |
| Tiny non-transformer structured decision model | Low | Primus Decision directly targets small CPU typed decisions with fixed schemas | Run as a baseline. |
| Counterfactual evidence/criterion training | Low | Nimble directly describes this curation | Reproduce or ablate; do not claim as new. |
| Fact–criterion evidence memory | Low to unclear | Adjacent rubric/evidence work; project evidence-use gates failed | Secondary ablation only. |
| Independent exits over a shared trunk | Low / close | Global allocation, common-context inference and adaptive exits | No longer the lead claim. |
| Shared-cost joint depth choice by itself | Close | AdaMTL has task-aware routes over a shared encoder; no local text test | Keep as a baseline and cost-accounting tool, not a novelty claim. |
| Intervention-supervised, task-risk-gated selective verification | Unclear / close | Mathematical objective and candidate loss defined; no model/CPU measurement | Screen with exact interventions and frozen CPU profiling before training. |
| Query-aware token routing | Low | TR-BERT, CoLT5 and document token-pruning work | Explore only if profiling shows token cost dominates. |
| Sparse intervention locality across typed fields | Unclear / close | Nimble counterfactual pairs, QA contrast consistency and logic-guided consistency regularization | Secondary exact-data diagnostic; no training unless baseline spillover is real. |

## Next discriminating work

1. Finish code-level review of IPPD, global QA allocation, recent exit methods and direct typed models.
2. Freeze a grouped typed/composition workload and reference CPU; keep locked labels sealed.
3. Profile per-depth task loss, shared-layer/branch costs and request-level any-error for $Q=1,5,20$.
4. Compare fixed taps, independent exits and the exact joint allocator with real CPU compaction overhead.
5. Run a short free Kaggle adaptation only if matched results reveal a trainable gap; otherwise stop or pivot.
