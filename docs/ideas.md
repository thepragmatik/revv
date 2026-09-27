# Hypothesis ledger: conditional evidence reuse

**[Home](../README.md) · [Plain explanation](start-here.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md) · [Codex research charter](../codex/MISSION.md)**

This page records ideas to falsify, not product features or claims of previously unknown invention. Before calling an idea novel, search its closest prior art and show which *measured* difference remains. The first synthesis joins state reuse, token-level evidence matching, calibrated deferral and workload-aware routing.

## In plain English

When there are only two answers to a short question, reading the whole text with each answer can be quick and sensitive to details. When 40 answers share one long text, rereading that text 40 times is wasteful. An experimental model could choose the cheaper route, inspect individual words when a summary is ambiguous, and ask a stronger scorer to verify only uncertain cases.

```mermaid
flowchart TD
    W["State and answer count"] --> R{"Short and few?"}
    R -->|Yes| X["Joint cross score"]
    R -->|No| S["Read state once"]
    S --> E["Check token evidence"]
    E --> U{"Uncertain?"}
    U -->|Yes| V["Verify a few candidates"]
    U -->|No| A["Return calibrated options"]
    X --> A
    V --> A
```

## Precise hypothesis and prior art

**CER-1, conditional evidence reuse:** share one encoder body for a state and its candidate descriptions, score pooled vectors, and optionally use a small question-conditioned set of state token vectors for late interaction. Route short, low-option requests to a joint cross encoder. Route uncertain high-option requests to a *bounded* cross verification of the top `k` candidates; include a separate `NONE`/defer mechanism. The router can be a fixed workload rule at first, avoiding an extra trained selector. This is a proposed combination and task adaptation, **not a claim that late interaction, cascades, dynamic routing or abstention are new individually**.

Closest sources: [ColBERT](https://arxiv.org/abs/2004.12832) introduces token-level late interaction; [token-pruning analysis](https://arxiv.org/abs/2403.13291) tests inexpensive pruning; [dual intent encoders](https://arxiv.org/abs/2003.04807) already apply shared representation to intent detection; [ModernBERT](https://arxiv.org/abs/2412.13663) brings efficient encoder design; [calibrated deferral](https://arxiv.org/abs/2202.03673) shows that a defer probability needs careful treatment. [Cascade Transformer](https://aclanthology.org/2020.acl-main.504/) already prunes candidates while sharing partial transformer encodings. [BoundaryMORPH](https://arxiv.org/abs/2609.27213) allocates bounded cross-encoder calls near a retrieval set's decision boundary; its objective differs from our calibrated typed decision, but budgeted verification itself is prior art. Check additional selective-reranking and decision-model papers before asserting architectural novelty. The [synthetic T4 pilot](../experiments/kaggle/2026-09-27-forward-pilot.md) supports a workload crossover question, not any quality claim.

## Cheap cost and correctness screens

Let `L` be state tokens, `m` candidate tokens, `N=QK` candidate decisions, `r` retained evidence tokens, `p` the fraction of requests sent to verification, and `k` verified candidates. Let `E(x)` be an encoder pass and `I(r,m)` a cheap token interaction. An optimistic upper-level work model is

`C_shared = E(L) + N E(m) + N I(r,m) + p k E(r+m) + C_gate`, versus `C_cross = N E(L+m)`.

Use a shared path only when **measured** time including tokenization and gate overhead is lower, and when its paired quality/calibration meets the predeclared floor. A very large `p`, ineffective `r`, or small `N` can erase its advantage. Joint cross verification can also change answer probabilities; recalibrate the *whole route mixture*, not only each head. Candidate order must not enter the router, and all options must be scored symmetrically before any top-`k` decision.

**Candidate-recall bottleneck:** if the first stage includes the right in-scope answer among the `k` verified candidates with probability `R@k`, then even a perfect verifier cannot achieve closed-set accuracy above `R@k` on those cases. Record `R@k` before training a verifier, including recall under negation and reordered options. An `oos` / not-mentioned answer needs its own calibrated path; a top-`k` guarantee says nothing about detecting missing evidence. This is a mathematical ceiling, not a measured CER-1 result.

| Stage | Matched control | Cheap falsifier / decision |
| --- | --- | --- |
| Workload routing | Always cross; always pooled shared | End-to-end CPU crossover at 1/2/5/20 options and short/long states; routing overhead included. |
| Token evidence | Pooled shared; all-token late interaction | Negation/missing-evidence errors and throughput; whether retained evidence drops the decisive clause. |
| Selective verification | Always verify; never verify | Error versus verification coverage at a predeclared risk cap, with p95 and worst-case memory. |
| Calibration and abstention | Same architecture without gate; explicit `NONE` | Domain-shift Brier, OOS recall and selective risk; confident wrong answers fail. |

The first [CLINC audit](../experiments/data/2026-09-27-clinc-audit.md) supports testing many candidate labels and an OOS answer, but a single utterance has no multi-question long document. Require a second evidence-grounded, multi-question dataset before claiming that the complete design works. Pause after each cheap falsifier; train only the part whose potential gain survives its control.

The [ContractNLI author audit](../experiments/data/2026-09-27-contractnli-audit.md) finds 17 decisions per contract and a development median of 1,727 words. The first [full-document CPU controls](../experiments/data/2026-09-27-contractnli-controls.md) expose an adversarial weakness for CER-1: lexical top-five evidence search retrieves at least one marked span for only **49.47% of contradictions**. A five-span verifier cannot be justified on this retriever, even though CLINC's top-five **label** recall was 99.00%. Evidence recall and label recall are different gates; test both. Retrieval of all marked spans is even harder.

A controlled [training-evidence prototype screen](../experiments/data/2026-09-27-contractnli-evidence-prototypes.md) raised contradiction any-gold@5 to **94.74%** and overall positive any-gold@5 to **95.11%**, while retrieving *all* marked evidence at five for only **63.52%**. Two separate entailment/refutation centroids gave zero additional contradiction any-gold@5 improvement, failing their preregistered gate. Use one pooled train-evidence centroid as a cheap shared-retrieval control, with a separate decision head and full-clause coverage tests. This retrieval success may reflect fixed hypotheses and contract templates; it is not a proven general evidence architecture.

## CER-2 research candidate: coverage-aware shared evidence

**Plain explanation.** Read a long document once, find a handful of potentially relevant clauses for each question using examples from training, then decide whether those clauses support, contradict or fail to mention the question. When there is not enough evidence, retrieve more or defer instead of forcing a confident answer. The central uncertainty is whether a small shortlist preserves **all necessary clauses**, particularly exceptions and negation.

**Proposed mechanism, not an invention claim.** Split each document into overlapping chunks without discarding its tail; encode them once into span vectors. Represent a hypothesis by its text vector **and** a compact pooled training-evidence prototype. Rank all spans cheaply, then use a small learned joint hypothesis–span scorer only for selected cases. Train a three-way document head with a separate not-mentioned route and calibrate on document-disjoint data. Make the evidence selector's budget `k` and verification coverage explicit. Retain a direct cross-encoder control that receives the **same evidence** and a full-document control that tests whether retrieval lost decisive clauses. The author [Span NLI BERT](https://github.com/stanfordnlp/contract-nli-bert), [SENTLI](https://aclanthology.org/2022.findings-emnlp.28/), [ColBERT](https://arxiv.org/abs/2004.12832), [Cascade Transformer](https://aclanthology.org/2020.acl-main.504/) and [BoundaryMORPH](https://arxiv.org/abs/2609.27213) cover major pieces of this approach. The empirical contribution would have to be a reproducible quality–compute gain **on the typed, evidence-grounded workload** that these controls do not already provide.

For `H` hypotheses, `C` document chunks, `S` spans, width `d`, selected budget `k`, verifier fraction `p`, encoder pass cost `E`, span-pair verifier cost `V`, and gate cost `G`, a first-order arithmetic screen is

`T_shared ≈ C·E(chunk) + H·E(hypothesis) + H·S·d + p·H·k·V + G`, whereas `T_cross ≈ H·C·E(chunk+hypothesis)`.

This counts work units, **not wall-clock latency**. It omits chunk-boundary copies, tokenization, cache misses and accelerator batching; measured local CPU time and RSS must decide the crossover. A 17-hypothesis contract makes reuse plausible; a two-option short state may favor direct cross scoring. The current data support testing this inequality, not inserting numeric `E` or `V` values guessed from GPU pilots.

The statistical risk has two separate stages. If a gold label is absent from a candidate shortlist, a restricted verifier cannot recover it. If all required evidence clauses are not in an evidence shortlist, a scorer **that needs those clauses** cannot be correct for that case; our annotations alone do not tell us exactly which clauses are necessary or sufficient. Train/evaluate both `any-gold@k` and `all-gold@k`, contradiction recall, ternary macro F1, not-mentioned recall and calibrated selective risk. Learnable independent span gates could enforce a fixed expected budget with a coverage penalty, but first measure how many marked spans exceed `k` and whether overlapping chunks split the decisive exception. Do not build this gate until simple pooled retrieval plus a matched classifier reveals a failure it can address.

The first [matched ternary lexical heads](../experiments/data/2026-09-27-contractnli-lexical-heads.md) turned better evidence retrieval into only a **1.35-point** accuracy gain, with a document-bootstrap interval from **−0.87 to +3.76 points**, and reached **70.20%** accuracy versus the **68.08%** no-reading prior. Retrieval quality alone did not resolve the typed answer. Prioritize an established NLI cross scorer on the same spans as a quality control; let its remaining errors justify a specific CER-2 training objective. Do not infer that a more elaborate retriever will solve stance or not-mentioned detection.

**Adversarial controls:** shuffle hypothesis descriptions while keeping priors; remove the training evidence prototype; present the same contract with question order changed; insert a negated or exception clause outside the first 512 tokens; duplicate irrelevant boilerplate; and compare templated train/dev contracts with a different source. If the method wins only where fixed-hypothesis priors or memorized boilerplate already predict the answer, it has not validated evidence use. Record per-route process RSS and p50/p95 latency under the 8 GiB local CPU gate, then test the quantized 4 GiB profile separately.

**Next dataset candidate:** [ContractNLI](https://github.com/stanfordnlp/contract-nli) annotates 17 hypotheses per contract with entailment, contradiction, not-mentioned and evidence spans across 607 contracts; its author paper describes exception/negation difficulties. It directly tests state reuse and missing evidence, unlike CLINC. Review split provenance, download conditions, token-length truncation and document-level leakage before using it. QASPER has multiple evidence-labelled questions over papers, but answer types include free text, so a narrow typed subset would need explicit preregistration. Neither dataset automatically validates production legal decisions.

The first discriminating [frozen encoder pilot](../experiments/kaggle/2026-09-27-frozen-matching.md) isolated pooled versus token evidence under one pretrained checkpoint and fixed public development split. **Untrained MaxSim lost 3.27 points in paired known-intent accuracy** and took about 4.9 times as long for the synchronized GPU scoring kernel, without OOS improvement. Reject it for the default *untrained label-name* route; evidence-trained token interaction remains an open hypothesis. The subsequent [class-prototype screen](../experiments/kaggle/2026-09-27-prototype-screen.md) raised known closed accuracy from 70.94% to 91.12% and retained the correct answer in top five 99.00% of the time on CLINC development. This suggests candidate representation, rather than token comparison, was the bottleneck for this task; it does not prove a causal mechanism or generalization. These are component screens, not trained CER-1 evaluations or local CPU superiority.
