# Next steps: a novelty-led, low-cost decision-model programme

**[Home](../README.md) · [Plain-English view](start-here.md) · [Novelty audit](novelty-audit.md) · [Research landscape](research.md) · [Ideas](ideas.md) · [Architecture](architecture.md) · [Codex handoff](../codex/RESUME.md)**

Research snapshot: 27 September 2026. This plan follows the original prompt: keep searching for a meaningful new combination, derive its economics, challenge it against close work, and use compute only after a cheap discriminator gives a reason. The target is under 8 GiB peak process RSS on local CPU, plus a separately measured quantized profile under 4 GiB.

## Updated research verdict

The original criterion/evidence idea is close to Nimble and RULERS. The generic per-field early-exit idea is also close to AdaMTL: it has task-aware policies over a shared encoder and merges task routes so shared work runs when any task needs it. QA cascades already share partial encodings across progressively deeper rankers.

The more promising weave to screen is **intervention-supervised, task-risk-gated verification of typed fields over a shared long state**: encode once; match fields/options to evidence cheaply; use verified fact-edit masks to teach field sensitivity and stability; spend extra CPU only on fields whose estimated gold-label error is likely to fall enough. This is a synthesis of known techniques, not yet an architecture novelty claim. The exact shared-cost depth solver is useful only if measured runtime really has a shared-prefix/max-depth cost; it has passed 3,000 randomized algebra checks, not a model or latency test.

No neural training is justified yet. First measure where CPU time goes, whether evidence recall is adequate, whether verifier branches are expensive enough to route, and whether the intervention mask improves target flips without spillover.

## Ordered execution plan and gates

### 0. Freeze the task and claim

Specify a request manifest: state, question bundle, response type/options, missing-evidence behavior, context limit and truncation policy. Freeze CPU, runtime, thread count, request mix, quality margins, latency boundary and memory accounting before opening locked tests. Full-depth agreement is a diagnostic; task labels and calibration metrics determine quality.

**Exit gate:** every adapter receives semantically identical requests and produces the same typed output contract.

### 1. Close the closest prior-art gap

Read papers and code, not only abstracts, for:

- Nimble typed scoring/data curation; AdaMTL task-aware shared-encoder routing; Cascade Transformer and EEE-QA selective rankers; RULERS evidence verification; CLIP-style retrieval and late interaction.
- Adaptive passage/resource allocation (Wu et al., EMNLP 2020), common-context parallel decoding (IPPD, 2026), CATs, CALM, LEAP, TR-BERT, CoLT5 and recent learned token-routing work.
- Counterfactual/contrastive QA consistency (PairCFR, minimally edited QA, logic-guided consistency) and deployment-specific early exits; compare objectives, action granularity, shared-work accounting and task-level metrics.

Pin checkpoint/commit revisions, licenses, supported schemas, question bundle size, runtime, calibration target, hardware, cost metric and data provenance. Distinguish GPU generation throughput from one-shot local CPU typed scoring.

**Exit gate:** if AdaMTL, a QA cascade or another method already covers the full task-risk/intervention combination under equivalent runtime assumptions, demote this to a matched adaptation/evaluation. Otherwise state the exact unfilled gap without calling it new.

### 2. Audit data and build grouped bundles

Use Typed Decisions and JevBench for direct typed comparisons, while labeling teacher-generated/synthetic subsets as such. Use a controlled RuleTaker/ProofWriter split for exact held-out operator compositions. Use RuleArena or AgentCoMa only where labels map cleanly to the typed contract. BoolQ/ANLI and ContractNLI are evidence/negation stress tests, not substitutes for the primary comparison.

Split by underlying state/document, source, template, rule combination, option/rubric family and counterfactual sibling. Create $Q=1,5,20$ workload bundles; retain 2/5/20-option and 128/512/2,048-token strata. Audit duplicates, rights, label lineage and benchmark exposure; keep calibration and test partitions separate.

**Exit gate:** at least one valid typed source and one independently held-out composition set support fair comparisons. Otherwise narrow the claim to the labels available.

### 3. Run a frozen-backbone evidence/risk response-surface probe

Choose the smallest licensed checkpoint with reproducible CPU execution. Measure shared state encoding, cheap field/evidence matching, each question-conditioned verifier stage and output calibration separately. Do not update the backbone. For every held-out question, record typed outputs and gold loss at each stage. First compute the oracle upper bound (best stage given gold labels), then test a simple margin threshold and a cross-fitted risk predictor.

Measure:

- evidence Recall@k and whether full gold evidence sets are present; report proof-oracle and retrieved-evidence conditions separately;
- target-field flip accuracy and unaffected-field probability spillover on exact fact interventions;
- per-stage accuracy, macro F1, log loss/Brier, abstention and source/type slices;
- bundle exact match and probability of at least one wrong answer for Q=1/5/20;
- CPU time for shared encoder, retrieval, branches, router, compaction and full process; p50/p95/p99 and RSS;
- cheap-only, fixed-k verifier, tuned confidence, calibrated task-risk routing, full cross-encoder and direct compact baselines.

**Exit gate:** at least one selective policy must beat fixed and direct controls on a held-out risk/latency frontier, evidence recall must support its scope, and end-to-end CPU timing must show real savings. If branch work is negligible, retriever recall caps quality, or the oracle has no headroom, stop without training.

### 4. Match direct CPU baselines

Run lexical/encoder controls and eligible Primus Decision, RSI-Jev, Laya, Kev, Typical, Decider and Rev configurations on the same reference CPU, requests and output contract. Measure process RSS, cold/warm p50/p95/p99, throughput, probability quality and every source/type slice. Hosted Jev and GPU-only IPPD remain separate references.

**Exit gate:** keep the candidate only if it has a plausible quality/latency advantage over the strongest eligible compact model under 8 GiB RSS. If an existing tiny model already meets the goal, stop rather than duplicate it.

### 5. Train one bounded adaptation only if needed

If frozen readouts show a trainable gap, add exits or a risk head to one compact model. Start with exact labels and final-depth distillation; test one change per arm. Counterfactual pairs, CLIP-style contrastive loss, query-aware token routing and ModernBERT adaptation are controlled ablations, not bundled defaults.

Use the free Kaggle T4 bridge only after a one-batch forward/backward/save/reload/export smoke. Cap the pilot from measured step time and free quota. Separate GPU training from local CPU inference. No paid compute.

**Exit gate:** stop if held-out task quality, bundle risk and measured CPU p95 fail registered margins.

### 6. Calibrate, quantize and red-team

Fit probabilities and any risk controller only on their own calibration partition. Test evidence deletion/swaps, counterfactual criterion flips, negation, option order/distractors, one hard question in an easy bundle, source shift, Q-count shift, truncation, missing evidence and unsupported schemas. Report gold-task correctness and model-agreement failures separately.

Profile full precision under 8 GiB, then a separate quantized artifact under 4 GiB. Include tokenizer, runtime, cache, buffers and concurrency in RSS.

**Final gate:** report only the named task, source, CPU, model, runtime and precision where paired quality/probability and p95 gates pass. “Fastest” requires a declared comparator set and workload.

### 7. Keep both reader levels connected

Every experiment gets a friendly summary/diagram linked to exact config, data hashes, source pins, raw aggregates and failure analysis. Preserve the original prompt and negative results. Update the [Codex handoff](../codex/RESUME.md) when the hypothesis or a gate changes.

## Immediate next actions

1. Update the prior-art audit for the exact overlap in AdaMTL (task-specific policies plus OR-combined shared execution), Nimble (multi-field context reuse and fact-edit data), and cascade rankers.
2. The in-repo synthetic generator and 256-state integrity smoke are complete. A seen-composition-fit TF-IDF floor also completed: on held-out Q=20 fields, any proof item was retrieved at top five for 100% of supported questions, but every item in the recorded proof was retrieved for only 24%. This is templated synthetic retrieval, not decision accuracy. External RuleTaker/ProofWriter terms remain unresolved. [Generator record](../experiments/data/2026-09-27-intervention-mask-probe.md) · [retrieval result](../experiments/data/2026-09-27-intervention-retrieval-floor.md).
3. Next, run a time-capped, private Kaggle inference-only comparison with pinned frozen MiniLM retrieval and NLI scoring on the same generated fixture. Compare shared sentence retrieval, top-k verification and direct full-state scoring; measure gold-path coverage, answer quality, truncation, GPU work, and a small named Kaggle CPU latency/RSS sample. This is a reference machine, not the user's target laptop; no backbone training.
4. Compare cheap-only, fixed verifier, margin-gated and task-risk-gated stages on Q=1/5/20, with source/rule-composition held out.
5. Use the results to decide whether CLIP-style evidence loss, intervention-local training, or any free Kaggle T4 pilot is justified.

No paid compute is authorized. The local probes and solver should decide whether GPU adaptation is justified.
