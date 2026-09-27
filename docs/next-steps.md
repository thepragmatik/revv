# Next steps: a novelty-led, low-cost decision-model programme

**[Home](../README.md) · [Plain-English view](start-here.md) · [Novelty audit](novelty-audit.md) · [Research landscape](research.md) · [Ideas](ideas.md) · [Architecture](architecture.md) · [Codex handoff](../codex/RESUME.md)**

Research snapshot: 27 September 2026. This plan follows the original prompt: keep searching for a meaningful new combination, derive its economics, challenge it against close work, and use compute only after a cheap discriminator gives a reason. The target is under 8 GiB peak process RSS on local CPU, plus a separately measured quantized profile under 4 GiB.

## Updated research verdict

The original criterion/evidence idea is close to Nimble and RULERS. The generic per-field early-exit idea is also close to AdaMTL: it has task-aware policies over a shared encoder and merges task routes so shared work runs when any task needs it. QA cascades already share partial encodings across progressively deeper rankers.

New primary-source review adds direct edit-aware cache repair (S67), paired answer-change supervision, and update-regression routing (S69). The seventh overlap pass adds BEST-Route, optimal routing/cascade analyses, and feature-cost cascades (S77–S79): generic quality-per-cost selection is established, not novel. The remaining discriminator is narrower: do verified arbitrary-edit pairs let a selector predict the **net gold-loss reduction** from refreshing a cached typed field, beyond confidence and unconditional local repair, at measured end-to-end cost? The current synthetic frozen-backbone screen predicts label change only, not repair value. S67 also warns that unconditional local repair may win on answer-relevant edits. See the [updated source ledger](sources.md#seventh-overlap-pass-quality-aware-routing-and-feature-cost), [novelty audit](novelty-audit.md), and [idea ledger](ideas.md).

No neural training is justified. The fresh-seed label-only count screen passed the registered Q20 minimum (74 changed fields in 22 held-out rule groups), but this is not an effect-size power analysis for the +10-point/CI gate. The first two collection windows timed out while queued; one version failed on the missing helper, and another reached CUDA setup but reset memory stats before allocator initialization. Both issues are fixed in version 3, which remained QUEUED for all 56 checks in its 14-minute collection window. A read-only recovery poll checked that exact version for 25 minutes: all 100 checks were QUEUED, and it timed out before inference without producing an artifact. No new kernel was submitted. The one-file bundle and 1,024-record hash match locally; 3 report-validator tests pass. No predictor/verifier scores exist; queue/setup failures are not model evidence. Even a positive screen cannot establish novelty or natural-language transfer; next would be a declared relevant-edit workload, rights review, and comparison with local update strategies.

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

1. **Close the Kaggle screen as inconclusive.** The no-resubmit [recovery Action 36319968934](https://github.com/thepragmatik/revv/actions/runs/36319968934) checked the exact version-3 kernel for 25 minutes; all 100 checks returned QUEUED. It timed out before inference and produced no artifact. Do not submit a fourth screen kernel or use paid compute.
2. **Keep the new CPU screen in its proper scope.** The [model-free value-of-refresh screen](../experiments/data/2026-09-28-refresh-value-screen.md) reused the in-repo paired-edit generator, evaluated 256 states over 32 compositions, and passed 10 unit tests; a repeated run produced the same row hash. On held-out Q20 at half budget, the unproved-output signal appeared to avoid 443/620 stale errors versus 310 expected at random. The red-team split showed 610 of those 620 were already wrong before the edit; only 10 were new edit-induced errors. The one-round symbolic cache is too weak to support a model or architecture claim.
3. **Resolve external-data fit and rights independently.** The [benchmark rights/fit preflight](../experiments/data/2026-09-27-edit-benchmark-selection.md) found no ready-made rights-verified primary set that combines paired edits with multiple decisions per state. ContractNLI remains a real-text no-edit baseline until paired labels are independently verified. CF-TriviaQA, RuleTaker/ProofWriter, and Contrast Sets remain subject to their recorded provenance and terms limits. Do not reuse/download their records in the primary controlled set.
4. **Design the next discriminator around a credible model pair.** Use a distinct imperfect cached predictor and refresher; separate errors that existed before an edit from new edit-induced regressions; compare confidence, edit/query similarity, label-change ranking, local refresh and full recomputation at equal refresh calls and then measured CPU cost. Register quality margins, latency boundary, RSS, uncertainty and negative-result gates before looking at held-out outcomes.
5. **Require a provider preflight before future GPU work.** A minimal free-provider job must start promptly before any training or larger screen is scheduled. Keep paid compute out of scope.
6. Train only after the preceding evidence shows a real model gap, useful verifier headroom and a rights-clear workload. Then compare direct compact baselines on the same CPU task before evaluating 8 GiB full-precision and separate 4 GiB quantized deployment profiles.

No paid compute is authorized. Synthetic mechanism results do not count as natural-language model quality, CPU latency, memory, or novelty results.
