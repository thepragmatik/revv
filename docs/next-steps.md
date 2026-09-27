# Next steps: a novelty-led, low-cost decision-model programme

**[Home](../README.md) · [Plain-English view](start-here.md) · [Novelty audit](novelty-audit.md) · [Research landscape](research.md) · [Ideas](ideas.md) · [Architecture](architecture.md) · [Codex handoff](../codex/RESUME.md)**

Research snapshot: 27 September 2026. This plan follows the original prompt: keep searching for a meaningful new combination, derive its economics, challenge it against close work, and use compute only after a cheap discriminator gives a reason. The target is under 8 GiB peak process RSS on local CPU, plus a separately measured quantized profile under 4 GiB.

## Updated research verdict

The original criterion/evidence idea is close to Nimble and RULERS. The generic per-field early-exit idea is also close to AdaMTL: it has task-aware policies over a shared encoder and merges task routes so shared work runs when any task needs it. QA cascades already share partial encodings across progressively deeper rankers.

New primary-source review adds direct edit-aware cache repair (S67), paired answer-change supervision (NormWorlds-CF and CRPO), and update-regression routing (S69). This removes any broad novelty claim for edit sensitivity or confidence comparison. The surviving discriminator is narrower: **does a typed-output-level router choose which old answers to verify after a fact edit, at equal verifier budget, while improving gold answers and reducing measured end-to-end CPU time?** The current screen is frozen-backbone and synthetic; its change-probability target is not expected repair value. S67 also warns that when edits are answer-relevant, unconditional local repair may outperform risk selection. See the [updated source ledger](sources.md#fourth-overlap-pass-edit-aware-inference-and-answer-changes) and [idea ledger](ideas.md).

No neural training is justified. The fresh-seed label-only count screen passed the registered Q20 minimum (74 changed fields in 22 held-out rule groups), but this is not an effect-size power analysis for the +10-point/CI gate. The first two collection windows timed out while queued; a later version first failed on a missing sibling helper. A standalone bundle then executed but stopped at `torch.cuda.reset_peak_memory_stats(0)` with an invalid-device error before either model loaded. Its CUDA availability check had passed. The failure exposed initialization order: peak counters were reset before the allocator had any CUDA allocation. The screen now initializes CUDA, creates and releases a warm-up tensor, synchronizes and then resets the counters. The exact one-file bundle and 1,024-record hash match locally; 3 report-validator tests pass. One more free T4 run is next. No predictor/verifier scores exist; these setup failures are not model evidence. Even a positive screen cannot establish novelty or natural-language transfer; next would be a declared relevant-edit workload, rights review, and comparison with local update strategies.

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

1. Submit one more corrected private free Kaggle T4 run with explicit CUDA initialization before resetting peak statistics. The prior standalone run reached CUDA setup but failed before model loading; its availability guard passed. The one-file bundle and frozen record hash match locally, and the report-validator tests pass. Keep all prior failures out of model-quality conclusions. [Methods and gates](../experiments/kaggle/2026-09-27-intervention-risk-probe.md) · [Pre-inference sample-size audit (not statistical power)](../experiments/kaggle/2026-09-27-intervention-risk-power-check.json) · [packaging/runtime correction](../experiments/kaggle/2026-09-27-intervention-risk-packaging-correction.json) · [initial queue record](../experiments/kaggle/2026-09-27-intervention-risk-queue-timeout.json) · [recovery timeout](../experiments/kaggle/2026-09-27-intervention-risk-recovery-timeout.json) · [run 36318581843](https://github.com/thepragmatik/revv/actions/runs/36318581843).
2. If the sample-size gate fails, label the result inconclusive and do not change the seed after looking at model scores. If change recall passes but refreshed-answer quality or CPU savings fail, archive the negative and stop this risk-gating path.
3. If both predictive and end-to-end signals are promising, audit NormWorlds-CF data/code access and build a relevant-edit evaluation slice; verify rights before fetching or redistributing anything. Compare against confidence refresh, unconditional edit-local repair, and full recomputation.
4. Only after an independent, rights-clear workload and a matched local-CPU quality/latency/RSS comparison show headroom should a training or architecture build be considered. Keep backbone training paused and paid compute out of scope.

No paid compute is authorized. The synthetic probe is a cheap screening step, not the final benchmark.
