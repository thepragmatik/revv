# Evaluation contract: how a claim earns its name

**[Home](../README.md) · [Plain explanation](start-here.md) · [Architecture](architecture.md) · [Roadmap](roadmap.md)**

## Task contract and data

A request has state, independently evaluated questions and typed responses (`yes/no`, `choice`, ordinal `score`) with explicit answer alternatives. Report a distribution and a way to defer when evidence is missing. Use semantically identical state, instructions, criteria and options across model adapters. A typed answer is not proof that it is correct.

| Role | Candidate primary dataset | Split rule / caveat |
| --- | --- | --- |
| Intent, unknown | [CLINC150](sources.md#s12), [BANKING77](sources.md#s13) | Train/dev on designated domains, reserve other domains and label descriptions for transfer. CLINC's out-of-scope examples test `NONE`; its [first train/validation audit](../experiments/data/2026-09-27-clinc-audit.md) found three exact normalized overlaps across splits, two with conflicting labels. Quarantine before baselines. Audit BANKING77 label errors. |
| Boolean evidence | [BoolQ](sources.md#s14), [ANLI](sources.md#s15) | Use supplied context; adversarial negation/entailment as separate slices. Keep related passages and templates in one partition. |
| Multi-label | [GoEmotions](sources.md#s16) | Treat labels as independent binary targets, not one forced softmax. Ordinal score needs its own annotated rubric; do not recast emotions as severity without labels. |
| Multilingual, later | [MASSIVE](sources.md#s17) | Per-language reporting; keep translated siblings together across splits. |
| External typed decision | [JevBench v1.4.2](sources.md#s32), [Typed Decisions](sources.md#s33), [Primus Decision](sources.md#s46) and [RSI-Jev](sources.md#s47) | Direct baselines include Primus and RSI-Jev. Typed Decisions has five questions per shared state but synthetic teacher labels; Primus supports a fixed question schema. Keep task scope and training history visible. Never tune on locked test results. |
| Composition | [RuleTaker/ProofWriter](sources.md#s42), [RuleArena](sources.md#s39), [AgentCoMa](sources.md#s40), plus a registered held-out rule generator | Split by rule operator combination, source, customer/document/template and surface form. Use exact labels for generated logic; adapt natural tasks only where typed mapping is valid. |

Freeze **train, development, calibration and locked test** manifests before training. Audit near-duplicates, translated and templated siblings, teacher data lineage and benchmark exposure. Development chooses models; calibration fits probabilities; locked test is read after candidate registration. Obtain a fresh blinded test when public benchmarks become optimization targets. Record dataset licenses and use rights before distributing any derived data.

## Metrics and adversarial slices

| Dimension | Report |
| --- | --- |
| Quality | Accuracy, macro F1 by source and task, balanced accuracy on skewed sets, `NONE` precision/recall, score MAE and ordinal log loss, chance/majority floors. |
| Probabilities | Multiclass/binary Brier, NLL, ECE with specified bins, reliability plot, error versus automation coverage at pre-registered risk caps (e.g. 1% and 5%). |
| Robustness | Counterfactual rubric flips, unseen operator compositions, evidence removal/shuffle/negation, option permutations and added distractors, paraphrase, missing/buried evidence, overflow, non-English, option count and question isolation. For adaptive depth, also report intermediate-to-full distribution drift and top-answer agreement per whole state/question bundle, stratified by Q and type. |
| Speed | Tokenization + model + postprocessing p50/p95/p99, cold/warm, fixed concurrency and sustained throughput; separately show model-only time, cache hits and network. |
| Memory | Peak resident set size (RSS), weights and per-request buffers at maximum supported input/options/concurrency. **Primary gate: peak process RSS < 8 GiB.** Separate compact quantized gate: **< 4 GiB**. Report quantization scheme and quality deltas. |
| Reproduction | Hardware, CPU governor and thread settings, runtime/export versions, model revision, precision, seeds, hashes, sample count, warm-up, timer boundaries and raw outcomes. |

First reference workload to pin during the baseline: named laptop-class x86 CPU, one process, 128-, 512- and 2,048-token English states, 1/5/20 questions per state, 2/5/20 options, batch 1 and concurrency 8. Report per-request and per-decision latency. The 1-question slice tests overhead; the 5- and 20-question slices test shared-state amortization. Name the exact machine in every report. Apple Silicon and T4 GPU are separate device profiles; hosted APIs are separate services. Declare context limits and measure overflow rather than silently truncating hard examples.

## Comparison and claim gates

Measure lexical, small cross-encoder and pooled-encoder controls plus supported Primus Decision, RSI-Jev, Laya, Kev, Typical and Decider configurations where they fit. Include Rev only as a quality reference if its actual local inference route is reproducible. Treat Nimble as a counterfactual-data and quality reference unless a measured quantized 8 GiB path is reproducible. Use identical inputs and hardware. For the adaptive-depth candidate, compare full-depth shared execution, fixed shallow taps, per-question early exit without shared state, shared fixed-depth execution, and shared per-question exits. Refresh the pinned [JevBench v1.4.2](sources.md#s32) board; its composite and individual axes differ, and author GPU times are not local CPU times. Check full process memory, not only parameter or weight-file size. Compare hosted Jev on the same valid labels where access permits, with network latency separate. Preregister the primary workload mix, held-out rule-composition mix and quality margins before training.

1. **Local fit:** under 8 GiB RSS at declared maximum workload; for a separately labelled compact result, under 4 GiB RSS. Neither may require a remote call or hidden helper process. A quantized result must report its difference from the same architecture before quantization.
2. **Quality floor:** locked new-source accuracy and Brier non-inferior to the strongest eligible local comparator under pre-registered margins and paired 95% intervals. Fix margins after the baseline pilot's variability, **before** training the candidate.
3. **Speed win:** lower end-to-end p95 with a 95% repeated-run interval excluding no change while passing the quality floor. Report p50, p99 and every workload slice, even if only some win.
4. **Quality win:** a positive paired accuracy difference with uncertainty and lower Brier, or state the measured trade-off. Say “beats [named model]” only for the device, mix and metrics actually won.

Bootstrap by source/document cluster rather than duplicated rows. Repeat timings over sessions and randomize run order. Publish sample counts, raw predictions, effect sizes, intervals and unsuccessful variants; a benchmark composite alone is insufficient. Use the [experiment template](../templates/experiment.md) for registration and verdict.
