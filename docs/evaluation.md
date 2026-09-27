# Evaluation contract: how a claim earns its name

**[Home](../README.md) · [Plain explanation](start-here.md) · [Architecture](architecture.md) · [Roadmap](roadmap.md)**

## Task contract and data

A request has state, independently evaluated questions and typed responses (`yes/no`, `choice`, ordinal `score`) with explicit answer alternatives. Report a distribution and a way to defer when evidence is missing. Use semantically identical state, instructions, criteria and options across model adapters. A typed answer is not proof that it is correct.

| Role | Candidate primary dataset | Split rule / caveat |
| --- | --- | --- |
| Intent, unknown | [CLINC150](sources.md#s12), [BANKING77](sources.md#s13) | Train/dev on designated domains, reserve other domains and label descriptions for transfer. CLINC's out-of-scope examples test `NONE`; audit BANKING77 label errors. |
| Boolean evidence | [BoolQ](sources.md#s14), [ANLI](sources.md#s15) | Use supplied context; adversarial negation/entailment as separate slices. Keep related passages and templates in one partition. |
| Multi-label | [GoEmotions](sources.md#s16) | Treat labels as independent binary targets, not one forced softmax. Ordinal score needs its own annotated rubric; do not recast emotions as severity without labels. |
| Multilingual, later | [MASSIVE](sources.md#s17) | Per-language reporting; keep translated siblings together across splits. |
| External | [JevBench](sources.md#s10), pinned release | Supplemental comparison only; do not train on public tests or tune on sealed results. |
| Workflow | Human-reviewed support/routing/policy cases and verified synthetic rules | Partition by source, customer/document/template and rule family; keep private states out of this public repo. |

Freeze **train, development, calibration and locked test** manifests before training. Audit near-duplicates, translated and templated siblings, teacher data lineage and benchmark exposure. Development chooses models; calibration fits probabilities; locked test is read after candidate registration. Obtain a fresh blinded test when public benchmarks become optimization targets. Record dataset licenses and use rights before distributing any derived data.

## Metrics and adversarial slices

| Dimension | Report |
| --- | --- |
| Quality | Accuracy, macro F1 by source and task, balanced accuracy on skewed sets, `NONE` precision/recall, score MAE and ordinal log loss, chance/majority floors. |
| Probabilities | Multiclass/binary Brier, NLL, ECE with specified bins, reliability plot, error versus automation coverage at pre-registered risk caps (e.g. 1% and 5%). |
| Robustness | Paired option permutations and renaming; paraphrase, negation, contradiction, missing or buried evidence, input overflow, non-English, option count and question isolation. |
| Speed | Tokenization + model + postprocessing p50/p95/p99, cold/warm, fixed concurrency and sustained throughput; separately show model-only time, cache hits and network. |
| Memory | Peak resident set size (RSS), weights and per-request buffers at maximum supported input/options/concurrency. **Primary gate: peak process RSS < 8 GiB.** Separate compact quantized gate: **< 4 GiB**. Report quantization scheme and quality deltas. |
| Reproduction | Hardware, CPU governor and thread settings, runtime/export versions, model revision, precision, seeds, hashes, sample count, warm-up, timer boundaries and raw outcomes. |

First reference workload to pin during the baseline: laptop-class x86 CPU, one process, 128- and 2,048-token English states, 1 and 8 questions, 2/5/20 options, batch 1 and concurrency 8. Name the exact machine in every report. Apple Silicon and T4 GPU are separate device profiles; hosted APIs are separate services. Declare context limits and measure overflow rather than silently truncating hard examples.

## Comparison and claim gates

Measure lexical, small cross-encoder and pooled dual-encoder controls plus supported Laya and Kev-0.8B/4B configurations **if they fit** on identical hardware and data, disclosing each precision and cache condition. Before a claimed win, refresh the pinned [JevBench](sources.md#s10) board and include leading open systems (currently Plumb-4B and decider-4b v2) if their *whole local process* fits 8 GiB; a parameter or weight count cannot establish eligibility. Rank compact 4 GiB builds only against compact eligible builds. Show ineligible models' quality separately. Compare hosted Jev on the same valid labelled tasks where access permits, with network latency in its own category. Pre-register the primary mix and an unseen-source mix before choosing a favorable result.

1. **Local fit:** under 8 GiB RSS at declared maximum workload; for a separately labelled compact result, under 4 GiB RSS. Neither may require a remote call or hidden helper process. A quantized result must report its difference from the same architecture before quantization.
2. **Quality floor:** locked new-source accuracy and Brier non-inferior to the strongest eligible local comparator under pre-registered margins and paired 95% intervals. Fix margins after the baseline pilot's variability, **before** training the candidate.
3. **Speed win:** lower end-to-end p95 with a 95% repeated-run interval excluding no change while passing the quality floor. Report p50, p99 and every workload slice, even if only some win.
4. **Quality win:** a positive paired accuracy difference with uncertainty and lower Brier, or state the measured trade-off. Say “beats [named model]” only for the device, mix and metrics actually won.

Bootstrap by source/document cluster rather than duplicated rows. Repeat timings over sessions and randomize run order. Publish sample counts, raw predictions, effect sizes, intervals and unsuccessful variants; a benchmark composite alone is insufficient. Use the [experiment template](../templates/experiment.md) for registration and verdict.
