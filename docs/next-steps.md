# Next steps: evidence before architecture

**[Home](../README.md) · [Plain explanation](start-here.md) · [Research landscape](research.md) · [Idea ledger](ideas.md) · [Codex handoff](../codex/RESUME.md)**

Research snapshot: 27 September 2026. This plan follows the [original mission](../codex/ORIGINAL_PROMPT.md): seek a genuinely useful new combination, derive and test its limits before building, keep it fast on local CPU under 8 GiB peak process memory, and maintain a separately measured quantized profile under 4 GiB. “Beats existing models” means a paired, same-task, same-hardware comparison with uncertainty intervals; it is not yet demonstrated.

## What the literature and our results say

**Enough has been reviewed to choose the next experiments, not enough to claim novelty or select a winning architecture.** The latest primary-source review materially narrows the contribution:

| Proposed ingredient | Closest existing work | Consequence |
| --- | --- | --- |
| Encode shared state once; branch questions from reused state | Kev's block-causal packed pass and pointer head; Laya also reuses a pooled state representation | State reuse alone is not a novelty claim. Compare exact released implementations and costs. |
| Shortlist options from a pooled representation before typed decision | Laya `predict_shortlist` | Must include this as a direct baseline, with top-k recall and full route cost. |
| Token-level evidence matching, compression, distillation and hard negatives | ColBERT / ColBERTv2 | These are known techniques. Only test token interaction if pooled evidence has a measured failure it can fix. |
| Aggregate evidence over long documents and use NLI scores for selection | SENTLI and ContractNLI work | Long-document aggregation and clause selection are prior art. Include same-evidence cross scoring. |
| Hierarchical evidence pruning / graph pooling | DocInfer | Evidence pruning is not itself new. Test whether the proposed gate adds value over its simple controls. |
| Budgeted cascades, confidence-based verification, abstention | Cascade Transformer; BoundaryMORPH; selective prediction literature | A bounded verifier needs a distinct measurable objective, not a newness claim. |
| Fixed-hypothesis evidence prototype plus not-mentioned decision | Direct pieces exist; our ContractNLI screens show strong retrieval but weak stance/aggregation evidence | Current data do not justify neural shared-encoder training. |

The possible research question worth testing is narrower: **can a decision system predict when its current evidence is incomplete or contradicted, and buy a bounded extra evidence/verification pass only in those cases—while improving the paired quality–latency frontier over direct cross scoring, Laya's shortlist, Kev's state reuse, and confidence-only routing?** This is a hypothesis, not a claim of a new architecture. It becomes interesting only if coverage-aware routing beats those controls under negation, exceptions, missing evidence, and domain shift at equal memory and compute budgets.

The current ContractNLI results are a concrete warning: pooled evidence retrieval reached 95.11% positive `any-gold@5` and 63.52% `all-gold@5`, yet the matched ternary classifier was only 70.20% accurate versus a 68.08% hypothesis-only prior, with an uncertain paired gain. Frozen NLI cross-scoring did not improve it. The marked-span diagnostic showed that evidence availability helps one restricted positive-only view, but did not meet the registered balanced gap and did not cover the full three-way task. Thus neither more retrieval complexity nor another GPU run is presently supported.

## Ordered execution plan and gates

### 0. Freeze the comparison contract

Write one machine-readable manifest for each task: exact source/version/license, document-level split and hashes, input and output schema, label mapping, truncation policy, allowed calibration data, and one primary quality metric plus safety/abstention metrics. Keep the ContractNLI test unopened. Preserve the existing CLINC/ContractNLI development results as exploratory; do not tune a new model repeatedly against them and call them held-out evidence.

Before external comparisons, choose the actual target CPU(s) and specify whether 8 GiB includes only the model process or the full host. If no target device is specified, report CPU/model/task measurements by named reference machine and make no universal fastest claim. Keep a distinct 4 GiB quantized acceptance profile. Pin competitor checkpoint commits, tokenizer, runtime, precision and adapters; do not compare author GPU timing to our CPU timing.

**Exit gate:** all methods can consume identical normalized examples and produce identical typed output records; quality thresholds, candidate option counts, route budgets and hardware are written before looking at new scores.

### 1. Establish matched baseline quality and CPU cost

Run the inexpensive controls first on frozen splits: current lexical and MiniLM baselines; direct cross scoring; official Laya (including `predict_shortlist`); and Kev where its license, checkpoint, runtime and hardware footprint permit. Do not treat hosted Jev latency as local inference latency. Measure both cold start and steady-state, complete tokenization-to-output wall time, p50/p95, throughput under stated batching, peak process RSS, model artifact size and quality. Use identical examples and record failures, not only successful timing rows.

Stress the workload dimensions separately: 2, 5, 20 and 77 options; short and long states; one versus many questions per state; option order permutations; and memory profile. For Kev/Laya, test supported input contracts directly rather than forcing unsupported modes into a misleading comparison. The full process, tokenizer and runtime count toward memory.

**Exit gate:** reproducible baselines and uncertainty intervals; a real local CPU bottleneck or quality gap is identified. If a baseline already meets the goal, do not train a duplicate model.

### 2. Add one independent evidence-grounded workload

ContractNLI is useful but fixed-hypothesis and domain-specific. Audit MultiRC as a shorter, varied question-plus-paragraph workload. Treat each candidate answer as its own binary decision because MultiRC permits multiple correct answers; do not silently turn it into single-label softmax. Audit QASPER as a possible long-document extension, but only use a preregistered yes/no or answerable/not-answerable slice with evidence and an explicit mapping; its free-form answer cases do not naturally fit our output contract. Check licensing, annotation, length, split leakage and overlap before selection.

For whichever dataset passes, report performance by question count per state, evidence length, answerability, negation/exception and whether all gold evidence was retrieved. Keep source-disjoint results separate from within-source generalization. A tiny manually adjudicated challenge set can supplement published data, but must be documented and not tuned into the test set.

**Exit gate:** at least one non-CLINC workload with independent questions and evidence supports the same typed contract; otherwise limit claims to the workloads actually measured.

### 3. Test the proposed mechanism without training

On the same frozen examples compare: (a) direct cross encoder; (b) pooled state/question or option shortlist; (c) shortlist plus fixed top-k evidence; (d) confidence-only extra verification; and (e) coverage/counterevidence-triggered extra retrieval or verification. Start with fixed, transparent gates and an existing scorer. Measure option `Recall@k`, evidence `any-gold@k` and `all-gold@k`, contradiction/negation slices, abstention risk, and paired latency/quality. Fit any threshold only on training/calibration documents; report development once after preregistration.

Counterfactual checks: reorder questions/options; swap support and contradiction evidence; move a negation/exception beyond the initial window; insert irrelevant boilerplate; remove all evidence; shuffle evidence across documents; and compare the same state with different hypotheses. For a valid verifier, report the hard ceiling imposed by shortlist recall and the number of essential evidence clauses excluded. The route must treat candidates symmetrically and cannot use answer order as a shortcut.

A cheap upper bound helps decide whether routing can pay: if the first stage retains the correct answer with probability `R@k`, a verifier restricted to those candidates has closed-set accuracy at most `R@k` on that slice, even with a perfect verifier. For the gated route, the maximum possible average gain over the baseline is bounded by `P(gate) × P(baseline-wrong and recoverable | gate)`; the realized gain is smaller after verifier error. Measure these terms on the frozen sample before tuning a gate. Similarly, a route that invokes verification on fraction `p` has expected cost `C_first + p·C_verify + C_gate`; compare this measured end-to-end cost and its p95 to always-verify and confidence-only controls. If the bound is below the registered quality margin, stop without training.

**Decision gate:** retain a coverage-aware route only if it beats both confidence-only routing and always-verify on the paired quality–latency frontier, has no material regression on missing-evidence/negation slices, and stays within the fixed local memory/compute budgets. Otherwise reject this mechanism and update the idea ledger.

### 4. Only then consider one small training experiment

If and only if step 3 shows a repeatable failure that training can plausibly address, create one small candidate with one hypothesis and one changed variable. Start with a batch-level forward/backward/save/load/export smoke. Then run one short pilot on training data; estimate full-run time and memory from that pilot. A sensible first model is a compact encoder or student, not ModernBERT-large or a 0.8B generative model by default. Compare trained pooling against the frozen direct scorer; add contrastive hard negatives, teacher distillation, token interaction or a learned evidence-coverage gate **one at a time**, based on error slices. Use grouped document splits, fixed seeds and a prewritten stop rule.

Do not spend Kaggle GPU quota until the CPU controls identify the target change and the run's maximum duration is estimated. Kaggle is the low-cost training venue, not proof of local CPU speed. Log GPU model/time separately from CPU inference and memory measurements.

**Exit gate:** a reproducible paired gain on development, retained on the independent-source workload, before calibration and the sealed test are touched. Stop after a failed registered gate; do not add architecture pieces to rescue it post hoc.

### 5. Red-team, calibrate, and measure deployment profiles

Calibrate on disjoint calibration data and evaluate Brier score, expected calibration error with binning details, selective risk at fixed coverage, abstention/OOS recall, and reliability by domain and option count. Red-team negation, exception clauses, multiple valid choices, no valid choice, contradictory passages, long-tail truncation, duplicate candidates, candidate-order changes, adversarial paraphrases and source shift. A prediction that is fast but confidently wrong on an unsupported state fails.

Export only a passing candidate. First profile the full-precision 8 GiB CPU process. Then independently export and validate a quantized profile under 4 GiB; report task quality deltas and full-process RSS, not just weight-file size. Require output parity within registered tolerance and repeat latency/RSS measurements after export. Keep test locked until the candidate, thresholds and comparison margin are frozen.

### 6. Publish evidence, including failures

For every experiment commit the preregistration, exact config/source hashes, scripts, machine-readable metrics and concise interpretation. Link reader-friendly interpretation to the technical method and raw records. A winning claim must state workload, competitor revision, CPU, precision, batching, memory definition, timing distribution, quality uncertainty and excluded cases. “Fastest” without these bounds is not a research result.

## Immediate next actions

1. Keep neural shared-encoder training paused.
2. Refresh the exact current Laya and Kev checkpoint/repository revisions, licenses, local inference support and benchmark commands; run a CPU smoke only if the full route is reproducible.
3. Select/record a reference CPU. If the user has not specified hardware, start with an explicitly named available runner and report it as a reference, not the target device.
4. Draft the matched baseline manifest and run inference-only baseline measurements. Do not download multi-gigabyte weights until their license, storage, memory fit and benchmark path are established.
5. Audit MultiRC/QASPER schema and licensing, choose at most one for the next cheap evaluation.
6. Preregister and run the no-training routing discriminator; only its result can unlock another training proposal.

## Research and benchmark sources

See the [full source ledger](sources.md). Key overlaps reviewed for this plan:

- [Laya repository and shortlist interface](https://github.com/NandhaKishorM/laya), including the reported top-20 Banking77 example (author issue; not independently reproduced by us).
- [Kev implementation](https://github.com/jaredpalmer/kev) and [0.8B model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-0.8b.md).
- [ColBERTv2](https://arxiv.org/abs/2112.01488), [SENTLI](https://aclanthology.org/2022.findings-emnlp.28/), [DocInfer](https://aclanthology.org/2022.acl-long.180/), and [ContractNLI](https://aclanthology.org/2021.findings-emnlp.164/).
- [MultiRC](https://cogcomp.seas.upenn.edu/multirc/) and [QASPER](https://aclanthology.org/2021.naacl-main.365/) for alternative evidence-grounded workloads.
- [NevIR](https://aclanthology.org/2024.eacl-long.139/) for negation-sensitive retrieval; [selective prediction evaluation](https://aclanthology.org/2022.acl-long.223/) for risk/coverage reporting.
