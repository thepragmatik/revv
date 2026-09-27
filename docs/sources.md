# Source ledger

**[Home](../README.md) · [Research synthesis](research.md) · [Evaluation](evaluation.md)**

Primary papers, author repositories and official documentation consulted **27 September 2026**. Live pages change: pin commits, checkpoint revisions and dataset versions in experiment records. Evidence labels: paper = author-reported study; project benchmark = author-run; independent benchmark = third-party harness with its own assumptions. Our own [synthetic T4 compute screen](../experiments/kaggle/2026-09-27-forward-pilot.md) is separate from all model quality or local CPU measurements.

| ID | Primary source | Use |
| --- | --- | --- |
| <a id="s1"></a>S1 | [ModernBERT paper](https://arxiv.org/abs/2412.13663), [project](https://github.com/AnswerDotAI/ModernBERT) | Architecture and reported efficiency. |
| <a id="s2"></a>S2 | [Controlled ModernBERT / DeBERTaV3 study](https://arxiv.org/abs/2504.08716) | Architecture versus pretraining-data caveat. |
| <a id="s3"></a>S3 | [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html), [code](https://github.com/openai/CLIP) | Cross-modal contrastive learning, adapted here as a text hypothesis. |
| <a id="s4"></a>S4 | [SimCSE paper](https://aclanthology.org/2021.emnlp-main.552/) | Text contrastive pairs and hard negatives. |
| <a id="s5"></a>S5 | [SetFit paper](https://arxiv.org/abs/2209.11055) | Few-shot contrastive text classifier. |
| <a id="s6"></a>S6 | [TinyBERT paper](https://arxiv.org/abs/1909.10351) | Teacher–student compression. |
| <a id="s7"></a>S7 | [Laya](https://github.com/NandhaKishorM/laya), [benchmarks](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | Typed outputs, author speed and calibration/zero-shot caveats. |
| <a id="s8"></a>S8 | [Kev](https://github.com/jaredpalmer/kev), [0.8B model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-0.8b.md), [research plan](https://github.com/jaredpalmer/kev/blob/main/PLAN.md) | Pointer head, state reuse, held-out metrics and limits. |
| <a id="s9"></a>S9 | [TypeSafe introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [workflow evals](https://evals.typesafe.ai/) | Jev contract, provider claims and label methodology. |
| <a id="s10"></a>S10 | [JevBench methods and results](https://github.com/fstandhartinger/jevbench) | Independent typed-decision harness, scoring and caveats; distinct from jevbench.dev's gaming project. |
| <a id="s11"></a>S11 | [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Deployment and scoring guidance. |
| <a id="s12"></a>S12 | [CLINC out-of-scope data](https://github.com/clinc/oos-eval) | Intent and unknown intent. |
| <a id="s13"></a>S13 | [BANKING77](https://huggingface.co/datasets/PolyAI/banking77), [label-error study](https://aclanthology.org/2022.insights-1.19/) | Fine-grained intent and label audit. |
| <a id="s14"></a>S14 | [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Boolean questions with supplied passages. |
| <a id="s15"></a>S15 | [ANLI](https://github.com/facebookresearch/anli) | Adversarial inference. |
| <a id="s16"></a>S16 | [GoEmotions paper](https://aclanthology.org/2020.acl-main.372/), [data](https://github.com/google-research/google-research/tree/master/goemotions) | Human-annotated multi-label emotion. |
| <a id="s17"></a>S17 | [MASSIVE paper](https://aclanthology.org/2023.acl-long.235/), [data](https://www.amazon.science/code-and-datasets/massive) | Multilingual intent gate. |
| <a id="s18"></a>S18 | [ColBERT paper](https://arxiv.org/abs/2004.12832), [Cascade Transformer](https://aclanthology.org/2020.acl-main.504/), [BoundaryMORPH](https://arxiv.org/abs/2609.27213) | Token interaction, candidate-pruning cascades and bounded cross verification are established prior art. The last is a September 2026 preprint; independently validate its reported results before relying on them. |
| <a id="s19"></a>S19 | [ContractNLI paper](https://aclanthology.org/2021.findings-emnlp.164/), [dataset site and terms](https://github.com/stanfordnlp/contract-nli) | Multiple hypotheses and evidence per long contract; prospective independent workload gate. |
| <a id="s20"></a>S20 | [QASPER paper](https://aclanthology.org/2021.naacl-main.365/), [data](https://huggingface.co/datasets/allenai/qasper) | Alternative multiple-question evidence workload; free-form answers need a typed-task mapping. |
| <a id="s21"></a>S21 | [Schuster et al. long-document NLI](https://aclanthology.org/2022.findings-emnlp.28/), [NevIR negation retrieval benchmark](https://aclanthology.org/2024.eacl-long.139/) | Existing evidence selection / aggregation controls and polarity-sensitive retrieval stress test; lexical evidence ranking is prior art. |
| <a id="s22"></a>S22 | [NLI MiniLM cross-encoder model card](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) | Frozen 82.1M-parameter Apache-2.0 sentence-pair NLI control, logits contradiction/entailment/neutral; checkpoint access and local suitability require measurement. |

## Additional overlap and evaluation sources

| ID | Primary source | Use |
| --- | --- | --- |
| <a id="s23"></a>S23 | [Laya shortlist implementation](https://github.com/NandhaKishorM/laya) and [issue reporting top-20 Banking77 results](https://github.com/NandhaKishorM/laya/issues/102) | Direct baseline for pooled candidate shortlist; issue numbers are author-reported, not independently reproduced here. Pin live commit before comparing. |
| <a id="s24"></a>S24 | [Kev source](https://github.com/jaredpalmer/kev), [model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-0.8b.md) | Packed question branches, shared state and pointer-style option scoring overlap with generic state reuse. Verify current license/runtime/checkpoint before inclusion. |
| <a id="s25"></a>S25 | [ColBERTv2](https://arxiv.org/abs/2112.01488) | Late interaction, hard-negative training, teacher distillation and residual compression are established retrieval techniques. |
| <a id="s26"></a>S26 | [DocInfer](https://aclanthology.org/2022.acl-long.180/) | Hierarchical document evidence retrieval and pruning prior art; compare simpler controls before proposing a new selector. |
| <a id="s27"></a>S27 | [MultiRC task and data](https://cogcomp.seas.upenn.edu/multirc/) | Alternative evidence-grounded evaluation. Multiple answers may be correct; model each answer as a separate binary decision or report appropriate multi-label metrics. |
| <a id="s28"></a>S28 | [Selective prediction evaluation](https://aclanthology.org/2022.acl-long.223/) | Report risk versus coverage and calibration with explicit selective-prediction controls. |


## Current System-1, cache and compositionality review

| ID | Primary source | Use |
| --- | --- | --- |
| <a id="s29"></a>S29 | [Typical research page](https://typical.ozlabs.ai/research) and [model release](https://typical.ozlabs.ai/) | Direct recent typed-decision comparator: shared state cache, contextual candidate readout, counterfactual rubric training, option-set tests and reported OOD hard-tier failure. Author-reported; remeasure locally. |
| <a id="s30"></a>S30 | [Decider model cards](https://github.com/Mapika/decider) and [4B v2.1 card](https://github.com/Mapika/decider/blob/main/MODEL_CARD_4B.md) | Current open 2B/4B typed decision baselines; supervised/replay/calibration stages. 4B BF16 weights are 8.4 GB before runtime, so require a measured quantized profile for our memory gates. |
| <a id="s31"></a>S31 | [Rev decision model](https://github.com/robbalian/rev) | Shared-state, pointer-scoring competitor and author-run quality reference; pin revision and verify CPU/local support before comparison. |
| <a id="s32"></a>S32 | [JevBench v1.4.2 results](https://www.benchmarkheaven.com/jev-models) and [harness](https://github.com/fstandhartinger/jevbench) | Current public/sealed decision-model comparison and evolving competitor field. Composite, speed, intelligence and hard-tier rankings differ; board hardware does not replace a matched local CPU run. |
| <a id="s33"></a>S33 | [Typed Decisions dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Five typed questions per shared state and soft probability labels; 400 public test cases over synthetic workflows, so useful for interface/calibration checks but not real-world ground truth. |
| <a id="s34"></a>S34 | [ContrastKV](https://aclanthology.org/2026.acl-long.417/) | ACL 2026 robust query-agnostic KV eviction specifically for multi-query generalization; generic reusable-memory novelty is not available. |
| <a id="s35"></a>S35 | [CacheNotes](https://aclanthology.org/2026.eacl-long.309/) | Task-aware one-time cache compression for diverse downstream reasoning queries; close compression prior art. |
| <a id="s36"></a>S36 | [KV-Distill](https://arxiv.org/abs/2503.10337) and [KVzip](https://arxiv.org/abs/2505.23416) | Question-independent learned compression and query-agnostic cache reconstruction across multiple downstream tasks. |
| <a id="s37"></a>S37 | [Set-LLM](https://papers.nips.cc/paper_files/paper/2025/hash/5abdb0ae08d5a00fbdcdc187178276ac-Abstract-Conference.html) | Permutation-invariant set/text processing; option-order invariance is known work. |
| <a id="s38"></a>S38 | [When KV Cache Reuse Fails in Multi-Agent Systems](https://aclanthology.org/2026.acl-long.327/) | ACL 2026 evidence that cache reuse can disrupt cross-candidate interaction for judge-like choices; include set-context controls. |
| <a id="s39"></a>S39 | [RuleArena](https://aclanthology.org/2025.acl-long.27/) | Real-world airline, NBA and tax rule reasoning with long context, logic and arithmetic; possible composition stress set if typed mapping is valid. |
| <a id="s40"></a>S40 | [AgentCoMa](https://aclanthology.org/2026.acl-long.380/) | ACL 2026 commonsense-plus-math composition benchmark; reported large accuracy drop when both steps are combined. |
| <a id="s41"></a>S41 | [C2Gen NLI](https://aclanthology.org/2024.tacl-1.51/) | Compositional NLI under continual learning; supports the need for held-out composition rather than random row splits. |
| <a id="s42"></a>S42 | [RuleTaker/ProofWriter generator](https://github.com/allenai/ruletaker) | Exact theorem-prover labels for controlled rule-composition diagnostics; synthetic results do not establish natural-language transfer. |
| <a id="s43"></a>S43 | [SURE-RAG](https://arxiv.org/abs/2605.03534) | Evidence sufficiency as a set-level property with support/refute/insufficient outcomes; coverage-gate novelty is weak. |
| <a id="s44"></a>S44 | [Learning Evidence Sufficiency Boundaries for Selective Answering](https://arxiv.org/abs/2609.01687) | Direct recent prior art on sufficiency boundaries and abstention as evidence accumulates. |
| <a id="s45"></a>S45 | [EEE-QA](https://arxiv.org/abs/2403.02176) | Joint question/answer-option representations and throughput analysis; compare before proposing candidate-set encoding as new. |

The current candidate is question-specific early exit over a shared state pass, with a calibration bound across all questions and layers in one request bundle. Shared state, adaptive depth and conformal consistency are established components; this exact typed multi-question intersection remains a search target, not an established novelty gap. Criterion/evidence factorization is a secondary ablation. See [ideas](ideas.md), [research synthesis](research.md) and [the gated plan](next-steps.md).



## New direct competitors and adaptive-compute prior art

| ID | Primary source | Use |
| --- | --- | --- |
| <a id="s46"></a>S46 | [Primus Decision 0.1 model card](https://huggingface.co/The-Aame/primus-decision-0.1), [source and experiment records](https://github.com/pally-sai-tilak/primus-decision) | Direct tiny CPU typed-decision baseline: 3.7M S4D/GRU weights plus 142.8 MB LSA features; fixed supported schema; author-reported 75.1% on the 2,000-decision Typed Decisions test and 1.4–1.6 GB peak RSS on its named CPU. Checkpoint and labels are teacher-specialized; independently reproduce on a matched CPU. |
| <a id="s47"></a>S47 | [RSI-Jev 1.0 model card](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b), [training and evaluation repository](https://github.com/Shanghua-Gao/RSI-Jev) | Open 0.8B and 2B typed scorers with question-conditioned option cross-attention, shuffled choices and teacher-distribution training. Direct quality reference; independently test CPU memory, latency and distribution calibration. |
| <a id="s48"></a>S48 | [Bespoke Nimble source](https://github.com/bespokelabsai/nimble), [published 9B adapter](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) | Direct prior art for counterfactual decision data: edit a relevant fact while keeping policy/question/context fixed, validate evidence use and assign typed labels. Its large backbone and synthetic model-checked data constrain direct comparison. |
| <a id="s49"></a>S49 | [RULERS paper](https://arxiv.org/abs/2601.08654), [code](https://github.com/LabRAI/Rulers) | Rubric compilation, deterministic evidence verification and post-hoc calibration; close overlap for criterion/evidence factorization, although not the same small neural typed model. |
| <a id="s50"></a>S50 | [Consistent Accelerated Inference via Confident Adaptive Transformers](https://aclanthology.org/2021.emnlp-main.406/) | Early-exit consistency with a conformal stopping rule. Closest source for calibrating an early-versus-full prediction guarantee; not a typed multi-question bundle method. |
| <a id="s51"></a>S51 | [Confident Adaptive Language Modeling](https://papers.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html) | Adaptive depth by example or generation step; generic dynamic compute is prior art. |
| <a id="s52"></a>S52 | [LEAP: Layer-wise Exit-Aware Pretraining](https://aclanthology.org/2026.acl-industry.52/) | Recent exit-aware intermediate-layer training; suggests intermediate/final alignment must be tested rather than assumed. |
| <a id="s53"></a>S53 | [AdaMTL: Adaptive Input-Dependent Inference for Efficient Multi-Task Learning](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Neseem_AdaMTL_Adaptive_Input-Dependent_Inference_for_Efficient_Multi-Task_Learning_CVPRW_2023_paper.html) | Shared encoder with task-aware adaptive compute; close prior art for per-branch compute allocation. Exact text typed-query overlap remains to be checked. |

The current lead is now the narrow shared-state, question-specific exit and bundle-calibration experiment. Criterion/evidence slots and counterfactual evidence training remain secondary controls because direct prior art is close. Read [the idea ledger](ideas.md), [research synthesis](research.md) and [the gated execution plan](next-steps.md) for the precise claim boundary.
