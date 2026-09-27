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
