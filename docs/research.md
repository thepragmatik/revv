# Research landscape and adversarial reading

**[Home](../README.md) · [Plain explanation](start-here.md) · [Design](architecture.md) · [Sources](sources.md)**

This is a synthesis of primary sources checked 27 September 2026, not an independently reproduced leaderboard. See the [source ledger](sources.md).

| Approach | Idea to borrow | Evidence and limitation | Experiment |
| --- | --- | --- | --- |
| [ModernBERT](sources.md#s1) | Bidirectional encoder, alternating attention, unpadding | Strong reported efficiency, but a [controlled study](sources.md#s2) warns that pretraining data confounds architectural comparisons. Long-context tricks may help little at 128 tokens. | Match data and runtime against smaller encoders; stratify by input length. |
| [CLIP](sources.md#s3), [SimCSE](sources.md#s4) | Contrastive alignment | CLIP is image–text; we propose borrowing its *objective* for text–criterion pairs. Semantically similar valid answers can be false negatives. | Supervised loss versus contrastive auxiliary loss with verified hard negatives. |
| [SetFit](sources.md#s5), [TinyBERT](sources.md#s6) | Few-shot pair training and distillation | Promising small-model recipes; transfer to decisions needs testing. | Matched teacher, examples, compute and held-out sources. |
| [Laya](sources.md#s7) | Typed heads on an encoder | Authors report 32.8–39.5 ms for one question on a T4; they also disclose weak base zero-shot typed performance and large calibration corrections. Their fine-tuned score uses the benchmark's training split. | Reproduce eligible configuration on target CPU and held-out tasks. |
| [Kev](sources.md#s8) | Qwen backbone, option pointer, state reuse | Kev-0.8B trades accuracy for size. Server cache and GPU timings are not laptop CPU timing; current Qwen3.5 questions occupy independent rows. | Compare local 0.8B if full process fits 4 GiB; expose precision and cache condition. |
| [Jev](sources.md#s9) | Typed contract and workflow decomposition | Hosted architecture and training details are limited. 70–500 ms endpoint response is not local model latency; provider workflow labels partly use model consensus. | Hosted quality comparator on independent labels, separately labelled latency. |
| [JevBench](sources.md#s10) | Shared typed-decision harness and sealed tasks | Composite mixes intelligence, calibration, speed and estimated cost; rankings depend on weights, hardware and public-task exposure. | Pinned external view alongside our locked, hardware-matched tests. |

## Pattern and challenge

The shared primitive is **state + independent question + supplied answers → probabilities**. Avoiding sequential text generation can be efficient. Reading a state once across questions may help, but fine evidence can be lost in a pooled representation. A compact model can mistake semantic resemblance for proof. The [architecture](architecture.md) makes this trade-off measurable.

Critical questions before scaling: does reuse actually reduce end-to-end time for 1 versus 8 questions and 2 versus 20 options? Can it spot negation and missing evidence? Does contrastive training improve genuinely unseen label descriptions? Is calibration reliable after a domain or language shift? What happens at 2,048 tokens and beyond? Each question has a slice in the [evaluation contract](evaluation.md).

Laya's calibration and option-order notes and Kev's held-out new-source results are authors' measurements on different setups. They motivate hypotheses; they do not establish that the proposed model will outperform either one.
