# Feasibility screen before renting a GPU

**[Home](../README.md) · [Plain explanation](start-here.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md)**

This is a **mathematical filter, not a benchmark**. It can reject impossible weight budgets and identify where shared-state computation might pay. It cannot establish accuracy, calibrated probabilities, actual RSS or milliseconds. Reproduce the tables with [`python analysis/screen.py`](../analysis/screen.py) from the repository root. Parameter counts below are nominal, and the 100M body is illustrative.

## 1. Weight floor versus the local memory limit

For `P` parameters stored at `b` bits each, the idealized weight floor is `W = P·b/(8·2³⁰) GiB`. Quantized formats require additional scales, zero points, packing metadata and often unquantized layers. Resident process memory also includes runtime, tokenizer and working buffers.

| Nominal size | 16-bit weight floor | 8-bit weight floor | 4-bit weight floor | 8 GiB minus 16-bit weights |
| --- | ---: | ---: | ---: | ---: |
| 100M | 0.19 GiB | 0.09 GiB | 0.05 GiB | 7.81 GiB |
| ModernBERT-base, ~149M | 0.28 | 0.14 | 0.07 | 7.72 |
| Laya-sized, ~421M | 0.78 | 0.39 | 0.20 | 7.22 |
| 0.8B | 1.49 | 0.75 | 0.37 | 6.51 |
| 4B | 7.45 | 3.73 | 1.86 | 0.55 |
| 9B | 16.76 | 8.38 | 4.19 | −8.76 |

**Inference:** a nominal 4B 16-bit model is a poor 8 GiB process candidate: only ~0.55 GiB remains for everything else. A 4-bit 4B model deserves a *measured* feasibility check, as does 4-bit 9B; neither has a guaranteed fit, accuracy or CPU speed. Keep the **primary 8 GiB** and **compact quantized 4 GiB** leaderboards separate. [ModernBERT's model family](sources.md#s1), [Laya's checkpoints](sources.md#s7) and [Kev's family](sources.md#s8) motivate the nominal rows; the rows are approximate counts, not a claim about their shipped storage format.

## 2. When reading the state once can save work

Let state length `L=128`, candidate `(question + option)` length `m=32`, `N=Q·K` candidates and an illustrative same-size `P=100M` encoder body. If linear layers dominate, a shared-state design takes approximately `2P(L+Nm)` operations; a cross encoder that repeats the full state takes `2PN(L+m)`. This **ignores** attention, embedding lookups, extra scorer work, cache reuse, batching, tokenization and device-specific kernels. It assumes equal encoder capacity and thus does not predict equal accuracy.

| Questions × options | Shared linear proxy | Repeated cross-encoder proxy | Arithmetic ratio |
| --- | ---: | ---: | ---: |
| 1 × 2 | 38.4 GFLOPs | 64.0 GFLOPs | 1.67× |
| 1 × 5 | 57.6 GFLOPs | 160.0 GFLOPs | 2.78× |
| 8 × 5 | 281.6 GFLOPs | 1,280.0 GFLOPs | 4.55× |

The ratio `N(L+m)/(L+Nm)` grows with repeated questions and saturates near `(L+m)/m=5` in this short-text example. For one yes/no question the theoretical saving is modest; the small cross encoder is a serious contender. For many questions over a long document, state reuse merits a controlled test. A full token-interaction scorer could erase part of the saving; include it only after a quality failure points to missing evidence. Running quantized 4B weights through a CPU also has a bandwidth and kernel cost, so memory fit alone cannot justify a speed claim.

## 3. Ranked shortlist and rejection rules

| Priority | Candidate | Why it earns a low-cost test | What would stop it |
| --- | --- | --- | --- |
| 1 | Pretrained ~100–150M cross encoder plus lexical control | Strong evidence interaction; cheap and fast to fine-tune relative to larger models | Weak quality on new domains or CPU p95. |
| 2 | Shared-state small/base encoder, first pooled then token interaction | Arithmetic advantage grows with question count; test on 1 and 8 questions | Lost negation/answerability accuracy or no end-to-end speedup. |
| 3 | Existing Laya-sized encoder and quantized Kev-0.8B/4B, plus current leading open decision models | Quality ceilings and real competitors under the new cap, **without training them first** | Actual RSS above profile cap or worse speed/quality trade-off. |
| Deferred | Full pretraining, image/text tower, broad RL or 9B student training | No image input in phase 1; no evidence these expenses solve observed errors | Revisit only with a registered failure, data rights and measured budget. |

Reuse pretrained representations. Try supervised loss before contrastive or distillation, and inspect public model authors' honest limitations: [Laya discloses weak base zero-shot performance and calibration repair](sources.md#s7); [Kev reports new-source gaps by size](sources.md#s8). Neither author result predicts our workload; it helps choose sensible controls.

## 4. What compute is currently reachable

In the present workspace we can run Python arithmetic, dataset and overlap audits, CPU-only scripts and small smoke checks, and edit the connected GitHub repository. **There is no GPU here or connected GPU-provider account.** GitHub's [standard runners for public repos](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) provide free CPU CI (listed Linux runner: 4 CPUs, 16 GB); they are useful for reproducibility and modest experiments, **not an official laptop latency result**. Action execution depends on the repository's workflow settings.

For a later user-connected GPU, Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) describes free access with fluctuating limits and no guaranteed runtime. [Modal's live pricing](https://modal.com/pricing) currently lists T4 at `$0.000164/s`, L4 at `$0.000222/s`, H100 at `$0.001097/s`, and a Starter allowance of `$30/month` free compute; a 10-minute GPU-only pilot at those rates is approximately **$0.10 / $0.13 / $0.66** respectively, *before* CPU, RAM and any other charges. These are illustrative rates, not a provisioned account or a spending authorization. [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) is presented for Spaces demos and shared inference, not a guaranteed training allocation.

**Spending gate:** first measure free CPU baselines and a one-batch smoke. Register a maximum budget and an abort threshold in the [experiment template](../templates/experiment.md). Only use paid compute when a specific comparison cannot be resolved by published weights, algebra, free CPU tests or a bounded free allocation. Keep actual training memory separate from the 8 GiB **deployment** constraint.
