# Research charter: invent, falsify, measure

**[Original prompt](ORIGINAL_PROMPT.md) · [Live status](RESUME.md) · [Work rules](WORK_RULES.md) · [Research](../docs/research.md) · [Architecture](../docs/architecture.md)**

The user's original request asks for an unusually **fast, accurate, genuinely useful local decision model**, developed through original synthesis of ModernBERT-style encoders, CLIP-style contrastive objectives, and System One decision models such as Laya and Kev. We are expected to **propose and test novel architectural ideas**, not merely reproduce or fine-tune existing models. Treat novelty as a research hypothesis until prior art is checked; report similarities honestly.

Work like a rigorous, pragmatic researcher:

1. Pause before each research or compute cycle: restate the real decision problem, what is known, what could be wrong, and what observation would change the next choice.
2. Derive costs and invariants from first principles, then check second-order effects: batching, calibration, tokenization, data provenance, hardware, counterexamples and user cost. Reject infeasible ideas on paper before expensive runs.
3. Combine established mechanisms only when they solve a measured failure or uncover a falsifiable trade-off. Maintain an idea ledger with provenance, mathematical prediction, closest known alternatives, cheap discriminating test, failure modes and go/no-go criterion. Search primary prior art before describing an idea as novel.
4. Red-team each promising result: leakage, shortcut learning, option reordering, negation, missing evidence, shifted source, calibration, concurrency and adversarial workload. Prefer an explicit negative result to a fragile win.
5. Keep plain-English and expert artifacts linked both ways; use concise Mermaid where a mechanism or experimental decision is clearer visually. Keep raw data and precise methods accessible to experts.

The measurable goal is a **local CPU decision process under 8 GiB peak process RSS** that wins justified comparisons of quality and latency against eligible existing models on the same tasks and hardware. Maintain a separately labelled **quantized under-4 GiB profile** for constrained devices. “Fastest” and “beats” require declared workloads, hardware, uncertainty and comparator set; the current documents define those gates. A GPU pilot or arithmetic proxy cannot establish the final claim.

## Later decisions

The original prompt set a less-than-4 GB goal. The user later raised the primary memory ceiling to **8 GB** and explicitly allowed quantization for smaller devices. The 4 GB compact profile remains a distinct target. The user uses ChatGPT Pro/Codex as the research and experiment controller, with free Kaggle GPU through the repo's GitHub Actions secret; Colab is an optional path, not a requirement. The user wants the repo handoff to remain current and wants **regular progress reports in the active conversation** while research and focused experiments continue.

## Current research question

The first proposal combined shared state, candidate scoring, criterion decomposition, evidence matching and counterfactual training. New direct sources narrow the novelty space: Primus Decision already targets tiny local CPU typed decisions; RSI-Jev and Typical directly score typed options over shared state; Nimble publishes counterfactual evidence-pair curation; RULERS compiles criteria and verifies evidence. Treat these as prior art and baselines.

The remaining candidate is an unverified efficiency question: when one state has several independent typed questions, can each question branch exit at a different layer while reusing a shared state pass, under a calibration bound over all questions and exit layers in that request bundle? Compare it with fixed-depth shared scoring, static shallow taps, independent early exits and full-depth inference. Early exit, cache reuse and conformal consistency are established; the exact typed multi-question combination is not claimed new.

The stopping rule could preserve agreement with the same model's full-depth answer under an exchangeability-based drift bound; it cannot ensure ground-truth correctness. First run exact prior-art review, data audit, layerwise no-training probes and matched CPU baselines. Train only if a useful speed/quality gap remains. Do not start paid compute. See [the revised hypothesis](../docs/ideas.md), [technical derivation](../docs/architecture.md), [research map](../docs/research.md) and [next gates](../docs/next-steps.md).
