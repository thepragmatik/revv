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

Can a workload-aware decision encoder reuse a state across many options, retain the evidence sensitivity of a cross encoder, and calibrate abstention while beating a strong eligible baseline? Test when to use a direct cross pass, shared pooled scoring, and selective token interaction. The existing [synthetic screen](../experiments/kaggle/2026-09-27-forward-pilot.md) motivates the crossover question but provides no decision quality evidence. Record proposed mechanisms and pre-run falsifiers in [architecture hypotheses](../docs/architecture.md) and experiment records; never conflate an idea with a validated architecture.
