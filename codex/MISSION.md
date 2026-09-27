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

The first proposal combined shared state, candidate scoring, criterion decomposition, evidence matching and counterfactual training. Direct work now covers much of that ground: Primus Decision targets tiny CPU typed decisions; RSI-Jev and Typical score typed options over shared state; Nimble publishes multi-field context reuse and fact-edit curation; RULERS compiles criteria and verifies evidence.

The next generic proposal—independent per-question exits over shared state—is too close to AdaMTL, which already learns task-aware compute policies over a shared encoder and merges task demands, and to progressive QA cascades. The exact shared-cost depth objective has an O(Q L) solver that matched exhaustive search on 3,000 randomized small cases; this is algebra validation only and remains a baseline.
The higher-upside hypothesis to screen is **intervention-supervised, task-risk-gated verification over a shared long state**: encode once, cheaply match typed fields/options to evidence, use verified fact edits to teach which fields should change, and run a costly verifier only when predicted gold-task error reduction justifies the measured CPU time. CLIP-style matching is conditional on exact evidence and retrieval headroom. Nimble, PairCFR, QA contrast consistency, AdaMTL and cascade ranking are close prior art; the possible contribution is the precise sparse-field-mask plus task-risk allocation interaction, and it is not yet a novelty claim.

Next, compare code and objectives, build grouped RuleTaker/ProofWriter interventions with theorem-prover labels/evidence, profile shared encoding versus field-verifier costs on CPU, and test cheap-only/fixed/risk-gated controls at Q=1/5/20 before training. ContractNLI remains a realistic stress set because earlier evidence retrieval did not establish a deployable path. Bundle any-error risk, gold correctness and model agreement stay separate. See [the novelty audit](../docs/novelty-audit.md), [hypothesis ledger](../docs/ideas.md), [technical derivation](../docs/architecture.md), [research map](../docs/research.md) and [gated plan](../docs/next-steps.md).
