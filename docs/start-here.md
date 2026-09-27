# The idea in plain English

**[Home](../README.md) · [Technical design](architecture.md) · [Evaluation](evaluation.md)**

A help desk receives: “My parcel is late, and I was charged twice.” A decision model can answer “Which team should see this?” and “Does this need urgent review?” It returns answer probabilities; ordinary program rules or a person choose the action. A probability is useful only if it reflects real error rates on the work at hand.

```mermaid
flowchart TD
    S["Read the ticket once"] --> Q["Ask: billing, shipping or other?"]
    S --> U["Ask: urgent review?"]
    Q --> P["Answers and probabilities"]
    U --> P
    P --> R{"Enough evidence?"}
    R -->|Yes| A["Apply a program rule"]
    R -->|No| H["Human review"]
```

The sharper idea we will screen combines a shared state encoding, cheap typed/evidence scores, verified fact-edit masks, and a risk-gated verifier that spends extra compute only on fields likely to benefit. AdaMTL already has task-aware compute over a shared encoder; QA cascades and Nimble cover related pieces. We will treat this as a research hypothesis, not a novelty claim. Keep it only if it improves local CPU risk/latency against fixed cascades and compact models. [The plain-language novelty audit](novelty-audit.md), [technical derivation](architecture.md), [research map](research.md) and [gated test plan](next-steps.md) explain the idea and falsifiers.

Our next small experiment asks: after one fact changes, can we pick which saved answers need checking better than confidence or a simple similarity score? The fresh synthetic sample-size check found 74 changed fields in 22 held-out rule groups, enough to pass the registered count screen but not a power analysis. No model scores exist. The first two collection windows timed out in the queue; later attempts exposed and fixed a missing helper import and a CUDA initialization-order issue. The fixed version 3 then stayed queued for all 56 checks in its 14-minute window. A read-only collector is checking that same version, without submitting another kernel. These setup and queue outcomes say nothing about the model. Even a positive result would be an early synthetic clue, not proof that the idea is new or faster on your device. A later, untested idea is to refresh the changed sentence first, then spend slower verification only on answers where it is expected to prevent enough mistakes to justify the extra CPU time. [Idea ledger](ideas.md) · [Technical design](architecture.md). [Friendly experiment note](../experiments/kaggle/2026-09-27-intervention-risk-probe.md) · [Power check](../experiments/kaggle/2026-09-27-intervention-risk-power-check.json) · [packaging/runtime correction](../experiments/kaggle/2026-09-27-intervention-risk-packaging-correction.json) · [version 3 Action](https://github.com/thepragmatik/revv/actions/runs/36319051136) · [read-only recovery Action](https://github.com/thepragmatik/revv/actions/runs/36319968934).

```mermaid
flowchart TD
    E["One fact changes"] --> R["Rank saved answers"]
    R --> B{"Use a fixed check budget?"}
    B -->|selected| V["Recheck with edited evidence"]
    B -->|not selected| C["Keep cached answer"]
    V --> M["Compare accuracy and CPU time"]
    C --> M
```



The main local limit is **under 8 GiB peak memory for the entire process**, including the tokenizer, runtime and working buffers. We will also test a **quantized under-4 GiB version** for smaller devices. This is a separate profile: quantization might change both speed and answers. A small model file alone does not prove either version fits. The first reference device is a laptop-class CPU. We will measure cold start, warm requests, multiple questions and long text separately; GPU and hosted API timings get their own comparisons. [See the memory and compute arithmetic](feasibility.md).

The first phase covers **English text**, yes/no, choices among supplied options, and ordered scores. It must be able to say “none of these” or “insufficient evidence” where appropriate. “Decision” here means judging supplied text against a criterion; it does not mean open-ended planning or unlimited factual recall. We will measure other languages before claiming to support them.

To win, the model must fit, answer new and difficult cases correctly, give trustworthy probabilities, and be faster **on the same machine** as a comparable model. [Evaluation rules](evaluation.md) and the [roadmap](roadmap.md) spell that out. No trained neural decision model has been validated, and local CPU deployment has not yet been measured.

## What the first experiments found

| Plain-English question | Finding on public development data | What it means |
| --- | --- | --- |
| Can a small frozen encoder recognize short requests? | With examples of each intent from training, **91%** of 1,497 known requests were classified correctly; it rejected **43 of 50** unknown requests. | Promising starting point. It has not been compared fairly with existing decision models on a new source or local CPU. [Details](../experiments/kaggle/2026-09-27-prototype-screen.md). |
| Does comparing individual words help by itself? | Untrained token matching did **worse** than a simple sentence summary and cost more scoring work in the GPU pilot. | Save this complexity until a trained version shows a reason for it. [Details](../experiments/kaggle/2026-09-27-frozen-matching.md). |
| Can we find relevant clauses in a long contract? | Matching question words found a marked contradiction clause among five candidates about half the time. Examples of marked clauses from training raised this to about **95%**. | Finding a clause is easier with examples, but it does **not** answer whether the contract supports or contradicts a claim. Only about **64%** of positive cases have *every* marked clause in the first five. [Details](../experiments/data/2026-09-27-contractnli-evidence-prototypes.md). |

| Can we build reliable change masks? | A small synthetic integrity probe created 256 rule states in 32 composition groups and passed five checks. | This verifies answer/proof/mask plumbing only; it does not measure a model's quality or speed. [Details](../experiments/data/2026-09-27-intervention-mask-probe.md). |
| Can a cheap word search find all the proof clauses? | On held-out generated rule compositions, it found at least one proof clause in the first five for every supported query, but all proof clauses for only 24%. | This is a synthetic TF-IDF result, not user text or model accuracy. It flags evidence coverage as a gating metric. [Details](../experiments/data/2026-09-27-intervention-retrieval-floor.md). |

The first matched three-way classifier rose from **68.85% to 70.20%** accuracy when given better-retrieved clauses, but the uncertainty interval includes no gain. A per-question guess from training frequencies alone already reached **68.08%**. An established frozen language-inference model scored on the **same retrieved clauses** reached **70.01%** and did worse at contradictions. On cases where the answer was known to be either support or contradiction, replacing retrieved clauses with the **correct marked clauses** raised its accuracy from **57.65% to 67.92%**; even then it fell below an **84.53%** always-support guess on raw accuracy. The marked-clause result is an **oracle diagnosis**, not a working model: real requests do not supply the correct clauses or the answer type. Better evidence helped, but interpreting that evidence still needs work. We are pausing neural architecture training while we test cheaper controls. [Exact diagnosis](../experiments/kaggle/2026-09-27-nli-gold-stances.md) · [Live handoff](../codex/RESUME.md) · [Idea ledger](ideas.md).

The cheap [evidence check](../experiments/data/2026-09-27-contractnli-evidence-shuffle.md) supports that caution: the three-way classifier scored **70.20%** with its own contract clauses, **68.08%** with only the fixed question, and **66.35%** after replacing its clauses with another contract's. Reading the right contract helps a little and changing clauses changes answers; the improvement missed the preset bar for further model training. These are public development results, not a local speed or reliability claim.
