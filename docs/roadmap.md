# Research roadmap and spending gates

**[Home](../README.md) · [Plain explanation](start-here.md) · [Design](architecture.md) · [Evaluation](evaluation.md) · [Executable next steps](next-steps.md)**

~~~~mermaid
flowchart TD
    A["Revisit prior art and freeze task"] --> B["Audit data and estimate cost"]
    B --> C["Run matched CPU baselines"]
    C --> D{"Candidate survives cheap tests?"}
    D -->|No| E["Record failure and revise"]
    D -->|Yes| F["One small free GPU pilot"]
    F --> G["Calibrate and red-team"]
    G --> H{"Quality, speed and memory gates pass?"}
    H -->|No| E
    H -->|Yes| I["Export and publish evidence"]
~~~~

| Phase | Artifact | Exit gate |
| --- | --- | --- |
| 0. Mission and novelty | [Original prompt](../codex/ORIGINAL_PROMPT.md), primary-source overlap ledger and candidate board | No established component is mislabelled as novel; one falsifiable candidate survives. |
| 1. Data and cost audit | Exact schemas, rights/splits, rule-composition manifest, analytical compute and memory estimate | No leakage or invalid label conversion; candidate has plausible headroom. |
| 2. Matched baselines | CPU quality/latency/RSS for Primus, RSI-Jev, compact encoder, direct cross scorer, Laya, Kev, Typical and Decider where eligible | Same inputs/hardware; identify a real gap or stop without training. |
| 3. No-training discriminator | Layerwise typed probes, per-question exit depth and maximum probability drift across the whole bundle | Useful early exits survive calibration and task-quality checks. |
| 4. Bounded prototype | One compact student, one-batch smoke and short free-Kaggle pilot | Paired gains survive a natural source and an unseen rule composition. |
| 5. Calibrate and red-team | Separate calibration split, counterfactuals, model export and CPU profiles | Registered quality/calibration/risk gates and full-process <8 GiB gate pass; separate quantized <4 GiB gate measured. |
| 6. Report | Reader-friendly explanation, expert method, configs, hashes, raw aggregates and failure analysis | Every win is tied to a named model, revision, task, CPU, precision, uncertainty and timing boundary. |

Before any experiment, use the [experiment template](../templates/experiment.md) to record hypothesis, one changed variable, exact hashes, maximum runtime and free-provider quota. Start with dataset checks and algebra, then a short inference or one-batch smoke. No cost estimate in this roadmap authorizes paid compute.

The current lead is a **research hypothesis**: share one state pass across multiple typed questions, let each question branch stop at its own depth, and calibrate the maximum intermediate-to-full drift over the whole question bundle. Shared state, early exit and conformal stopping each have prior art; the combination is not yet checked exhaustively. It may fail because early predictions stabilize too late, one hard question keeps the state at full depth, calibration is too conservative, or control overhead erases the savings. See the [novelty review](research.md) and [idea ledger](ideas.md).

**Open project choices:** exact target CPU, domain/error costs, whether 8 GiB means full host or model process, English-only phase-one use, and dataset rights. Until a target is provided, select a named reproducible reference CPU and limit claims to that machine. Free Kaggle T4 is for a justified training pilot; it does not substitute for local CPU measurement.
