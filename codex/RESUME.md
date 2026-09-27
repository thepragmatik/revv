# Resume brief — 27 September 2026

**[Handoff index](README.md) · [Pasteable prompt](PROMPT.md) · [Working rules](WORK_RULES.md) · [Plain explanation](../docs/start-here.md) · [Research sources](../docs/sources.md)**

## Mission and target

Build and test a fast local text **decision model**: given a state and independent typed questions, return choice, yes/no or ordered-score probabilities, with a way to abstain when evidence is missing. Favor a single state encoding reused across questions only when it earns its speed *and* quality. Deployment gate is peak entire local CPU process RSS **below 8 GiB**; a separate quantized compact profile is **below 4 GiB**. English text is phase one; do not infer image, language or open-ended planning support. See the [reader-friendly overview](../docs/start-here.md) and [technical architecture](../docs/architecture.md).

The user has ChatGPT Pro and wants Codex to research, orchestrate and evaluate GPU experiments. Kaggle supplies free GPU capacity separately; Codex is the controller. The user has verified the Kaggle account by phone and already set `KAGGLE_API_TOKEN` as a **GitHub Actions repository secret**. The Actions bridge works; do not ask for the token again. Free Colab agent integration is optional and unvalidated here. Never spend money or start a long train cycle without a specific registered budget.

## Verified evidence and current status

| Item | Status | Pointer |
| --- | --- | --- |
| Research plan, datasets, competing models, claim gates | Draft; many choices deliberately open | [Research](../docs/research.md), [roadmap](../docs/roadmap.md), [evaluation](../docs/evaluation.md) |
| Architecture arithmetic and weight floors | Reproducible analytical proxies; **not** measured quality or latency | [Feasibility](../docs/feasibility.md), [`analysis/screen.py`](../analysis/screen.py) |
| Kaggle authentication and GPU access | Verified; third smoke run executed on one T4 | [Run 36290758927](https://github.com/thepragmatik/revv/actions/runs/36290758927), [raw JSON](../experiments/kaggle/2026-09-27-gpu-smoke.json) |
| Decision model weights, real dataset benchmark, CPU RSS | **Not yet measured** | [Evaluation gates](../docs/evaluation.md) |
| Synthetic forward-pass crossover pilot | **Complete.** Original T4 kernel recovered without a second GPU run: cross faster for 128 tokens/2 options; shared faster on four other slices. No accuracy or CPU result. | [Experiment record](../experiments/kaggle/2026-09-27-forward-pilot.md), [raw JSON](../experiments/kaggle/2026-09-27-forward-pilot.json), [recovery run 36291800978](https://github.com/thepragmatik/revv/actions/runs/36291800978) |
| First public labelled dataset integrity | CLINC train/validation audit registered; CPU run pending. Test examples remain uninspected. | [Audit plan](../experiments/data/2026-09-27-clinc-audit.md), [script](../analysis/audit_clinc.py) |

The smoke run performed eight tiny matrix multiplications. Its submillisecond GPU time and memory counters are **not** model or CPU numbers. Do not turn it into a performance headline.

The working GitHub branch is `research/decision-model-plan`; [draft PR #1](https://github.com/thepragmatik/revv/pull/1) targets `main`. The pilot recovery ran at `d7cdbe3934d73db871c62c5bc7846a1d274ad618`; the result-record commit is newer. **Fetch the live PR head before editing.** The GitHub connector has been used to commit to the PR branch; this scratch workspace need not be a Git clone.

## Next concrete actions

1. Collect the [pinned CLINC train/validation audit](../experiments/data/2026-09-27-clinc-audit.md). Use its duplicate and domain findings to freeze group-aware train/dev/calibration/test partitions and lexical/majority controls. CLINC tests out-of-scope intent; it does not cover policy grounding or multiple questions. No data or benchmark training has happened yet.
2. Run a real pretrained small cross-encoder baseline and a pooled shared-state control on *the same labelled cases*. Start with one-batch tests, bounded Kaggle pilot, then local CPU RSS/end-to-end latency on a specified machine. Keep a locked test sealed until the candidate and margins are registered. The [synthetic pilot](../experiments/kaggle/2026-09-27-forward-pilot.md) justifies testing reuse at large option counts, while retaining cross encoding for short two-option cases.
3. Compare to eligible published Laya, Kev and leading open decision models under a matched contract, precision and local RSS. Broader CLIP-style contrastive or distillation work comes only after actual errors identify a reason.

**Update discipline:** after a run, record the observed versus predicted crossover, timing distributions, exact hardware, limits and any errors in its experiment record. Add the commit/run/artifact links here, change the status and the next action. Do not silently overwrite preregistered gates. Link technical notes back to the [plain explanation](../docs/start-here.md). The docs and templates remain the deeper source of truth.
