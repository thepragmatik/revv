# Registered Kaggle CPU preflight for a pretrained labelled-data pilot

**[Plain explanation](../../docs/start-here.md) · [Lexical quality floor](../data/2026-09-27-clinc-lexical.md) · [Idea ledger](../../docs/ideas.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete.** This checks the model/data acquisition path without using Kaggle GPU quota. It is not itself a decision experiment.

The proposed frozen encoder comparison uses Apache-2.0 [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) at exact revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with the same pinned CLINC source, cleaned development/calibration folds and 5% in-scope calibration defer target as the lexical baseline. Before GPU allocation, a private internet-enabled **CPU-only** Kaggle kernel downloads only the pinned encoder config/tokenizer and 2.5 MB public dataset, verifies the CLINC Git blob, records hashes and checks whether `transformers` is present. It does **not** download model weights, fit a model, expose examples or claim quality.

Source/data links and license are recorded in [the audit](../data/2026-09-27-clinc-audit.md). Kernel execution is capped at 180 seconds, collector at eight minutes, Actions at 15 minutes; no paid compute. If network, expected schema or package access fails, stop and fix on CPU before trying a GPU pilot. Then register the real pooled-versus-token comparison with fixed metrics, source revision and pilot cap.

## Result

The [Actions run](https://github.com/thepragmatik/revv/actions/runs/36293210784) collected the original private CPU kernel `rathworx/revv-internet-preflight/1`. The [raw report](2026-09-27-internet-preflight.json) verifies pinned model config/tokenizer fetches, pinned CLINC Git blob and Python `3.12.13`; Kaggle had `transformers 5.0.0` installed and reported `ready_for_gpu_pilot=true`. It did not load weights, import model classes or run inference, so weight acquisition and Transformers 5 compatibility are still risks for the bounded GPU pilot. No GPU quota was consumed by this preflight.
