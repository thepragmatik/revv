# Working rules for Codex

**[Resume](RESUME.md) · [Roadmap](../docs/roadmap.md) · [Evaluation](../docs/evaluation.md) · [Experiment template](../templates/experiment.md)**

1. Read the actual repo and recent Actions artifacts first. Verify links, commit SHA, exact workload, versions and raw results. Treat this handoff as a dated snapshot.
2. Before each run, register a narrow question, control, falsifying outcome, raw outputs, timeout and budget in `experiments/`. Never read a locked test to choose a model.
3. Prefer algebra, published weights, dataset audit and a one-batch smoke before a train or download cycle. Change one important factor at a time and keep failures and partial runs.
4. Use Python for research and PyTorch training; use ONNX Runtime for first local CPU inference. Rust is optional only after measured overhead. Compare identical inputs and hardware, reporting process RSS, cold/warm end-to-end p50/p95/p99, quality, calibration and error slices.
5. Primary target: **peak entire local CPU process RSS <8 GiB** on a declared workload. Secondary quantized target: **<4 GiB** and a measured quality delta. A Kaggle T4 speed or device-memory figure is neither target.
6. The Kaggle API token exists as the GitHub Actions repository secret `KAGGLE_API_TOKEN`. Never print or commit it. Keep kernels private and internet disabled unless a registered data/model acquisition needs access and licenses allow it. Respect free-provider quotas. Paid compute needs an explicit budget from the owner.
7. Distinguish synthetic compute screens, real decision quality, calibrated probabilities, and local CPU deployment results. Do not say "beats" without the matched-task, matched-hardware, uncertainty and memory gates in [`docs/evaluation.md`](../docs/evaluation.md).
8. Update `codex/RESUME.md` on every meaningful result: date, commit, run ID, artifact, inference versus observation, failures, decision, and next smallest discriminating test. Add a plain-English result and technical raw link in the same experiment record. Keep navigation links working.

Open decisions: target domain and error costs, exact reference laptop CPU, whether 8 GiB also bounds training, and dataset use rights. The current working assumptions (English typed text decisions, 8 GiB local CPU inference RSS) permit cheap research while these remain open.
