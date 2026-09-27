# Registered experiment: does shared-state inference save T4 forward time?

**[Plain explanation](../../docs/start-here.md) · [Architecture](../../docs/architecture.md) · [Evaluation](../../docs/evaluation.md) · [Handoff](../../codex/RESUME.md)**

**Registered 27 September 2026 before submission. Status: complete.** This is a synthetic architecture screen, not a trained decision model or local CPU benchmark.

## Question and falsifier

Given identical untrained transformer body weights, does encoding the state once and scoring candidate vectors beat encoding each state+candidate pair together as candidate count grows? The analytical linear-work proxy predicts ratios of `N(L+32)/(L+32N)` in favor of reuse. It omits batching, attention and launch costs; the measured ratio can disagree. A cross encoder might win at low `N`, and a GPU's parallelism might erase reuse benefits at high `N`. This screen cannot measure accuracy, so an apparent speedup cannot select the final architecture.

## Registered procedure

- Device: one Kaggle T4, private internet-disabled kernel; PyTorch available in Kaggle runtime. No downloads, datasets, training or paid compute.
- Body: same untrained two-layer 256-wide four-head encoder with 1,024-wide feed-forward layers and zero dropout; random seed 271. State is `L×256`, option is `32×256`; shared path encodes state once and all options as a batch; repeated cross path encodes all concatenated state+option pairs as a batch.
- Predeclared slices: `(L,N)` = `(128,2)`, `(128,40)`, `(512,2)`, `(512,40)`, `(2048,8)`. Two warm-up pairs and five timed pairs per slice, alternating path order. Synchronized wall time includes GPU work and Python dispatch, excludes tokenizer. Keep raw timings and median ratio `cross/shared`.
- Pilot cap: script stops adding slices at 210 seconds; Kaggle runtime cap 300 seconds; GitHub Actions cap 20 minutes, collector poll cap 12 minutes. If incomplete, record partial rows and rerun only after a specific fix. Worst-case consumed GPU time <5 minutes for one run (subject to provider scheduling); no monetary spend authorized.
- Decision gate: if the shared path is consistently faster in high-`N` slices, keep it for a **labelled-data quality comparison**; if not, prioritize the simpler cross encoder on those slices. Keep cross encoder as a quality control regardless. Do not infer CPU p95 or production memory from T4 timing. A speed ratio from five samples has no confidence interval.

## Result and limitations

The [first Actions attempt](https://github.com/thepragmatik/revv/actions/runs/36291723816) submitted kernel `rathworx/revv-bounded-forward-pilot/1`, then immediately failed to poll it: its human title resolved to a different URL slug than the requested ID. The [read-only recovery run](https://github.com/thepragmatik/revv/actions/runs/36291800978) polled that title-derived slug, found the **completed original kernel**, and collected the [raw timing JSON](2026-09-27-forward-pilot.json). No second GPU session was started. The workflow is now manual-only so changing the result record does not automatically rerun GPU work.

| State tokens, options | Shared median, ms | Cross median, ms | Cross/shared | Linear-work proxy |
| --- | ---: | ---: | ---: | ---: |
| 128, 2 | 1.36 | 1.15 | 0.84× | 1.67× |
| 128, 40 | 3.01 | 10.36 | 3.44× | 4.55× |
| 512, 2 | 2.13 | 2.91 | 1.36× | 1.89× |
| 512, 40 | 2.57 | 28.48 | 11.07× | 12.14× |
| 2,048, 8 | 6.18 | 38.38 | 6.21× | 7.22× |

All five registered slices completed in one T4 session using PyTorch `2.10.0+cu128`. The body has **1,579,520 untrained parameters**; the pilot recorded five paired timings per slice after two warm-up pairs, with alternating execution order. For the short state with two options, the cross path was faster despite the arithmetic proxy. State reuse was faster in the other slices, particularly at 40 options. A measured 512-token, 40-option shared median below the 128-token, 40-option median also shows why tiny GPU timings should not be extrapolated monotonically. The [raw JSON](2026-09-27-forward-pilot.json) preserves all samples and exact hardware; the GitHub Actions artifact is [here](https://github.com/thepragmatik/revv/actions/runs/36291800978).

**Verdict against the preregistered gate:** keep shared-state scoring in the *next labelled-data comparison* for high option counts; retain the cross encoder as a serious control for small workloads and quality. Five timings from one GPU session have no confidence interval. These paths implement different functions even though the body weights are shared. Synthetic vectors omit tokenization, actual pretrained checkpoints, probabilities, labels, CPU process RSS and CPU end-to-end p95. No claim about decision quality, superiority to Laya/Kev or the 8 GiB deployment target follows.
