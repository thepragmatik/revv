# Registered experiment: does shared-state inference save T4 forward time?

**[Plain explanation](../../docs/start-here.md) · [Architecture](../../docs/architecture.md) · [Evaluation](../../docs/evaluation.md) · [Handoff](../../codex/RESUME.md)**

**Registered 27 September 2026 before submission. Status: registered.** This is a synthetic architecture screen, not a trained decision model or local CPU benchmark.

## Question and falsifier

Given identical untrained transformer body weights, does encoding the state once and scoring candidate vectors beat encoding each state+candidate pair together as candidate count grows? The analytical linear-work proxy predicts ratios of `N(L+32)/(L+32N)` in favor of reuse. It omits batching, attention and launch costs; the measured ratio can disagree. A cross encoder might win at low `N`, and a GPU's parallelism might erase reuse benefits at high `N`. This screen cannot measure accuracy, so an apparent speedup cannot select the final architecture.

## Registered procedure

- Device: one Kaggle T4, private internet-disabled kernel; PyTorch available in Kaggle runtime. No downloads, datasets, training or paid compute.
- Body: same untrained two-layer 256-wide four-head encoder with 1,024-wide feed-forward layers and zero dropout; random seed 271. State is `L×256`, option is `32×256`; shared path encodes state once and all options as a batch; repeated cross path encodes all concatenated state+option pairs as a batch.
- Predeclared slices: `(L,N)` = `(128,2)`, `(128,40)`, `(512,2)`, `(512,40)`, `(2048,8)`. Two warm-up pairs and five timed pairs per slice, alternating path order. Synchronized wall time includes GPU work and Python dispatch, excludes tokenizer. Keep raw timings and median ratio `cross/shared`.
- Pilot cap: script stops adding slices at 210 seconds; Kaggle runtime cap 300 seconds; GitHub Actions cap 20 minutes, collector poll cap 12 minutes. If incomplete, record partial rows and rerun only after a specific fix. Worst-case consumed GPU time <5 minutes for one run (subject to provider scheduling); no monetary spend authorized.
- Decision gate: if the shared path is consistently faster in high-`N` slices, keep it for a **labelled-data quality comparison**; if not, prioritize the simpler cross encoder on those slices. Keep cross encoder as a quality control regardless. Do not infer CPU p95 or production memory from T4 timing. A speed ratio from five samples has no confidence interval.

## Result and limitations

The [first Actions attempt](https://github.com/thepragmatik/revv/actions/runs/36291723816) submitted kernel version 1, then immediately failed to poll it: its human title resolves to a different URL slug than the submitted ID. **No result was collected, and the kernel may still have run.** A read-only recovery now checks the title-derived slug before deciding whether a second submission is needed. The registered slices and gates above are unchanged. Raw result will be attached to a subsequent Actions run and, if valid, committed beside this record. Outputs are different mathematical functions, with a shared compute body only; this is a **cost topology comparison**, not functional parity or a head-to-head quality contest.
