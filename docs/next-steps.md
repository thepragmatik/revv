# Next steps: a novelty-led, low-cost decision-model programme

**[Home](../README.md) · [Plain explanation](start-here.md) · [Research landscape](research.md) · [Idea ledger](ideas.md) · [Architecture](architecture.md) · [Codex handoff](../codex/RESUME.md)**

Research snapshot: 27 September 2026. This plan follows the original mission: try hard to find a novel idea, reason from first principles, challenge it against literature and evidence, and spend compute only after a cheap test shows a reason. The main deployment goal is local typed decisions under 8 GiB peak process RSS; a separate quantized build must fit under 4 GiB.

## Updated research verdict

The earlier rubric/fact-memory proposal is too close to newly identified work to lead as a novelty claim. Nimble publishes counterfactual evidence pairs for typed choices, yes/no and score tasks; RULERS compiles criteria, checks evidence and calibrates scores; Primus Decision already gives a very small CPU typed-decision model; RSI-Jev is a sub-billion-parameter open typed scorer; Typical and other systems already reuse one state across questions.

**One focused hypothesis remains worth a low-cost test:** use a shared state pass, but let each question branch stop at a different layer when a statistically calibrated check says its output matches the same model's full-depth answer. Early exit and multi-task adaptive compute are established. The possible contribution is this particular combination of (a) reusable state across independent request-time questions, (b) separate exit depth per question, and (c) bundle-level calibration across all questions and exit layers. This is a search target, not a certified novelty gap.

The proposed method can at most preserve the full-depth model's answer while reducing compute. It cannot make an incorrect full-depth answer correct. Its useful result would be lower local latency at non-inferior task quality, probability quality and memory.

## Ordered execution plan and gates

### 0. Freeze the task and claim

Write a request manifest for state, question set, answer type, options, missing-evidence behavior, context limit and truncation policy. Distinguish whole-process RSS from host RAM. Fix the target workload, CPU, batch/concurrency, primary task mix, quality margins, calibration coverage and timing boundary before using a locked test.

**Exit gate:** every model adapter receives semantically identical requests and returns the same typed output contract. The test set and any sealed benchmark results stay unopened.

### 1. Close the nearest prior-art gap

Read the implementation and release materials for Primus Decision, RSI-Jev, Typical, Nimble, RULERS, Laya, Kev, Rev and Decider. Pin exact commit/checkpoint revisions, licenses, tokenizers, supported schema, state length, model size, memory and local runtime. Add early-exit papers CATs, CALM, LEAP and AdaMTL to the overlap map. Search specifically for multi-query or task-set adaptive exits over a shared state, not just generic early exit.

**Exit gate:** either find an identical mechanism and reframe as replication/evaluation, or document precisely which tested property remains distinct without calling it new. Do not download multi-gigabyte weights until license, runtime, memory and a valid comparison path are known.

### 2. Audit data and build the workload bundle

Use Typed Decisions for its five-questions-per-state interface and probability plumbing, while treating its small synthetic, teacher-labeled test as secondary evidence. Use RuleTaker/ProofWriter for exact held-out operator combinations. Consider RuleArena, AgentCoMa and JevBench only when schema and label provenance are clean. Keep ContractNLI as a long-document/evidence stress test, not as the sole primary task; earlier evidence-use gates failed.

Create splits grouped by state, source, template, option/rubric family and counterfactual sibling. Build workload bundles with Q=1/5/20 questions, 2/5/20 options and 128/512/2,048 state-token strata. Lock calibration and test manifests separately. Audit rights, duplicates, label provenance and benchmark exposure first.

**Exit gate:** one exact-label composition test and one independent natural typed workload support valid matched comparisons. If not, make narrower claims about the one usable task.

### 3. Run the no-training exit-headroom probe

Select the smallest open model that supports per-layer hidden states and a reproducible CPU path. Extract representations/logits at candidate depths on training and development partitions. Fit only lightweight per-depth readout probes or use the model's native candidate scorer; do not update the backbone yet.

Measure:

- final-layer quality versus each early exit by typed question and source;
- whole-bundle maximum probability drift over layers and questions;
- top-answer stability as Q grows;
- number of questions that could exit at each depth under a 95% calibrated stability target;
- score-question expected-value drift and probability metrics separately;
- storage/RSS and the cost of producing each layer's probe.

**Exit gate:** a useful fraction of question branches exits materially before full depth on an independent source, with a non-vacuous bundle calibration bound. If exits occur only near the final layer or the calibrated bound is too wide, stop before writing a serving path.

### 4. Establish matched CPU baselines

Run small lexical/encoder controls and eligible direct models: Primus, RSI-Jev 0.8B, Laya, Kev, Typical-small, Decider and Rev where exact revisions and runtimes permit. Treat Nimble's 9B model as a reference only if an 8 GiB profile can be validated. Include fixed full-depth, fixed shallow tap, independent question exits and shared-state fixed depth.

Use the same CPU, software versions, thread count, request bundles and timing boundaries. If the user's machine is not known, use a named reproducible runner and label it as such; do not generalize its speed to the user's laptop.

**Exit gate:** the adaptive shared-state path has measurable compute headroom over the strongest fitted baseline and a realistic chance to beat its end-to-end p95 at the registered quality margin. If an existing compact model already meets the target, stop rather than duplicate it.

### 5. Build one bounded candidate and train only if needed

Only after the probe and baseline gates pass, add per-depth decision heads to one small open model. Start with distillation from the model's own final-depth distributions plus exact/human labels. Add one change per arm: exit-head supervision, then optionally set-level stability calibration, then separately criterion/evidence alignment or counterfactual examples.

Use the free Kaggle T4 bridge only for a short, registered experiment: first a one-batch forward/backward/export smoke, then a capped pilot sized from measured step time. No paid training. Report GPU training separately from CPU inference. Preserve seed, checkpoint revision, code commit, data hashes, peak GPU memory and elapsed time.

**Exit gate:** on a held-out source and held-out question bundles, the candidate meets quality and calibration margins and saves end-to-end CPU time after all checks. If a neural adaptation fails once under a prespecified gate, record it and stop that arm.

### 6. Calibrate, quantize and red-team

Fit calibration only on its own partition. Test counterfactual evidence swaps/removals, question/rubric changes, negation, option order and distractors, state truncation, source shift, question count, a single hard question in an easy bundle, and malformed/unsupported schemas. Compare early decisions with both the full model and task labels.

Profile the full-precision process under 8 GiB, then separately export and profile the quantized build under 4 GiB. Measure answer, probability, latency and RSS changes.

**Final gate:** state only the named task, CPU, runtime, model revision and precision where paired quality and p95 latency pass. The full model's own correctness is not a formal guarantee of correctness for early exits.

### 7. Keep both reader levels connected

For every experiment, write a plain-English summary and a compact diagram that link to the model equations, source pins, configuration, result record and failed variants. Keep the original prompt untouched and preserve negative results. Update the Codex handoff when a gate or research hypothesis changes.

## Immediate next actions

1. Add Primus Decision, RSI-Jev, Nimble and RULERS to the direct prior-art and baseline matrix with their limits.
2. Finish a targeted overlap search for shared-state, question-specific early exits with bundle-level calibration; retain only exact primary sources.
3. Choose one small checkpoint and name a reproducible CPU after verifying local runtime and licensing; obtain the user's CPU model later before claiming device relevance.
4. Audit Typed Decisions and a controlled RuleTaker/ProofWriter composition split without opening locked test labels.
5. Register and run the no-training layerwise probe, including its bundle-level calibration and timing costs.
6. If the probe survives, run the direct matched CPU baselines, then one capped Kaggle pilot only if training materially improves the projected result.
7. Update this plan from the measured probe; stop if the prior art or exit headroom defeats the hypothesis.

The source pins are in [sources](sources.md), equations and controls in [architecture](architecture.md), and measured prior experiments in [research](research.md). Nothing here authorizes paid compute.
