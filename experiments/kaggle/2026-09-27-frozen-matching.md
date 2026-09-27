# Preregistered frozen encoder evidence comparison

**[Plain explanation](../../docs/start-here.md) · [Lexical floor](../data/2026-09-27-clinc-lexical.md) · [CER-1 hypothesis](../../docs/ideas.md) · [Source access preflight](2026-09-27-internet-preflight.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete, 27 September 2026.** This inexpensive falsifier asked whether token evidence improves a pooled representation on a real labelled decision task. It tests one component of CER-1, not the whole architecture, trained quality, or local CPU speed.

## Frozen comparison and fixed budget

Use `sentence-transformers/all-MiniLM-L6-v2` at exact revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0), the pinned CLINC Git commit/blob and the **same** cleaned 1,547 development / 1,550 calibration folds as [the lexical floor](../data/2026-09-27-clinc-lexical.md). Never read CLINC test or `oos_test`. The pretrained encoder is frozen; its tokenizer, 150 candidate descriptions obtained by replacing underscores in class names with spaces, vector width and input truncation are identical in both methods. No labels from training, calibration, or development fit the encoder or candidate embeddings. Maximum state length is 64 model tokens, candidate length 16; report truncated examples.

- **Pooled:** attention-masked mean of last hidden states, normalize both vectors, cosine with all 150 label vectors.
- **Token MaxSim:** normalize non-special token vectors, for each candidate token take its maximum cosine over state tokens, then average the candidate-token maxima. Use *all* non-special tokens; no learned pruning. Different scoring only; the same encoder state and label vectors are cached for the batch.
- **Abstention:** choose the 5th percentile of the maximum in-scope scores on the calibration **known-only** rows separately for each method (`np.quantile(..., method="higher")`); defer if the development maximum is below the threshold. Do not optimize on OOS development labels. These scores are not probabilities.

Record known-intent closed accuracy, accuracy after deferral, known deferral, OOS recall on 50 OOS development rows and Wilson 95% intervals. Report the paired token-minus-pool closed-accuracy difference with a 1,000-resample **cluster bootstrap by the 150 true labels**; save per-example hashed IDs, truth, selected labels and maximum scores, without text. Record weights revision, dataset SHA-256, package versions, GPU identity, peak CUDA allocation, model acquisition time, full fold batch inference time and separate synchronized scoring kernel times with alternating order. These batch times include GPU synchronization overhead and are **not** local CPU p50/p95, process RSS or a statistically independent latency sample.

**Cost limit:** one private internet-enabled free Kaggle T4 kernel, timeout 420 seconds, internal 330-second inference budget, GitHub collector 14 minutes, Actions 25 minutes. Fail closed on missing GPU, pin/hash mismatch, unexpected fold size or model import. After a failure, read logs and change the smallest justified cause before resubmission; no repeated full training. The [CPU preflight](2026-09-27-internet-preflight.md) established source access and `transformers 5.0.0`, but did not import or load weights.

## Falsifier and decision after the run

Advance token evidence to a supervised comparison only if it has a positive paired benefit with meaningful uncertainty or a specific, reproducible OOS/related-label error reduction worth its extra measured compute. A broad cluster interval crossing zero or a slow MaxSim without a clear error benefit points to plain pooling for the first trained control. Favoring a method on these development examples does **not** earn a locked-test win; document selection bias. Both methods may lose to a fitted lexical baseline because natural-language label names are crude descriptors. Next train matched pooled and cross controls, test a multi-question evidence dataset, then measure an actual local CPU process under 8 GiB. Keep the 4 GiB quantized profile separate.

## Observations and adversarial reading

The [successful Actions run](https://github.com/thepragmatik/revv/actions/runs/36293473497) collected **one** private Kaggle `rathworx/revv-frozen-matching/1` T4 kernel. [Aggregate raw metrics](2026-09-27-frozen-matching.json) and [1,547 hashed predictions](2026-09-27-frozen-matching-predictions.jsonl) preserve its result. Pinned dataset hash, model revision, fold counts and GPU assertions passed. There were zero state inputs truncated at 64 tokens.

| Development measure | Pooled cosine | Token MaxSim |
| --- | ---: | ---: |
| Known-intent closed accuracy, n=1,497 | **70.94%** | 67.67% |
| Known accuracy after threshold; known deferred | 69.07%; 4.28% | 66.27%; 4.28% |
| OOS recall, n=50 | **64%** (Wilson 95%: 50.1–75.9%) | 50% (36.6–63.4%) |
| Synchronized GPU scoring per 32-example batch, median | 0.061 ms | 0.298 ms |

On paired known cases, MaxSim minus pooled closed accuracy is **−3.27 percentage points** (1,000-label-cluster bootstrap 95%: −6.54 to −0.20). The MaxSim scoring kernel took about **4.9×** the pooled kernel's median time under this shared-encoding T4 setup. Those submillisecond values exclude encoding and CPU serving costs, include synchronization, and cannot establish a local latency ratio. The full 3,097 development/calibration input batch inference took 1.00 second on this GPU; the peak PyTorch CUDA allocation was 123,460,608 bytes, excluding all CPU process memory and CUDA context/allocator overhead.

The [lexical development floor](../data/2026-09-27-clinc-lexical.md) attained 82.43% known closed accuracy and 36% OOS recall at a similarly calibrated known defer rate. Thus even pooled pretrained label matching trades *lower known accuracy* for *higher OOS recall* on these cases, not a quality win. A joined hashed-example check found 322 known utterances where lexical was correct and pooled failed, versus 148 in the other direction; 20 OOS examples rejected by pooled but missed by lexical versus six in the other direction. OOS has only 50 examples and wide intervals; do not claim an OOS population advantage yet. Both methods confuse related names: pooled `user_name → what_is_your_name` ten times and `calendar_update → calendar` ten times.

**Decision:** reject untrained all-token MaxSim with underscore-derived label text as the default component. The pretrained checkpoint was tuned for sentence embeddings, and label names need not describe the user phrasing: these facts are plausible explanations, *not proven causes*. Before training a heavier verifier, screen frozen **class prototypes fit on labelled training utterances** against the same label-text vectors, and report first-stage candidate recall at `k=1,2,5,10`, OOS under the fixed threshold, and embedding acquisition cost. If prototypes recover known accuracy cheaply, the bottleneck was likely candidate representation; if not, move to a small supervised control and evidence-bearing dataset. This result neither disproves all learned token interaction nor validates CER-1 on long shared states. The locked test remains sealed.
