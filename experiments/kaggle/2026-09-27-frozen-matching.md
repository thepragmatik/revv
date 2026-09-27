# Preregistered frozen encoder evidence comparison

**[Plain explanation](../../docs/start-here.md) · [Lexical floor](../data/2026-09-27-clinc-lexical.md) · [CER-1 hypothesis](../../docs/ideas.md) · [Source access preflight](2026-09-27-internet-preflight.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: registered; one free T4 run pending.** This inexpensive falsifier asks whether token evidence improves a pooled representation on a real labelled decision task. It tests one component of CER-1, not the whole architecture, trained quality, or local CPU speed.

## Frozen comparison and fixed budget

Use `sentence-transformers/all-MiniLM-L6-v2` at exact revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0), the pinned CLINC Git commit/blob and the **same** cleaned 1,547 development / 1,550 calibration folds as [the lexical floor](../data/2026-09-27-clinc-lexical.md). Never read CLINC test or `oos_test`. The pretrained encoder is frozen; its tokenizer, 150 candidate descriptions obtained by replacing underscores in class names with spaces, vector width and input truncation are identical in both methods. No labels from training, calibration, or development fit the encoder or candidate embeddings. Maximum state length is 64 model tokens, candidate length 16; report truncated examples.

- **Pooled:** attention-masked mean of last hidden states, normalize both vectors, cosine with all 150 label vectors.
- **Token MaxSim:** normalize non-special token vectors, for each candidate token take its maximum cosine over state tokens, then average the candidate-token maxima. Use *all* non-special tokens; no learned pruning. Different scoring only; the same encoder state and label vectors are cached for the batch.
- **Abstention:** choose the 5th percentile of the maximum in-scope scores on the calibration **known-only** rows separately for each method (`np.quantile(..., method="higher")`); defer if the development maximum is below the threshold. Do not optimize on OOS development labels. These scores are not probabilities.

Record known-intent closed accuracy, accuracy after deferral, known deferral, OOS recall on 50 OOS development rows and Wilson 95% intervals. Report the paired token-minus-pool closed-accuracy difference with a 1,000-resample **cluster bootstrap by the 150 true labels**; save per-example hashed IDs, truth, selected labels and maximum scores, without text. Record weights revision, dataset SHA-256, package versions, GPU identity, peak CUDA allocation, model acquisition time, full fold batch inference time and separate synchronized scoring kernel times with alternating order. These batch times include GPU synchronization overhead and are **not** local CPU p50/p95, process RSS or a statistically independent latency sample.

**Cost limit:** one private internet-enabled free Kaggle T4 kernel, timeout 420 seconds, internal 330-second inference budget, GitHub collector 14 minutes, Actions 25 minutes. Fail closed on missing GPU, pin/hash mismatch, unexpected fold size or model import. After a failure, read logs and change the smallest justified cause before resubmission; no repeated full training. The [CPU preflight](2026-09-27-internet-preflight.md) established source access and `transformers 5.0.0`, but did not import or load weights.

## Falsifier and decision after the run

Advance token evidence to a supervised comparison only if it has a positive paired benefit with meaningful uncertainty or a specific, reproducible OOS/related-label error reduction worth its extra measured compute. A broad cluster interval crossing zero or a slow MaxSim without a clear error benefit points to plain pooling for the first trained control. Favoring a method on these development examples does **not** earn a locked-test win; document selection bias. Both methods may lose to a fitted lexical baseline because natural-language label names are crude descriptors. Next train matched pooled and cross controls, test a multi-question evidence dataset, then measure an actual local CPU process under 8 GiB. Keep the 4 GiB quantized profile separate.
