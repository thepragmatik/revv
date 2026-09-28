# Preregistered marked-evidence NLI stance diagnostic

**[Plain findings](../../docs/start-here.md#what-the-first-experiments-found) · [Failed frozen NLI decision control](2026-09-27-frozen-nli-contract.md) · [Evidence prototype](../data/2026-09-27-contractnli-evidence-prototypes.md) · [Red-team ledger](../../docs/red-team.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete; mixed result, no clean gate.** [Actions run 36295169108](https://github.com/thepragmatik/revv/actions/runs/36295169108) · [aggregate JSON](2026-09-27-nli-gold-stances.json). The frozen 82M NLI cross control failed to beat a cheap lexical ternary head despite 95.11% any-gold evidence retrieval. Gold evidence spans are label-supplied **oracle data** on public development and are **never** deployable input or a benchmark model score.

Read the same pinned author ContractNLI train/dev archive and the same pinned Apache-2.0 `cross-encoder/nli-MiniLM2-L6-H768` revision `c4d86af4493123990d7762712de9ed730c876161`; test remains unopened. Refit the exact pooled training-evidence TF-IDF centroid on all 423 train documents and verify top-five retrieval found a gold span in **584/614** positive dev decisions. For each of those **614 entailment/contradiction** development decisions, score (A) the five selected spans and (B) **all author-marked gold spans** against the same hypothesis with the **same frozen NLI checkpoint**, premise first, hypothesis second, 256-token pair limit, batch 32. No classifier training, aggregation tuning or dev threshold search. Conditional binary stance is **entailment if max span entailment softmax ≥ max span contradiction softmax**, else contradiction. Report accuracy, each true-class recall with Wilson 95% intervals, **balanced** accuracy, evidence counts, pair truncation, and gold-minus-retrieved paired accuracy with a 1,000-bootstrap interval by 61 documents. The entailment-only accuracy floor is `519/614 = 84.53%`, with balanced accuracy 50%; always show both because class imbalance can disguise failure.

One private Kaggle T4 kernel with internet, 300-second timeout and 240-second internal budget, collector ten minutes, Actions 20 minutes; no paid compute. Stop on source/model mapping/pair-count mismatch. No raw texts, IDs, predictions or weights in the repo; aggregate and hashes only. The frozen checkpoint and its weight acquisition worked in the previous run; this is a smaller, diagnostic reuse.

**Interpretation gate:** if the marked-evidence oracle balanced accuracy is at least 70% and contradiction recall at least 70%, but retrieved balanced accuracy is **10 points lower** with a paired document-bootstrap interval above zero, the shortlist/coverage deserves targeted work. If even gold balanced accuracy is under 70% or contradiction recall under 70%, prioritize task/domain-matched stance learning or a stronger cross model before redesigning the evidence router. Mixed outcomes remain unresolved. This does not evaluate `NotMentioned`, calibrated three-way probabilities, model deployment, or locked-test quality.

## Observed results

The successful private Kaggle kernel `rathworx/revv-nli-gold-stances/1` scored **614 positive dev decisions** from 61 documents: 519 entailments and 95 contradictions. The same checkpoint and binary aggregation were used in both arms. The pooled training-evidence retriever found at least one gold clause among five for **584/614** positive decisions; this does not guarantee every required clause is present.

| Positive-only diagnostic | Retrieved top five | Author-marked gold spans |
| --- | ---: | ---: |
| Conditional binary accuracy | 57.65% | 67.92% |
| Balanced accuracy | 68.93% | 76.29% |
| Entailment recall, n=519 | 52.60% | 64.16% |
| Contradiction recall, n=95 | 85.26% | 88.42% |
| Span–hypothesis pairs | 3,070 | 1,228 |
| Truncated pairs at 256 tokens | 6 | 3 |

Gold minus retrieved conditional accuracy was **+10.26 percentage points**, paired 95% document-cluster bootstrap **[+6.96, +13.65]**. The preregistered **balanced-accuracy** gap was only **7.36 points**, below the required ten, although gold balanced accuracy and contradiction recall passed their respective 70% thresholds. Thus the result is **mixed**; it does not select retrieval alone as the next investment. An entailment-only predictor would reach **84.53% raw accuracy** on these imbalanced positives (50% balanced accuracy), above both NLI arms' raw accuracy. Better clause selection matters; stance error and decision aggregation remain material.

T4 inference for both arms together took **19.38 seconds**, with **618,408,960 bytes peak PyTorch GPU allocation**; neither is local CPU latency or whole-process memory. Gold had 1–10 marked spans per positive decision, mean 2. The positive-only oracle excludes `NotMentioned` entirely, has no fitted calibration, and is unavailable at deployment. There was no locked-test access and no training. A cheaper next diagnostic is a train-only, document-disjoint stance/aggregation calibration with an entailment-majority control and a question-shuffle control; compare an independently sourced clause set before committing to neural architecture training. Preserve these prespecified gates when evaluating any successor.
