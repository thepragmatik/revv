# Preregistered ContractNLI priors and full-document evidence screen

**[Plain explanation](../../docs/start-here.md) · [Author dataset audit](2026-09-27-contractnli-audit.md) · [CER-1 hypothesis](../../docs/ideas.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete, 27 September 2026.** The author audit found that 59/61 development contracts exceed 512 words. This screen quantifies the 17-hypothesis task's majority/prior floor and how much evidence a cheap lexical span selector can find when it reads the **entire document**.

Read the same pinned author archive SHA-256 `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b` (70 MB cap); open only bounded `train.json` and `dev.json`. Fit **a separate three-way categorical Laplace-smoothed prior per hypothesis** from 423 train documents; predict its majority label for 61 dev documents × 17 hypotheses. Report global-majority and per-hypothesis accuracy, macro F1, per-class recall, multiclass Brier (sum across three labels) and NLL. Bootstrap accuracy by **document**, 1,000 resamples, because 17 decisions share each contract. The priors are probabilities derived from training frequency, without fitting or calibrating on development.

For evidence, split each document at the author-provided span offsets, fit word unigram/bigram TF-IDF on **training spans plus the fixed 17 hypothesis descriptions** (`min_df=2`, at most 30,000 features, sublinear term frequency), and rank *every span in each full dev document* by cosine to each fixed hypothesis. On the 614 dev entailed/contradicted pairs with gold spans, report at least one-gold and all-gold recall at `k=1,3,5,10`, overall and by class. Stable score ordering breaks ties by span position. No labels from dev tune vocabulary, scoring, `k`, or priors. `NotMentioned` has no evidence and is not included in evidence recall; it remains in three-way classification. Save aggregates only, no text, URLs, document IDs or train/dev predictions. No test JSON or original PDFs opened. One free Actions CPU run capped at 15 minutes.

**Decision:** if small `k` misses many contradiction/exception spans, a verifier over top `k` cannot answer reliably even if its classifier is perfect; expand retrieval coverage before training. If top-`k` recall is high, next compare a *same-evidence* trained cross scorer with a shared encoder on 17 questions, report NLI macro F1, `NotMentioned` recall, evidence coverage and actual CPU RSS/latency. Prior-only classification is a necessary floor, not an eligible existing-model comparator. No legal decision or production claim follows from this benchmark.

## Observed and decision

The [CPU Actions run](https://github.com/thepragmatik/revv/actions/runs/36294073824) completed on the pinned author archive; [raw aggregate JSON](2026-09-27-contractnli-controls.json) contains the full breakdown. The per-hypothesis priors predict without document text: **68.08%** accuracy across 1,037 dev decisions (1,000 document-cluster bootstrap 95%: 64.51–71.36%), macro F1 **63.25%**, `NotMentioned` recall **58.63%**, multiclass Brier **0.4357** and NLL **0.7026**. The global majority `Entailment` gets 50.05%. These priors are supervised on the 423 training documents and have no independent calibration fold.

The TF-IDF vocabulary fitted 30,000 features from 32,895 training spans and fixed hypotheses, then examined **all 5,102 dev spans**. Recall below is conditional on 614 positive entailment or contradiction decisions, with a mean of two marked spans each. `NotMentioned` has no gold evidence and is excluded here.

| Marked evidence retrieved | At least one @1 | At least one @5 | At least one @10 | All marked @5 |
| --- | ---: | ---: | ---: | ---: |
| All positive, n=614 | 30.78% | 66.12% | 81.43% | 31.60% |
| Entailment, n=519 | 34.68% | 69.17% | 83.24% | 33.14% |
| Contradiction, n=95 | **9.47%** | **49.47%** | 71.58% | **23.16%** |

**Decision:** reject a five-span lexical-only shortlist as the evidence source for a trained verifier, especially for contradictions. Retrieving *one* gold span is a permissive measure: when multiple spans are needed, all-gold recall at five is only 31.60%. A model may still infer the correct label from unmarked text; these are retrieval ceilings **only for an evidence-grounded route restricted to these gold spans**, not formal upper bounds on every possible classifier. Prior [ContractNLI work](https://aclanthology.org/2022.findings-emnlp.28/) already compares lexical span retrieval with NLI-based ranking, so lexical retrieval is a control, not a new method. The next cheap screen fits separate support/refute evidence prototypes on train spans and compares them with a pooled evidence prototype and the fixed-hypothesis query. Move to GPU evidence learning only if this low-cost screen shows a material remaining gap.
