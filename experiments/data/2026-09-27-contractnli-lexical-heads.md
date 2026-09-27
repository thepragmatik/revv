# Preregistered ContractNLI matched evidence and ternary decision heads

**[Plain explanation](../../docs/start-here.md) · [Train-evidence retrieval](2026-09-27-contractnli-evidence-prototypes.md) · [Prior-only floor](2026-09-27-contractnli-controls.md) · [CER-2 hypothesis](../../docs/ideas.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete, 27 September 2026.** A pooled training-evidence centroid retrieved at least one gold span in the top five for 95.11% of positive ContractNLI development decisions. Retrieval is not a ternary answer. This screen uses a matched lightweight decision head to see whether better retrieval improves end-to-end entailment/contradiction/not-mentioned classification, before training a GPU encoder.

Keep the pinned author train/dev archive, 17 questions and *all* author document spans. Fixed two retrieval arms: (A) question-text TF-IDF; (B) the pooled **training-evidence** TF-IDF centroid per question. Fit the lexical retrieval vocabulary on training spans plus 17 descriptions (word 1–2-gram, min_df=2, max 30,000, sublinear TF). For B's **training** examples, construct five document folds from SHA-256 of document ID, and obtain the centroid from the other four folds; this keeps each training document's gold spans out of its own retriever. Development retrieval uses all training evidence. Always select top five with stable position tie breaks. Concatenate the selected span text with the full fixed question text. Train the **same** shared word 1–2-gram TF-IDF vocabulary on both arms' training concatenations; append question one-hot, then separately fit matched multinomial logistic regressions (`C=1`, `lbfgs`, max 250 iterations). Fail on non-convergence. No development labels fit vocabulary, centroid, or head.

Report 1,037 development decisions' accuracy, macro F1, class recall, Brier/NLL as **uncalibrated raw classifier probabilities**, paired B-minus-A accuracy and a 1,000-bootstrap 95% interval resampling by 61 documents. Compare to the already measured per-hypothesis 68.08% prior accuracy and 63.25% macro F1. Distinguish retrieval coverage from ternary decision quality. One free CPU Actions run capped at 15 minutes; no raw contract text, document IDs or test JSON in outputs.

**Decision gate:** a centroid retrieval arm merits a neural shared-encoder pilot only if it beats direct-question retrieval in paired accuracy or macro F1, loses no more than **two percentage points** of contradiction or not-mentioned recall, and clears the prior-only floor in both accuracy and macro F1. A paired interval overlapping zero weakens a small apparent gain. If both heads are at/below the prior, better evidence retrieval alone does not justify GPU training; inspect stance/negation and stronger same-evidence cross controls first. Public development, fixed hypotheses and source boilerplate limit any win. Later compare matched cross and shared models on full-document coverage, local CPU RSS below 8 GiB and an explicit compact quantized profile; this CPU lexical result cannot establish those claims.

## Observed and next decision

The [CPU Actions run](https://github.com/thepragmatik/revv/actions/runs/36294517521) saved [aggregate raw metrics](2026-09-27-contractnli-lexical-heads.json). Five document-disjoint training folds were used for the centroid arm; both classifiers converged within the predeclared limit. The common TF-IDF head vocabulary fitted 30,000 features. No locked test or original PDF was opened.

| Development, 61 documents × 17 hypotheses | Per-hypothesis prior | Query-word top-five + lexical head | Training-evidence top-five + same head |
| --- | ---: | ---: | ---: |
| Three-way accuracy | 68.08% | 68.85% | **70.20%** |
| Macro F1 | 63.25% | 63.60% | **65.88%** |
| Contradiction recall | 58.95% | 57.89% | **61.05%** |
| Not-mentioned recall | 58.63% | 54.14% | **57.68%** |
| Raw classifier multiclass Brier | 0.4357 | 0.4017 | **0.3865** |

The paired training-evidence minus query accuracy is **+1.35 points**, but its 1,000-document-bootstrap 95% interval is **−0.87 to +3.76**; this is an uncertain difference. The centroid arm narrowly passes the *nominal* preregistered accuracy/macro-F1/recall gate, yet only modestly exceeds the no-reading prior. Brier/NLL use raw logistic probabilities without separate calibration, so they do not show deployable probability quality. The gap between **95.11% any-evidence retrieval** and **70.20% ternary accuracy** points to stance, multi-clause interpretation, or shortcuts as competing explanations; this experiment cannot isolate them. Next use a frozen pretrained NLI cross scorer on the **same selected spans** as a stronger answer-quality control, with training-only aggregation/calibration and a bounded free GPU run. Only then decide whether shared-encoder training is warranted. A 17-question template-specific classifier is not a general decision model.
