# Preregistered evidence-removal and cross-document shuffle controls

**[Plain findings](../../docs/start-here.md) · [Original matched head](2026-09-27-contractnli-lexical-heads.md) · [Research hypothesis](../../docs/ideas.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete; preregistered evidence-use gate failed.** ContractNLI has 17 fixed hypotheses and a hypothesis-only prior that already achieves 68.08% development accuracy. The pooled-evidence lexical head achieves 70.20%, with only an uncertain 1.35-point gain over query-word evidence. This negative control tests whether the pooled head relies on matched *contract evidence* to improve its decisions. It cannot establish semantic grounding or novel model quality.

Reuse the pinned author train/dev ZIP hash, 423/61 documents, the prior script's 17 hypothesis ordering, all document spans, five SHA-256 document folds, TF-IDF vocabulary, top-five training-evidence retrieval and shared multinomial logistic regression hyperparameters (`C=1`, `lbfgs`, max 250 iterations). Reproduce pooled dev accuracy **0.7020250723240116** and macro F1 **0.6587908259170189** exactly or fail. No development label fitting; test JSON remains unopened.

Compare three arms over **1,037 three-way dev decisions**, with identical hypothesis text and one-hot question identifier:

1. **Pooled evidence:** the already registered trained head with its own selected dev clauses.
2. **Hypothesis only:** refit the same head on the same training rows with evidence text omitted; use the common original training-only word vocabulary and question one-hot. This is a stronger learned fixed-question control than the frequency prior.
3. **Shuffled evidence:** keep the original pooled head without refitting, but replace each selected dev evidence list with the list for the **same hypothesis in the next dev document in fixed order**, wrapping at 61; do not use those documents' labels. This preserves question identity and makes most clause lists mismatched. It is a distribution shift and may accidentally supply same-label evidence.

Report accuracy, macro F1, class recall, raw Brier/NLL, and the two paired pooled-minus-control accuracy differences with 1,000-bootstrap 95% intervals resampled by 61 documents. No raw contract text, document IDs or row predictions in the artifact. One free CPU Actions run, max 15 minutes. No GPU spending.

**Interpretation gate:** before prioritizing more evidence architecture, require pooled-minus-hypothesis-only **at least +3 percentage points in accuracy and macro F1, with paired accuracy interval above zero**, plus pooled-minus-shuffled **at least +5 percentage points in accuracy with paired interval above zero**, without lower contradiction or not-mentioned recall than the hypothesis-only head. If only one criterion passes, the evidence-use claim is unresolved; if neither passes, prioritize fixed-hypothesis shortcuts or stance learning. A strong shuffle drop merely shows the classifier is sensitive to the evidence field, not that its decisions are correct because of evidence. Keep the preregistered gates even if the result disappoints.

**Run log:** [first CPU run 36295484774](https://github.com/thepragmatik/revv/actions/runs/36295484774) stopped before writing results because its strict pooled reproduction assertion failed; its traceback contained no observed accuracy. A diagnostic print was added to the same assertion, with no gate, task, split, feature set or hyperparameter changed. [Second run 36301269916](https://github.com/thepragmatik/revv/actions/runs/36301269916) passed both exact reproduction assertions and saved the [aggregate JSON](2026-09-27-contractnli-evidence-shuffle.json). The first mismatch is unexplained, so reproducibility remains a limitation; do not delete or retroactively call that run successful.

## Measured result

| Three-way public development, 1,037 decisions | Hypothesis only | Pooled matched evidence | Different document's evidence |
| --- | ---: | ---: | ---: |
| Accuracy | 68.08% | **70.20%** | 66.35% |
| Macro F1 | 63.25% | **65.88%** | 62.13% |
| Contradiction recall | 58.95% | **61.05%** | 57.89% |
| Not-mentioned recall | **58.63%** | 57.68% | 52.96% |
| Raw classifier Brier sum | 0.4355 | **0.3865** | 0.4586 |

Matched pooled minus learned hypothesis-only accuracy **+2.12 points**, document-cluster bootstrap 95% **[0.00, +4.34]**; macro F1 **+2.63 points**, but not-mentioned recall **−0.95 points**. Matched pooled minus swapped-evidence accuracy **+3.86 points**, document interval **[+0.77, +7.04]**. The first size and interval gate failed; the second size gate also failed despite a positive interval. Changing the clause field influences this model's outputs, but its modest gain above a learned fixed-question control does not establish accurate use of evidence. Cross-document swaps can accidentally share the true label and shift text distribution; there is no independent domain or locked-test result. **Keep neural CER-2 training paused.**
