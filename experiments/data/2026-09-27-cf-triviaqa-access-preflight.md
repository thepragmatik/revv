# CF-TriviaQA access and fit preflight

**As of:** 27 September 2026  
**Outcome:** Public archive and repository license found; upstream content rights and original/edit pairing remain unresolved.

## Plain-English result

Google Research’s archived CF-TriviaQA repository contains a counterfactual open-book QA set. Its README says it has 16,853 examples and labels the dataset Apache 2.0. The documented row contains a TriviaQA question, a generated counterfactual paragraph and answer, plus a question ID ([repository](https://github.com/google-research-datasets/cf_triviaqa)).

The official TriviaQA page says the University of Washington does not own copyright in the questions and documents. So the derivative repository’s Apache 2.0 statement does not by itself tell us the reuse terms for every upstream question and source document. The documented CF-TriviaQA row also omits the original passage and the exact edit operation; a join to TriviaQA might help reconstruct a pair, but that join and its provenance have not been checked. ([TriviaQA source page](https://nlp.cs.washington.edu/triviaqa/))

No data was downloaded or used in this preflight.

## Fit to our task

CF-TriviaQA could provide a secondary test of whether a model follows counterfactual evidence instead of its memorized answer. The repository reports that its counterfactual documents conflict with original TriviaQA answers and that their generated answers are grounded in those documents.

It does not directly provide the paired original-context / edited-context records needed to measure which cached typed answers should be refreshed. Its task is open-book answer generation, not a bundle of typed decisions. If source rights and pair reconstruction are later resolved, it may be useful as a context-grounding stress test. It is not the primary refresh-selection benchmark.

## Access and provenance checks

- The repository is a public archive; GitHub marks it read-only since 3 November 2025.
- Its README reports 16,853 examples, the listed question/paragraph/answer/ID schema, and an Apache 2.0 dataset license.
- The original TriviaQA data page offers separate downloads and explicitly states that UW does not own the copyright of included questions and documents.
- The cited HAR paper describes the counterfactual open-book QA task ([paper](https://arxiv.org/abs/2311.07424)).
- We did not fetch the JSONL, clone the repository, join it to TriviaQA, or inspect or redistribute any records.

## Decision and next step

**Do not use CF-TriviaQA as the primary benchmark yet.** First resolve the rights and provenance of the upstream questions/documents and verify that the published release can be paired with original records under those terms. If those checks pass, evaluate it only as a secondary context-grounding transfer test.

Continue CPU-side work on an independent, rights-clear workload. The frozen synthetic screen remains a diagnostic only, and the next selective-refresh target remains expected gold-loss reduction. Keep GPU training paused until the workload, local CPU baselines, and free-provider availability are established.
