# Preregistered CLINC train–development lexical similarity sensitivity

**[Plain explanation](../../docs/start-here.md) · [Class-prototype result](../kaggle/2026-09-27-prototype-screen.md) · [First dataset audit](2026-09-27-clinc-audit.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: complete, 27 September 2026.** The frozen class-prototype pilot beat the fitted lexical floor on CLINC public development. Its 100 train examples per intent and similar utterance templates could make this advantage brittle. This screen quantifies proximity to *any* training text without inspecting locked test or changing either candidate.

Reuse the pinned CLINC source and exact cleaned development fold. Fit character 3–5-gram TF-IDF on **15,100 official train plus OOS train utterances** (`min_df=2`, at most 50,000 features, sublinear TF), then record each of 1,547 development utterances' maximum cosine to a training utterance. Check the ordered hashed prediction artifacts and true labels match the source before joining. Fixed strata: similarity `<0.8`, `[0.8,0.95)`, and `≥0.95`; also report global median, p90 and p95 nearest similarity. For each stratum report row and known/OOS counts, nearest training label agreement on known cases, known accuracy after deferral and OOS recall for the already committed lexical and prototype predictions. No original utterances, nearest text or document IDs in output. No model fitting beyond this unsupervised similarity index, no threshold optimization. One free CPU Actions run capped at 15 minutes.

**Decision:** if the prototype advantage is concentrated in the `≥0.95` stratum and disappears below 0.8, hold back a generalization interpretation and prioritize independent-source or template-family split. If it persists below 0.8, still test a separate domain: low character similarity does not prove semantic independence. Stratum sizes and uncertainty matter, especially with only 50 total OOS examples. This is a sensitivity check on a public development split, not a new external validation set or locked test.

## Observed sensitivity

The [CPU Actions run](https://github.com/thepragmatik/revv/actions/runs/36294353039) reproduced all 1,547 hashed prediction IDs and source labels; its [aggregate report](2026-09-27-clinc-similarity-sensitivity.json) contains no original utterances. Median nearest-training character 3–5-gram cosine was **0.568**; p90 **0.799** and p95 **0.850**.

| Nearest training similarity | Development cases | Known cases | Known after deferral: lexical | Known after deferral: prototype |
| --- | ---: | ---: | ---: | ---: |
| `<0.8` | 1,394 | 1,344 | 79.32% | **87.80%** |
| `[0.8,0.95)` | 145 | 145 | 93.79% | 93.10% |
| `≥0.95` | 8 | 8 | 75.00% | 87.50% |

All **50 OOS cases** are in the `<0.8` stratum, so there is no high-similarity OOS comparison. The known prototype gain persists on low character-similarity examples and is not explained solely by near-copy training utterances. Only eight cases populate the highest stratum, too few for a stable claim. Different wording can preserve the same template or semantics; nearest character TF-IDF is a limited leakage screen, and no source-shift, paraphrase-family holdout or locked test was evaluated.
