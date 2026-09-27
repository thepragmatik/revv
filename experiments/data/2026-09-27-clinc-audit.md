# Registered public-data audit: CLINC intent and unknown intent

**[Plain explanation](../../docs/start-here.md) · [Evaluation contract](../../docs/evaluation.md) · [Codex handoff](../../codex/RESUME.md)**

**Registered 27 September 2026. Status: complete.** This is a free CPU integrity and rights check, not a Kaggle GPU training run or a model benchmark.

The [CLINC authors' dataset](https://github.com/clinc/oos-eval) has 150 in-scope intents and separate out-of-scope utterances. Its upstream [license](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE) is Creative Commons Attribution 3.0. This makes it a useful first test of supplied option labels and the explicit `NONE` answer, but single-utterance intent classification cannot by itself prove success on policy decisions, multiple questions per state or evidence grounding.

## Registered checks and stop rule

- Pin source commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`, `data/data_full.json` Git blob `7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa`, and `data/domains.json` Git blob `60a74358e52060e60b6128ae4efe679e9d6ba69c`. Reject a hash mismatch or unexpected schema. Read only public train/validation rows for this audit; do not inspect locked test examples.
- Report source SHA-256 values, train/validation counts, out-of-scope counts, mapping coverage by domain and in-scope label balance. Count normalized exact duplicate utterances and label conflicts within and across train/validation. No raw utterances, labels per example or secret values go into the repo.
- CPU GitHub Actions run capped at ten minutes. The audit fetch is capped at 4 MB per file; no Kaggle quota or paid compute. Its purpose is to decide how to group and split before fitting a baseline, not to choose a winning model.
- Stop and investigate if hash/schema/domain mapping checks fail or duplicates cross the intended split. Exact duplicate checking is only a lower bound on paraphrase/template leakage; record that limitation even if the count is zero.

## Result

The [first audit attempt](https://github.com/thepragmatik/revv/actions/runs/36292009816) verified both source hashes but its code counted only `train` and `val`, missing the source's separate `oos_train` and `oos_val` fields. This was an **invalid answerability audit**. The corrected [second Actions run](https://github.com/thepragmatik/revv/actions/runs/36292063048) includes both out-of-scope fields. Its [aggregate JSON](2026-09-27-clinc-audit.json) records source SHA-256 values, field names and duplicate counts, without publishing examples.

| Audit finding | Observed |
| --- | ---: |
| Train including out-of-scope | 15,100 utterances; 100 out-of-scope |
| Validation including out-of-scope | 3,100 utterances; 100 out-of-scope |
| In-scope mapping | 150 intents in 10 domains, 15 intents per domain |
| In-scope examples per intent | 100 train; 20 validation |
| Duplicate normalized text *within* each split | 0 train; 0 validation |
| Duplicate normalized text *across* splits | 3 unique utterances; 2 have conflicting label sets |
| Locked test examples examined | 0 |

**Decision:** quarantine the three overlapping validation utterances from model selection and calibration; do not silently relabel the two conflicts. Reproduce that exclusion deterministically with normalized-text hashes in the future split manifest. Check paraphrase or seed-template families before claiming independent new-source generalization: this audit detects exact normalized overlap only, and the public data does not expose complete grouping IDs. The 100 validation out-of-scope examples also limit calibration precision, so report uncertainty. A single-intent dataset cannot validate multi-question state reuse or evidence-grounded policy judgements. No weights trained and no locked test read. The next registered task is a small real-data lexical/majority baseline followed by matched pretrained cross/shared controls, without choosing an architecture from T4 synthetic timings alone.
