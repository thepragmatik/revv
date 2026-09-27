# Lexical evidence-retrieval floor on synthetic intervention states

**[Plain-English overview](../../docs/start-here.md) · [Gated research plan](../../docs/next-steps.md) · [Generator smoke](2026-09-27-intervention-mask-probe.md) · [Codex handoff](../../codex/RESUME.md)**

**Status: component diagnostic complete; no decision-model result.** This test asks whether a cheap lexical retriever can recover the full proof path for each question. It does not measure answer accuracy, verifier quality, or the proposed risk gate.

## Method

- Input: 256 locally generated states, with 32 rule-composition groups; 48 states in six groups were held out by composition.
- Retrieval fit: scikit-learn TF-IDF with word unigrams and bigrams, sublinear term frequency and L2 normalization. Vocabulary fit uses only `seen_composition` statements.
- Query: a declarative form generated from the structured target field, such as `Ari is blue.` This is a templated diagnostic, not a natural-language parsing test.
- For each entailed or refuted question, compare the retrieved statement IDs with one exact proof trace from the deterministic reasoner. `unknown` fields have no proof and are excluded from proof-recall denominators.
- Report `any-gold@k` (at least one statement in the recorded proof retrieved) and `all-gold@k` (every statement in that proof retrieved). One recorded proof path is evaluated; alternative proofs are not searched.
- The [JSON summary](2026-09-27-intervention-retrieval-floor.json) records the generator-input hash. Recreate the same input and scores with `python analysis/intervention_probe.py --states 256 --composition-group-size 8 --output-dir experiments/data/intervention-probe` followed by `python analysis/intervention_retrieval_probe.py --input experiments/data/intervention-probe/intervention-bundles.jsonl --output experiments/data/2026-09-27-intervention-retrieval-floor.json`.

## Held-out composition results

| Questions per state | Proof-supported questions | Any proof statement @1 | Any @5 | All proof statements @5 | All @10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 76.2% | 100.0% | 19.0% | 23.8% |
| 5 | 219 | 78.1% | 100.0% | 23.3% | 31.1% |
| 20 | 892 | 77.4% | 100.0% | 24.0% | 31.2% |

The Q=20 set therefore illustrates why `any-gold@k` alone is a weak gate: top-five retrieval always found some piece of the proof in this fixture, but retrieved the *entire recorded proof* only about one quarter of the time. This is not a general accuracy ceiling. It limits a verifier that needs every clause in this particular recorded proof; another proof path or a verifier that can infer from partial evidence could behave differently.

The seen-composition Q=20 control had `any-gold@5=100%` and `all-gold@5=23.5%`. Held-out groups had `100%` and `24.0%`, respectively. That similarity is descriptive only: there are six held-out rule groups, and the generator deliberately uses a small shared template vocabulary.

## Interpretation and stop conditions

This result motivates measuring complete evidence coverage and the verifier's actual dependence on each clause. It does **not** justify a more complex retriever: the task was generated from this project’s own short rules, and TF-IDF was fit on the same controlled vocabulary. It also says nothing about the text encoder, CPU latency, probability calibration, or whether a selective route beats direct scoring.

Next, compare the frozen pretrained sentence encoder against this lexical floor on the same generated records, then score direct-full-state and retrieved-evidence NLI controls. Keep the backbone frozen. If evidence coverage still caps the verifier or a confidence gate cannot identify fixable errors, stop before training and reconsider the mechanism.
