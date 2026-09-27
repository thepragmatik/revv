# Registered CLINC lexical development baseline

**[Plain explanation](../../docs/start-here.md) · [First dataset audit](2026-09-27-clinc-audit.md) · [Architecture idea](../../docs/ideas.md) · [Evaluation contract](../../docs/evaluation.md)**

**Status: registered before execution, 27 September 2026.** CPU and public dataset only; no Kaggle GPU and no pretrained checkpoint. This is a labelled-data floor for later controlled model comparisons, not an eligible local model claim.

## Fixed input, split and method

- Pin the upstream CLINC source commit/blob from the [audit](2026-09-27-clinc-audit.md) and verify its hash. Use all official `train` and `oos_train` rows for corpus accounting, but fit the lexical in-scope centroids on the 15,000 known-intent training utterances. OOS training examples remain available for a later supervised head. **Never inspect `test` or `oos_test`.**
- Normalize utterances with Unicode NFKC, case folding and whitespace collapse. Remove all three official validation rows whose normalized text overlaps training; do not repair conflicting labels by hand. Split the remaining official `val + oos_val` **per label** by sorting SHA-256 of `revv-clinc-calibration-v1`, label and normalized text: first floor(n/2) to development, rest to calibration. This creates 50 OOS rows in each. Exact duplicated text cannot cross these folds; paraphrases and seed families still can.
- Fixed baseline: scikit-learn word unigram/bigram TF-IDF (`min_df=2`, `max_features=40,000`, sublinear term frequency); mean feature vector per in-scope label normalized to unit length; cosine similarity to 150 class centroids. A fixed `NONE` detector abstains when maximum known-label cosine is below the **5th percentile of known examples in calibration**. This threshold targets approximately 5% calibration in-scope deferral; no OOS development tuning. Majority intent (alphabetical tie break) is the floor.
- Record development closed-set known-intent accuracy; known accuracy/defer rate after the `NONE` threshold; OOS recall and Wilson 95% intervals; calibration threshold and split sizes. Preserve hashed ID, truth, predicted label, maximum score for each development example, without text. Scores are **not calibrated probabilities** and Brier/NLL are not reported. Do not call batch runtime a per-request latency result.
- One free GitHub Actions CPU job, capped at 15 minutes. Stop if source/schema/split assertions fail or there are no OOS rows. Next, hold this same cleaned development fold fixed when comparing pretrained shared and cross controls on Kaggle, with a separate calibration fold and small one-batch preflight.

## Result

Pending. Add the Actions run, aggregate JSON, raw hashed predictions, failure analysis and the next discriminating Kaggle experiment. Keep any failed attempts in the record.
