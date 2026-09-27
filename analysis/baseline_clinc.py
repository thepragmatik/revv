"""Registered lexical CLINC baseline using train and cleaned validation only."""

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from audit_clinc import COMMIT, fetch_pinned, normal


SEED = "revv-clinc-calibration-v1"
METRICS = Path("results/clinc-lexical.json")
PREDICTIONS = Path("results/clinc-lexical-predictions.jsonl")


def fold_key(label, utterance):
    data = f"{SEED}\0{label}\0{normal(utterance)}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def wilson(hits, n):
    if n == 0:
        return None
    z = statistics.NormalDist().inv_cdf(0.975)
    p = hits / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [mid - half, mid + half]


def main():
    data, source_sha256 = fetch_pinned("data_full.json")
    train = data["train"] + data["oos_train"]
    held = data["val"] + data["oos_val"]
    train_normalized = {normal(text) for text, _ in train}
    excluded = [(text, label) for text, label in held if normal(text) in train_normalized]
    clean = [(text, label) for text, label in held if normal(text) not in train_normalized]
    if len(excluded) != 3 or len(clean) != 3097:
        raise ValueError("Cleaned CLINC validation differs from registered overlap audit")

    # Stratify by label with fixed SHA-256 order; exact duplicates cannot cross folds.
    groups = defaultdict(list)
    for row in clean:
        groups[row[1]].append(row)
    development, calibration = [], []
    for label in sorted(groups):
        ordered = sorted(groups[label], key=lambda row: fold_key(label, row[0]))
        midpoint = len(ordered) // 2
        development.extend(ordered[:midpoint])
        calibration.extend(ordered[midpoint:])
    development.sort(key=lambda row: fold_key(row[1], row[0]))
    calibration.sort(key=lambda row: fold_key(row[1], row[0]))
    assert {normal(x[0]) for x in development}.isdisjoint({normal(x[0]) for x in calibration})

    labels = sorted({label for _, label in train if label != "oos"})
    if len(labels) != 150:
        raise ValueError("Expected exactly 150 in-scope intent labels")
    label_index = {label: i for i, label in enumerate(labels)}
    train_known = [(text, label) for text, label in train if label != "oos"]
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(text for text, _ in train_known)
    y = [label_index[label] for _, label in train_known]
    indicator = csr_matrix((np.ones(len(y)), (np.arange(len(y)), y)), shape=(len(y), len(labels)))
    centroids = normalize(indicator.T @ x_train)
    fit_seconds = time.perf_counter() - started

    def score(rows):
        return (vectorizer.transform(text for text, _ in rows) @ centroids.T).toarray()

    cal_scores = score(calibration)
    cal_known = np.array([label != "oos" for _, label in calibration])
    # Fixed 5% in-scope calibration false-positive target, not an OOS-tuned search.
    threshold = float(np.quantile(cal_scores[cal_known].max(axis=1), 0.05, method="higher"))
    started = time.perf_counter()
    dev_scores = score(development)
    score_seconds = time.perf_counter() - started
    top = dev_scores.argmax(axis=1)
    confidence = dev_scores.max(axis=1)
    true = [label for _, label in development]
    predicted = ["oos" if float(confidence[i]) < threshold else labels[int(top[i])] for i in range(len(true))]
    known_indices = [i for i, label in enumerate(true) if label != "oos"]
    oos_indices = [i for i, label in enumerate(true) if label == "oos"]
    closed_hits = sum(labels[int(top[i])] == true[i] for i in known_indices)
    known_hits = sum(predicted[i] == true[i] for i in known_indices)
    oos_hits = sum(predicted[i] == "oos" for i in oos_indices)
    known_deferred = sum(predicted[i] == "oos" for i in known_indices)
    majority_label = sorted(Counter(label for _, label in train_known).items(), key=lambda x: (-x[1], x[0]))[0][0]
    predictions = [
        {
            "example_sha256": hashlib.sha256(normal(text).encode("utf-8")).hexdigest(),
            "gold": label,
            "predicted": predicted[i],
            "top_known": labels[int(top[i])],
            "max_known_cosine": float(confidence[i]),
        }
        for i, (text, label) in enumerate(development)
    ]
    report = {
        "kind": "clinc_lexical_cpu_development_baseline_not_model_win",
        "source_commit": COMMIT,
        "source_sha256": source_sha256,
        "seed": SEED,
        "split_rule": "exclude exact normalized train/val overlaps; per-label SHA-256 order, first floor(n/2) development, remainder calibration",
        "locked_test_examples_examined": 0,
        "quarantined_val_examples": len(excluded),
        "development_examples": len(development),
        "calibration_examples": len(calibration),
        "development_oos": len(oos_indices),
        "calibration_oos": int(sum(not x for x in cal_known)),
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 40000, "sublinear_tf": True, "features_fit": int(x_train.shape[1])},
        "majority_label": majority_label,
        "majority_accuracy_all_dev": sum(x == majority_label for x in true) / len(true),
        "calibration_known_fpr_target": 0.05,
        "calibration_threshold_max_known_cosine": threshold,
        "development_known_closed_accuracy": closed_hits / len(known_indices),
        "development_known_closed_wilson95": wilson(closed_hits, len(known_indices)),
        "development_known_accuracy_with_defer": known_hits / len(known_indices),
        "development_known_defer_rate": known_deferred / len(known_indices),
        "development_oos_recall": oos_hits / len(oos_indices),
        "development_oos_wilson95": wilson(oos_hits, len(oos_indices)),
        "fit_seconds": fit_seconds,
        "development_batch_score_seconds": score_seconds,
        "python_and_libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "limitations": "Public benchmark development only; no paraphrase grouping, calibrated probabilities, CPU process RSS, per-request latency, or locked test evaluation. OOS threshold uses calibration known-score quantile."
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    PREDICTIONS.write_text("\n".join(json.dumps(p, sort_keys=True) for p in predictions) + "\n", encoding="utf-8")
    print(f"CLINC lexical pilot: {len(development)} development, {len(calibration)} calibration; in-scope accuracy {report['development_known_closed_accuracy']:.3f}, OOS recall {report['development_oos_recall']:.3f}.")


if __name__ == "__main__":
    main()
