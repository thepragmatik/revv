"""Fixed lexical similarity strata for existing CLINC development predictions."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from audit_clinc import fetch_pinned, normal
from baseline_clinc import fold_key


OUT = Path("results/clinc-similarity-sensitivity.json")
LEXICAL = Path("experiments/data/2026-09-27-clinc-lexical-predictions.jsonl")
PROTOTYPE = Path("experiments/kaggle/2026-09-27-prototype-screen-predictions.jsonl")


def bins(scores):
    return {
        "low_lt_0.8": [i for i, x in enumerate(scores) if x < 0.8],
        "middle_0.8_to_lt_0.95": [i for i, x in enumerate(scores) if 0.8 <= x < 0.95],
        "high_ge_0.95": [i for i, x in enumerate(scores) if x >= 0.95],
        "all": list(range(len(scores))),
    }


def main():
    data, source_sha = fetch_pinned("data_full.json")
    if source_sha != "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0":
        raise ValueError("Pinned CLINC source changed")
    train = data["train"] + data["oos_train"]
    train_normal = {normal(text) for text, _ in train}
    clean = [(text, label) for text, label in data["val"] + data["oos_val"] if normal(text) not in train_normal]
    grouped = defaultdict(list)
    for row in clean:
        grouped[row[1]].append(row)
    dev = []
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda row: fold_key(label, row[0]))
        dev.extend(ordered[:len(ordered)//2])
    dev.sort(key=lambda row: fold_key(row[1], row[0]))
    if len(dev) != 1547 or len(train) != 15100:
        raise ValueError("Dataset fold differs from preregistered predictions")
    lexical = [json.loads(row) for row in LEXICAL.read_text().splitlines()]
    prototype = [json.loads(row) for row in PROTOTYPE.read_text().splitlines()]
    for i, (text, label) in enumerate(dev):
        checksum = hashlib.sha256(normal(text).encode()).hexdigest()
        if lexical[i]["example_sha256"] != checksum or prototype[i]["example_sha256"] != checksum:
            raise ValueError("Prediction artifact does not match exact development ordering")
        if lexical[i]["gold"] != label or prototype[i]["gold"] != label:
            raise ValueError("Prediction truth differs from pinned source")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)
    xtrain = vectorizer.fit_transform(text for text, _ in train)
    xdev = vectorizer.transform(text for text, _ in dev)
    similarities = xdev @ xtrain.T
    nearest, train_label = [], []
    for i in range(len(dev)):
        row = similarities.getrow(i)
        if row.nnz:
            loc = int(row.data.argmax())
            nearest.append(float(row.data[loc]))
            train_label.append(train[int(row.indices[loc])][1])
        else:
            nearest.append(0.0)
            train_label.append(None)
    def per_bin(indices):
        known = [i for i in indices if dev[i][1] != "oos"]
        oos = [i for i in indices if dev[i][1] == "oos"]
        def rate(idx, model):
            return sum(model[i]["predicted" if model is lexical else "training_prototype"] == dev[i][1] for i in idx) / len(idx) if idx else None
        return {
            "rows": len(indices), "known": len(known), "oos": len(oos),
            "nearest_train_label_matches_known": sum(train_label[i] == dev[i][1] for i in known) / len(known) if known else None,
            "known_after_defer_lexical": rate(known, lexical),
            "known_after_defer_prototype": rate(known, prototype),
            "oos_recall_lexical": rate(oos, lexical),
            "oos_recall_prototype": rate(oos, prototype),
        }
    ordered = sorted(nearest)
    report = {
        "kind": "clinc_dev_nearest_train_char_tfidf_similarity_sensitivity_not_new_test",
        "source_sha256": source_sha,
        "prediction_artifacts": [str(LEXICAL), str(PROTOTYPE)],
        "train_rows_including_oos": len(train),
        "development_rows": len(dev),
        "features_fit": xtrain.shape[1],
        "nearest_train_char_3_to_5gram_cosine_median_p90_p95": [ordered[int((len(ordered)-1)*q)] for q in (0.5,0.9,0.95)],
        "strata": {name: per_bin(indices) for name, indices in bins(nearest).items()},
        "sklearn_version": sklearn.__version__,
        "locked_test_examined": 0,
        "limits": "Character n-gram proximity is neither proven paraphrase leakage nor an independent source shift; stratification uses public development text and two already-selected models. Thresholds fixed before computation; no training or threshold tuning on this screen."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CLINC similarity strata: low {report['strata']['low_lt_0.8']['rows']}, high {report['strata']['high_ge_0.95']['rows']}")


if __name__ == "__main__":
    main()
