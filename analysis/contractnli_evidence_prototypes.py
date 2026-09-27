"""Train-only support/refute TF-IDF evidence prototypes on ContractNLI."""

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from contractnli_controls import ARCHIVE_SHA256, KS, download_train_dev, text_spans


OUT = Path("results/contractnli-evidence-prototypes.json")
METHODS = ("hypothesis", "pooled_evidence", "polarity_evidence")


def format_counts(count):
    return {"positive": count["positive"],
            "at_least_one_gold_at_k": {str(k): count[f"any_{k}"] / count["positive"] for k in KS},
            "all_gold_at_k": {str(k): count[f"all_{k}"] / count["positive"] for k in KS}}


def main():
    started = time.monotonic()
    data = download_train_dev()
    train, dev = data["train"], data["dev"]
    hypotheses = train["labels"]
    ids = sorted(hypotheses)
    if len(ids) != 17 or (len(train["documents"]), len(dev["documents"])) != (423, 61):
        raise ValueError("Unexpected source dimensions")
    spans = [text_spans(doc) for doc in train["documents"]]
    corpus = [span for doc_spans in spans for span in doc_spans]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    vectorizer.fit(corpus + [hypotheses[h]["hypothesis"] for h in ids])
    train_matrix = vectorizer.transform(corpus)
    query = vectorizer.transform(hypotheses[h]["hypothesis"] for h in ids)
    if train_matrix.shape[0] != 32895 or train_matrix.shape[1] != 30000:
        raise ValueError("Unmatched prior lexical control vocabulary")
    gold_indices = defaultdict(list)
    offset = 0
    for doc, doc_spans in zip(train["documents"], spans):
        annotations = doc["annotation_sets"][0]["annotations"]
        for h in ids:
            entry = annotations[h]
            if entry["choice"] in ("Entailment", "Contradiction"):
                gold_indices[(h, entry["choice"])].extend(offset + i for i in entry["spans"])
        offset += len(doc_spans)
    pooled, entail, contradict = [], [], []
    for h in ids:
        en, co = gold_indices[(h, "Entailment")], gold_indices[(h, "Contradiction")]
        if not en and not co:
            raise ValueError("Hypothesis has no gold training evidence")
        def center(indices):
            if not indices:
                return np.zeros(train_matrix.shape[1], dtype=np.float64)
            mean = np.asarray(train_matrix[indices].mean(axis=0)).reshape(1, -1)
            return normalize(mean)[0]
        pooled.append(center(en + co))
        entail.append(center(en))
        contradict.append(center(co))
    pooled, entail, contradict = [np.stack(v) for v in (pooled, entail, contradict)]
    counts = {method: {name: Counter() for name in ("all_positive", "Entailment", "Contradiction")} for method in METHODS}
    paired_effect_by_doc = []
    dev_spans_seen = 0
    for doc in dev["documents"]:
        doc_spans = text_spans(doc)
        dev_spans_seen += len(doc_spans)
        if not doc_spans:
            raise ValueError("Development document without spans")
        dvec = vectorizer.transform(doc_spans)
        score = {
            "hypothesis": (query @ dvec.T).toarray(),
            "pooled_evidence": np.asarray((dvec @ pooled.T).T),
            "polarity_evidence": np.maximum(np.asarray((dvec @ entail.T).T),
                                            np.asarray((dvec @ contradict.T).T)),
        }
        entries = doc["annotation_sets"][0]["annotations"]
        effect, contradiction_count = 0, 0
        for j, h in enumerate(ids):
            entry = entries[h]
            label = entry["choice"]
            if label == "NotMentioned":
                continue
            gold = set(entry["spans"])
            if not gold:
                raise ValueError("Positive case without gold evidence")
            ranks = {method: np.argsort(-score[method][j], kind="stable") for method in METHODS}
            for method in METHODS:
                for group in ("all_positive", label):
                    result = counts[method][group]
                    result["positive"] += 1
                    for k in KS:
                        selected = set(ranks[method][:k])
                        result[f"any_{k}"] += bool(selected & gold)
                        result[f"all_{k}"] += gold <= selected
            if label == "Contradiction":
                contradiction_count += 1
                effect += int(bool(set(ranks["polarity_evidence"][:5]) & gold)) - int(bool(set(ranks["pooled_evidence"][:5]) & gold))
        paired_effect_by_doc.append((effect, contradiction_count))
    if dev_spans_seen != 5102 or counts["hypothesis"]["all_positive"]["positive"] != 614:
        raise ValueError("Development span/positive count differs from prior control")
    baseline_any5 = counts["hypothesis"]["all_positive"]["any_5"] / 614
    if abs(baseline_any5 - 0.6612377850162866) > 1e-12:
        raise ValueError("Hypothesis retrieval does not reproduce registered lexical control")
    draw = np.random.default_rng(511)
    effects = np.array(paired_effect_by_doc)
    boot = []
    for _ in range(1000):
        sample = effects[draw.integers(0, len(effects), len(effects))]
        if sample[:, 1].sum():
            boot.append(sample[:, 0].sum() / sample[:, 1].sum())
    report = {
        "kind": "contractnli_train_supervised_lexical_evidence_prototype_development_only",
        "source_sha256": ARCHIVE_SHA256,
        "train_documents": 423,
        "development_documents": 61,
        "train_spans": len(corpus),
        "development_spans": dev_spans_seen,
        "hypotheses": len(ids),
        "training_evidence_span_counts": {
            label: {"min": min(len(gold_indices[(h, label)]) for h in ids),
                    "median": statistics.median(len(gold_indices[(h, label)]) for h in ids),
                    "max": max(len(gold_indices[(h, label)]) for h in ids)}
            for label in ("Entailment", "Contradiction")
        },
        "methods": {method: {group: format_counts(value) for group, value in groups.items()}
                    for method, groups in counts.items()},
        "contradiction_any_at_5_paired_polarity_minus_pooled": float(effects[:, 0].sum() / effects[:, 1].sum()),
        "contradiction_any_at_5_document_cluster_bootstrap95": [float(v) for v in np.quantile(boot, (0.025, 0.975))],
        "test_documents_examined": 0,
        "libraries": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.monotonic() - started,
        "limitations": "Fixed 17 hypotheses with train-supervised lexical span centroids, public dev only; no ternary classifier, learned neural encoder, probability calibration, CPU per-request RSS/latency, or test access. At least one marked span is a weak evidence criterion."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ContractNLI evidence prototypes: contradiction any@5 {report['methods']['hypothesis']['Contradiction']['at_least_one_gold_at_k']['5']:.3f} hypothesis / {report['methods']['pooled_evidence']['Contradiction']['at_least_one_gold_at_k']['5']:.3f} pooled / {report['methods']['polarity_evidence']['Contradiction']['at_least_one_gold_at_k']['5']:.3f} polarity")


if __name__ == "__main__":
    main()
