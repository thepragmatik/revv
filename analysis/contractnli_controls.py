"""Registered ContractNLI train/dev priors and full-span lexical evidence screen."""

import hashlib
import json
import math
import statistics
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


ARCHIVE_URL = "https://raw.githubusercontent.com/stanfordnlp/contract-nli/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip"
ARCHIVE_SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
LABELS = ("Entailment", "Contradiction", "NotMentioned")
KS = (1, 3, 5, 10)
OUT = Path("results/contractnli-controls.json")


def download_train_dev():
    with tempfile.TemporaryDirectory(prefix="revv-contract-control-") as directory:
        path = Path(directory) / "author-release.zip"
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(ARCHIVE_URL, timeout=90) as response, path.open("wb") as file:
            while block := response.read(4_000_000):
                size += len(block)
                if size > 70_000_000:
                    raise ValueError("Source exceeds 70 MB cap")
                digest.update(block)
                file.write(block)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Pinned author archive SHA mismatch")
        with zipfile.ZipFile(path) as archive:
            outputs = {}
            for split in ("train", "dev"):
                member = archive.getinfo(f"contract-nli/{split}.json")
                if member.file_size > 25_000_000:
                    raise ValueError("Source JSON size exceeds preregistration")
                outputs[split] = json.loads(archive.read(member))
            return outputs


def text_spans(doc):
    text = doc["text"]
    return [text[start:end] for start, end in doc["spans"]]


def metrics(truth, pred, probs):
    accuracy = sum(a == b for a, b in zip(truth, pred)) / len(truth)
    recalls, f1 = {}, {}
    for label in LABELS:
        tp = sum(a == label and b == label for a, b in zip(truth, pred))
        fp = sum(a != label and b == label for a, b in zip(truth, pred))
        fn = sum(a == label and b != label for a, b in zip(truth, pred))
        recalls[label] = tp / (tp + fn) if tp + fn else None
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    brier = np.mean([sum((p[label] - int(t == label)) ** 2 for label in LABELS) for t, p in zip(truth, probs)])
    nll = np.mean([-math.log(p[t]) for t, p in zip(truth, probs)])
    return {"accuracy": accuracy, "macro_f1": statistics.mean(f1.values()), "class_f1": f1,
            "class_recall": recalls, "multiclass_brier_sum": float(brier), "nll": float(nll)}


def main():
    start_time = time.monotonic()
    data = download_train_dev()
    train, dev = data["train"], data["dev"]
    hypo_train = train["labels"]
    hypotheses = dev["labels"]
    if sorted(hypotheses) != sorted(hypo_train) or len(hypotheses) != 17:
        raise ValueError("Fixed hypothesis set differs across train and dev")
    if len(train["documents"]) != 423 or len(dev["documents"]) != 61:
        raise ValueError("Unexpected author document counts")
    ids = sorted(hypotheses)
    counts = {h: Counter(doc["annotation_sets"][0]["annotations"][h]["choice"] for doc in train["documents"]) for h in ids}
    priors = {h: {label: (counts[h][label] + 1) / (len(train["documents"]) + len(LABELS)) for label in LABELS} for h in ids}
    global_counts = sum((counts[h] for h in ids), Counter())
    global_label = max(LABELS, key=lambda label: (global_counts[label], -LABELS.index(label)))
    pred_label = {h: max(LABELS, key=lambda label: (priors[h][label], -LABELS.index(label))) for h in ids}
    truth, guessed, probabilities = [], [], []
    per_document_hits = []
    for doc in dev["documents"]:
        annotations = doc["annotation_sets"][0]["annotations"]
        doc_hits = 0
        for h in ids:
            y = annotations[h]["choice"]
            truth.append(y)
            guessed.append(pred_label[h])
            probabilities.append(priors[h])
            doc_hits += y == pred_label[h]
        per_document_hits.append(doc_hits / len(ids))
    if len(truth) != 1037:
        raise ValueError("Unexpected development decision count")
    baseline = metrics(truth, guessed, probabilities)
    baseline["global_majority_label"] = global_label
    baseline["global_majority_accuracy"] = sum(y == global_label for y in truth) / len(truth)
    bootstrap = np.random.default_rng(4242)
    baseline["document_cluster_bootstrap95_accuracy"] = [float(x) for x in np.quantile(
        [np.mean(bootstrap.choice(per_document_hits, size=len(per_document_hits), replace=True)) for _ in range(1000)], [0.025, 0.975])]

    corpus = [span for doc in train["documents"] for span in text_spans(doc)]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    vectorizer.fit(corpus + [hypotheses[h]["hypothesis"] for h in ids])
    query = vectorizer.transform(hypotheses[h]["hypothesis"] for h in ids)
    row_counts = defaultdict(Counter)
    documents_without_spans = 0
    dev_spans = 0
    for doc in dev["documents"]:
        spans = text_spans(doc)
        dev_spans += len(spans)
        if not spans:
            documents_without_spans += 1
            continue
        scores = (query @ vectorizer.transform(spans).T).toarray()
        annotations = doc["annotation_sets"][0]["annotations"]
        for j, h in enumerate(ids):
            entry = annotations[h]
            if entry["choice"] == "NotMentioned":
                continue
            gold = set(entry["spans"])
            if not gold:
                raise ValueError("Positive example has no evidence")
            rank = np.argsort(-scores[j], kind="stable")
            group = row_counts[entry["choice"]]
            group["positive"] += 1
            group["gold_span_total"] += len(gold)
            for k in KS:
                selected = set(rank[:k])
                group[f"any_at_{k}"] += bool(selected & gold)
                group[f"all_at_{k}"] += gold <= selected
    all_counts = sum(row_counts.values(), Counter())
    if all_counts["positive"] != 614:
        raise ValueError("Unexpected positive evidence count")
    if documents_without_spans:
        raise ValueError("Development document has no spans")
    def formatted(count):
        return {"positive": count["positive"], "gold_span_mean": count["gold_span_total"] / count["positive"],
                "at_least_one_gold_at_k": {str(k): count[f"any_at_{k}"] / count["positive"] for k in KS},
                "all_gold_at_k": {str(k): count[f"all_at_{k}"] / count["positive"] for k in KS}}
    report = {
        "kind": "contractnli_train_dev_cpu_priors_and_full_span_retrieval_not_model_claim",
        "source_sha256": ARCHIVE_SHA256,
        "train_documents": 423,
        "development_documents": 61,
        "hypotheses_per_document": 17,
        "test_documents_examined": 0,
        "baseline_per_hypothesis_laplace_prior": baseline,
        "evidence_word_unigram_bigram_tfidf": {
            "min_df": 2, "max_features": 30000, "features_fit": len(vectorizer.get_feature_names_out()),
            "train_spans_for_vocab": len(corpus), "development_spans": dev_spans,
            "positive_any_and_all": {name: formatted(count) for name, count in [
                ("all_positive", all_counts), ("Entailment", row_counts["Entailment"]), ("Contradiction", row_counts["Contradiction"])]},
        },
        "libraries": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.monotonic() - start_time,
        "limitations": "Only public development, lexical evidence similarity and label priors; no classifier using evidence, no retrieved-span reranking, no learned state encoder, no local CPU latency/RSS, no probability calibration on separate data or locked-test access."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ContractNLI controls: prior accuracy {baseline['accuracy']:.3f}, lexical evidence any@5 {report['evidence_word_unigram_bigram_tfidf']['positive_any_and_all']['all_positive']['at_least_one_gold_at_k']['5']:.3f}")


if __name__ == "__main__":
    main()
