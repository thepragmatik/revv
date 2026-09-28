"""Frozen NLI cross-scoring on document-disjoint evidence shortlists."""

import hashlib
import io
import json
import math
import platform
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "c4d86af4493123990d7762712de9ed730c876161"
SOURCE_URL = "https://raw.githubusercontent.com/stanfordnlp/contract-nli/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip"
SOURCE_SHA = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
LABELS = ("Entailment", "Contradiction", "NotMentioned")
OUT = Path("/kaggle/working/revv-frozen-nli-contract.json")
K = 5
BATCH = 32
TOKEN_LIMIT = 256


def fetch_author():
    with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
        raw = response.read(70_000_001)
    if len(raw) > 70_000_000 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Pinned author source exceeds cap or differs in SHA-256")
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        result = {}
        for name in ("train", "dev"):
            member = source.getinfo(f"contract-nli/{name}.json")
            if member.file_size > 25_000_000:
                raise ValueError("Unbounded source JSON")
            result[name] = json.loads(source.read(member))
        return result


def spans(doc):
    return [doc["text"][a:b] for a, b in doc["spans"]]


def fold(doc):
    return int(hashlib.sha256(str(doc["id"]).encode()).hexdigest(), 16) % 5


def assemble(documents, ids, hypotheses, query, prototypes, vectorizer):
    rows, covered = [], defaultdict(Counter)
    for doc_index, doc in enumerate(documents):
        texts = spans(doc)
        if not texts:
            raise ValueError("Document has no spans")
        vec = vectorizer.transform(texts)
        scores = np.asarray((vec @ prototypes.T).T)
        annotations = doc["annotation_sets"][0]["annotations"]
        for hindex, h in enumerate(ids):
            entry = annotations[h]
            selected = np.argsort(-scores[hindex], kind="stable")[:K]
            if entry["choice"] != "NotMentioned":
                gold = set(entry["spans"])
                for key in ("all", entry["choice"]):
                    covered[key]["positive"] += 1
                    covered[key]["any_gold"] += bool(gold & set(selected))
                    covered[key]["all_gold"] += gold <= set(selected)
            rows.append({"doc_index": doc_index, "hyp_index": hindex, "gold": entry["choice"],
                         "premises": [texts[int(i)] for i in selected], "hypothesis": hypotheses[h]["hypothesis"],
                         "retrieval_top1": float(scores[hindex, selected[0]]),
                         "retrieval_mean5": float(np.mean(scores[hindex, selected]))})
    return rows, {key: {"positive": count["positive"], "any_gold_at_5": count["any_gold"] / count["positive"],
                         "all_gold_at_5": count["all_gold"] / count["positive"]} for key, count in covered.items()}


def classify_metrics(truth, pred, probs):
    f1, recalls = {}, {}
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        recalls[label] = tp / (tp + fn) if tp + fn else 0.0
    indices = [LABELS.index(t) for t in truth]
    return {"accuracy": sum(t == p for t, p in zip(truth, pred)) / len(truth),
            "macro_f1": sum(f1.values()) / len(LABELS), "class_recall": recalls, "class_f1": f1,
            "multiclass_brier_sum": float(np.mean(np.sum((probs - np.eye(3)[indices]) ** 2, axis=1))),
            "nll": float(np.mean(-np.log(np.maximum(probs[np.arange(len(truth)), indices], 1e-12))))}


def main():
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU required for this control")
    torch.manual_seed(811)
    torch.set_num_threads(2)
    data = fetch_author()
    train, dev = data["train"]["documents"], data["dev"]["documents"]
    hypotheses = data["train"]["labels"]
    ids = sorted(hypotheses)
    if len(train) != 423 or len(dev) != 61 or len(ids) != 17 or ids != sorted(data["dev"]["labels"]):
        raise ValueError("Author split dimensions changed")
    fit_docs, calib_docs = [doc for doc in train if fold(doc) != 0], [doc for doc in train if fold(doc) == 0]
    if len(calib_docs) != 76 or len(fit_docs) != 347:
        raise ValueError("Unexpected document-disjoint calibration split")
    train_spans = [spans(doc) for doc in train]
    corpus = [sentence for group in train_spans for sentence in group]
    retriever = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    retriever.fit(corpus + [hypotheses[h]["hypothesis"] for h in ids])
    vectors = retriever.transform(corpus)
    evidence_calibration = defaultdict(list)
    evidence_development = defaultdict(list)
    start = 0
    for doc, snippets in zip(train, train_spans):
        entries = doc["annotation_sets"][0]["annotations"]
        for h in ids:
            if entries[h]["choice"] != "NotMentioned":
                indices = [start + int(i) for i in entries[h]["spans"]]
                evidence_development[h].extend(indices)
                if fold(doc) != 0:
                    evidence_calibration[h].extend(indices)
        start += len(snippets)
    def make_centroids(evidence):
        prototypes = []
        for h in ids:
            if not evidence[h]:
                raise ValueError("No fit evidence for a hypothesis")
            prototypes.append(normalize(np.asarray(vectors[evidence[h]].mean(axis=0)).reshape(1, -1))[0])
        return np.stack(prototypes)
    calibration_prototypes = make_centroids(evidence_calibration)
    development_prototypes = make_centroids(evidence_development)
    query = retriever.transform(hypotheses[h]["hypothesis"] for h in ids)
    calibration, cal_coverage = assemble(calib_docs, ids, hypotheses, query, calibration_prototypes, retriever)
    development, dev_coverage = assemble(dev, ids, hypotheses, query, development_prototypes, retriever)
    if len(calibration) != 1292 or len(development) != 1037 or dev_coverage["all"]["positive"] != 614:
        raise ValueError("NLI pair/example count changed")
    if abs(dev_coverage["all"]["any_gold_at_5"] - 0.9511400651465798) > 1e-12:
        raise ValueError("Dev evidence shortlist differs from registered pooled prototype control")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, use_safetensors=True,
                                                               trust_remote_code=False).cuda().eval()
    mapping = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if mapping != {0: "contradiction", 1: "entailment", 2: "neutral"}:
        raise ValueError("NLI checkpoint label mapping differs from preregistration")
    loaded_seconds = time.monotonic() - started
    rows = calibration + development
    premises = [premise for row in rows for premise in row["premises"]]
    questions = [row["hypothesis"] for row in rows for _ in row["premises"]]
    if len(premises) != (1292 + 1037) * K:
        raise ValueError("Unexpected NLI pair count")
    outputs, truncated = [], 0
    infer_start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(premises), BATCH):
            p = premises[offset:offset+BATCH]
            q = questions[offset:offset+BATCH]
            lengths = tokenizer(p, q, truncation=False)["input_ids"]
            truncated += sum(len(ids) > TOKEN_LIMIT for ids in lengths)
            enc = tokenizer(p, q, padding=True, truncation=True, max_length=TOKEN_LIMIT, return_tensors="pt")
            prob = F.softmax(model(**{k: v.cuda() for k, v in enc.items()}).logits.float(), dim=-1)
            outputs.append(prob.cpu().numpy())
            if time.monotonic() - started > 480:
                raise TimeoutError("NLI experiment exceeded internal 480-second bound")
    torch.cuda.synchronize()
    infer_seconds = time.monotonic() - infer_start
    pair_prob = np.concatenate(outputs).reshape((len(rows), K, 3))
    def features(part, probs):
        data = np.zeros((len(part), 6 + len(ids)), dtype=np.float64)
        data[:, 0] = probs[:, :, 1].max(axis=1)  # entailment
        data[:, 1] = probs[:, :, 0].max(axis=1)  # contradiction
        data[:, 2] = probs[:, :, 2].min(axis=1)  # neutral cannot alone prove NotMentioned
        data[:, 3] = probs[:, :, 1].mean(axis=1)
        data[:, 4] = probs[:, :, 0].mean(axis=1)
        data[:, 5] = probs[:, :, 2].mean(axis=1)
        for i, row in enumerate(part):
            data[i, 6 + row["hyp_index"]] = 1.0
        return data
    ncal = len(calibration)
    learner = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
    learner.fit(features(calibration, pair_prob[:ncal]), [row["gold"] for row in calibration])
    if int(learner.n_iter_[0]) >= 300:
        raise RuntimeError("Training-only NLI aggregation head hit iteration cap")
    dev_prob_raw = learner.predict_proba(features(development, pair_prob[ncal:]))
    dev_prob = dev_prob_raw[:, [list(learner.classes_).index(x) for x in LABELS]]
    predicted = [LABELS[int(i)] for i in dev_prob.argmax(axis=1)]
    truth = [row["gold"] for row in development]
    model_metrics = classify_metrics(truth, predicted, dev_prob)
    prior_counts = {h: Counter(doc["annotation_sets"][0]["annotations"][h]["choice"] for doc in train) for h in ids}
    priors = {h: np.array([(prior_counts[h][y] + 1) / (len(train) + 3) for y in LABELS]) for h in ids}
    prior_prob = np.stack([priors[ids[row["hyp_index"]]] for row in development])
    prior_pred = [LABELS[int(i)] for i in prior_prob.argmax(axis=1)]
    prior_metrics = classify_metrics(truth, prior_pred, prior_prob)
    by_doc = np.zeros(len(dev), dtype=np.float64)
    for i, row in enumerate(development):
        by_doc[row["doc_index"]] += int(predicted[i] == truth[i]) - int(prior_pred[i] == truth[i])
    random = np.random.default_rng(811)
    boot = [float(by_doc[random.integers(0, len(dev), len(dev))].sum() / len(development)) for _ in range(1000)]
    report = {
        "kind": "frozen_nli_cross_on_selected_contract_spans_dev_control_not_local_win",
        "model": MODEL, "revision": REVISION, "model_parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": SOURCE_SHA, "calibration_retrieval_fit_documents": len(fit_docs),
        "development_retrieval_and_prior_fit_documents": len(train), "calibration_documents": len(calib_docs),
        "development_documents": len(dev), "hypotheses_per_document": len(ids), "evidence_k": K,
        "evidence_recall_calibration": cal_coverage, "evidence_recall_development": dev_coverage,
        "token_max_length": TOKEN_LIMIT, "truncated_nli_pairs": truncated, "nli_pairs": len(premises),
        "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(), "peak_torch_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "loaded_seconds_including_source_and_weights": loaded_seconds, "frozen_pair_batch_inference_seconds": infer_seconds,
        "aggregation_head_iterations": int(learner.n_iter_[0]),
        "training_only_prior_fit_423": prior_metrics, "frozen_nli_with_calibration_head": model_metrics,
        "paired_nli_minus_prior_accuracy": model_metrics["accuracy"] - prior_metrics["accuracy"],
        "paired_document_cluster_bootstrap95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "test_documents_examined": 0,
        "limitations": "Frozen SNLI/MultiNLI NLI checkpoint; calibration examples use 347-document evidence prototypes excluding themselves, dev evidence and prior use all 423 training documents. Shared retrieval vocabulary fits training spans including the 76 calibration texts without their labels. Same 61 public dev documents. NotMentioned can have similar spans. No local CPU latency/RSS, no separate probability calibration holdout, no locked test or full-document cross control."
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Frozen NLI control: dev accuracy {model_metrics['accuracy']:.3f}, fit-only prior {prior_metrics['accuracy']:.3f}; scored {len(premises)} pairs", flush=True)


if __name__ == "__main__":
    main()
