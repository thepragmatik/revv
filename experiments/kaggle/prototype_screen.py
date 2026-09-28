"""Frozen MiniLM class prototypes versus label-text vectors on pinned CLINC."""

import hashlib
import json
import math
import platform
import statistics
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

OUT = Path("/kaggle/working/revv-prototype-screen.json")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
DATA_BLOB = "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"
DATA_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
SEED = "revv-clinc-calibration-v1"
STATE_LIMIT = 64
LABEL_LIMIT = 16
BATCH = 64


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fold_key(label, text):
    return hashlib.sha256(f"{SEED}\0{label}\0{normal(text)}".encode()).hexdigest()


def fetch_data():
    url = f"https://raw.githubusercontent.com/clinc/oos-eval/{DATA_COMMIT}/data/data_full.json"
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("CLINC source exceeds 4 MB cap")
    actual_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if actual_blob != DATA_BLOB:
        raise ValueError("Pinned CLINC Git blob mismatch")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def split(dataset):
    train_normal = {normal(text) for text, _ in dataset["train"] + dataset["oos_train"]}
    clean = [(text, label) for text, label in dataset["val"] + dataset["oos_val"] if normal(text) not in train_normal]
    if len(clean) != 3097:
        raise ValueError("CLINC validation overlap count changed")
    grouped = defaultdict(list)
    for row in clean:
        grouped[row[1]].append(row)
    dev, cal = [], []
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda row: fold_key(label, row[0]))
        dev.extend(ordered[:len(ordered) // 2])
        cal.extend(ordered[len(ordered) // 2:])
    dev.sort(key=lambda row: fold_key(row[1], row[0]))
    cal.sort(key=lambda row: fold_key(row[1], row[0]))
    labels = sorted({label for _, label in dataset["train"]})
    if (len(dev), len(cal), len(labels)) != (1547, 1550, 150):
        raise ValueError("Unexpected validation fold or label sizes")
    if {normal(row[0]) for row in dev} & {normal(row[0]) for row in cal}:
        raise ValueError("Duplicate text crosses development and calibration")
    return labels, dev, cal


def encode(model, tokenizer, texts, limit, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=limit, return_special_tokens_mask=True)
    enc.pop("special_tokens_mask")
    enc = {name: value.to(device) for name, value in enc.items()}
    mask = enc["attention_mask"].bool()
    hidden = model(**enc).last_hidden_state
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return F.normalize(pooled, dim=-1)


def wilson(hits, total):
    z = statistics.NormalDist().inv_cdf(0.975)
    p = hits / total
    mid = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [mid - half, mid + half]


def pooled_vectors(model, tokenizer, rows, limit, device):
    vectors = []
    truncations = 0
    with torch.inference_mode():
        for offset in range(0, len(rows), BATCH):
            texts = [text for text, _ in rows[offset:offset + BATCH]]
            raw_lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
            truncations += sum(length > limit for length in raw_lengths)
            vectors.append(encode(model, tokenizer, texts, limit, device))
    return torch.cat(vectors), truncations


def bootstrap_delta(rows, a, b):
    by_label = defaultdict(list)
    for i, (_, label) in enumerate(rows):
        if label != "oos":
            by_label[label].append(int(b[i] == label) - int(a[i] == label))
    generator = np.random.default_rng(313)
    keys = sorted(by_label)
    draws = [np.mean([delta for key in generator.choice(keys, len(keys), replace=True) for delta in by_label[key]]) for _ in range(1000)]
    return [float(np.mean([x for v in by_label.values() for x in v])), *[float(x) for x in np.quantile(draws, [0.025, 0.975])]]


def metrics(scores, rows, calibration, labels):
    ndev = len(rows)
    dev, cal = scores[:ndev], scores[ndev:]
    cal_known = np.array([label != "oos" for _, label in calibration])
    threshold = float(np.quantile(cal[cal_known].max(axis=1), 0.05, method="higher"))
    ranking = np.argsort(-dev, axis=1, kind="stable")
    closed = [labels[i] for i in ranking[:, 0]]
    maxima = dev.max(axis=1)
    selected = ["oos" if maxima[i] < threshold else closed[i] for i in range(ndev)]
    known = [i for i, (_, label) in enumerate(rows) if label != "oos"]
    oos = [i for i, (_, label) in enumerate(rows) if label == "oos"]
    hits = sum(closed[i] == rows[i][1] for i in known)
    rejected = sum(selected[i] == "oos" for i in oos)
    ranks = {str(k): sum(rows[i][1] in [labels[j] for j in ranking[i, :k]] for i in known) / len(known) for k in (1, 2, 5, 10)}
    result = {
        "known_calibration_5th_percentile": threshold,
        "known_closed_accuracy": hits / len(known),
        "known_closed_wilson95": wilson(hits, len(known)),
        "known_after_defer_accuracy": sum(selected[i] == rows[i][1] for i in known) / len(known),
        "known_defer_rate": sum(selected[i] == "oos" for i in known) / len(known),
        "oos_recall": rejected / len(oos),
        "oos_wilson95": wilson(rejected, len(oos)),
        "known_candidate_recall_at_k": ranks,
    }
    return result, closed, selected


def main():
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Expected Kaggle GPU; no CPU performance claim")
    torch.manual_seed(313)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    dataset, data_sha = fetch_data()
    if data_sha != DATA_SHA256:
        raise ValueError("Pinned dataset hash mismatch")
    labels, dev, cal = split(dataset)
    train = dataset["train"]
    if len(train) != 15000 or any(label == "oos" for _, label in train):
        raise ValueError("Unexpected labelled training split")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True, trust_remote_code=False).to(device).eval()
    load_seconds = time.perf_counter() - started
    print(f"Loaded frozen {MODEL}; fitting only class centroids", flush=True)
    fitting_started = time.perf_counter()
    train_vec, train_truncated = pooled_vectors(model, tokenizer, train, STATE_LIMIT, device)
    prototypes = []
    for label in labels:
        indices = [i for i, (_, y) in enumerate(train) if y == label]
        if len(indices) != 100:
            raise ValueError("Expected 100 training examples per in-scope intent")
        prototypes.append(F.normalize(train_vec[indices].mean(dim=0), dim=0))
    prototypes = torch.stack(prototypes)
    fit_seconds = time.perf_counter() - fitting_started
    del train_vec
    label_texts = [(label.replace("_", " "), label) for label in labels]
    label_vec, label_truncated = pooled_vectors(model, tokenizer, label_texts, LABEL_LIMIT, device)
    infer_started = time.perf_counter()
    rows = dev + cal
    validation_vec, validation_truncated = pooled_vectors(model, tokenizer, rows, STATE_LIMIT, device)
    label_scores = (validation_vec @ label_vec.T).cpu().numpy()
    proto_scores = (validation_vec @ prototypes.T).cpu().numpy()
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - infer_started
    label_result, label_closed, label_selected = metrics(label_scores, dev, cal, labels)
    proto_result, proto_closed, proto_selected = metrics(proto_scores, dev, cal, labels)
    delta = bootstrap_delta(dev, label_closed, proto_closed)
    if time.perf_counter() - started > 330:
        raise TimeoutError("Prototype screen exceeded internal 330-second budget")
    report = {
        "kind": "frozen_clinc_label_vs_prototype_development_not_model_win",
        "model": MODEL,
        "model_revision": REVISION,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "data_commit": DATA_COMMIT,
        "data_sha256": data_sha,
        "split_seed": SEED,
        "training_known": len(train),
        "development_rows": len(dev),
        "calibration_rows": len(cal),
        "development_oos": sum(label == "oos" for _, label in dev),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "state_token_limit": STATE_LIMIT,
        "label_token_limit": LABEL_LIMIT,
        "train_rows_truncated": train_truncated,
        "validation_rows_truncated": validation_truncated,
        "label_rows_truncated": label_truncated,
        "peak_torch_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "model_load_seconds": load_seconds,
        "train_encode_and_centroid_seconds": fit_seconds,
        "validation_encode_and_score_seconds": validation_seconds,
        "methods": {"label_text": label_result, "training_prototype": proto_result},
        "known_closed_paired_prototype_minus_label": delta[0],
        "known_closed_paired_cluster_bootstrap95": delta[1:],
        "predictions": [
            {"example_sha256": hashlib.sha256(normal(text).encode()).hexdigest(), "gold": gold,
             "label_text": label_selected[i], "training_prototype": proto_selected[i],
             "label_closed": label_closed[i], "prototype_closed": proto_closed[i]}
            for i, (text, gold) in enumerate(dev)
        ],
        "limitations": "Frozen sentence encoder; prototypes use labelled CLINC training text. Public development fold, label-source overlap may favor semantic names, no local CPU serving latency/RSS or probability calibration, no locked test or multi-question evidence."
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Prototype screen complete: label {label_result['known_closed_accuracy']:.3f}, prototype {proto_result['known_closed_accuracy']:.3f}; R@5 {proto_result['known_candidate_recall_at_k']['5']:.3f}", flush=True)


if __name__ == "__main__":
    main()
