"""Frozen MiniLM pooled versus token evidence on pinned CLINC development."""

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


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
DATA_BLOB = "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"
SPLIT_SEED = "revv-clinc-calibration-v1"
OUT = Path("/kaggle/working/revv-frozen-matching.json")
BATCH_SIZE = 32
MAX_STATE_TOKENS = 64
MAX_LABEL_TOKENS = 16


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fold_key(label, text):
    return hashlib.sha256(f"{SPLIT_SEED}\0{label}\0{normal(text)}".encode()).hexdigest()


def fetch_data():
    url = f"https://raw.githubusercontent.com/clinc/oos-eval/{DATA_COMMIT}/data/data_full.json"
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Pinned dataset exceeded 4 MB cap")
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != DATA_BLOB:
        raise ValueError("Pinned dataset Git blob mismatch")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def split(dataset):
    train = dataset["train"] + dataset["oos_train"]
    known_texts = {normal(text) for text, _ in train}
    clean = [(text, label) for text, label in dataset["val"] + dataset["oos_val"] if normal(text) not in known_texts]
    if len(clean) != 3097:
        raise ValueError("Unexpected cleaned validation size")
    grouped = defaultdict(list)
    for row in clean:
        grouped[row[1]].append(row)
    development, calibration = [], []
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda row: fold_key(label, row[0]))
        development.extend(ordered[: len(ordered) // 2])
        calibration.extend(ordered[len(ordered) // 2 :])
    development.sort(key=lambda row: fold_key(row[1], row[0]))
    calibration.sort(key=lambda row: fold_key(row[1], row[0]))
    labels = sorted({label for _, label in dataset["train"]})
    if len(development) != 1547 or len(calibration) != 1550 or len(labels) != 150:
        raise ValueError("Validation folds or label count differ from lexical floor")
    if {normal(row[0]) for row in development} & {normal(row[0]) for row in calibration}:
        raise ValueError("Normalized text crosses development and calibration folds")
    return labels, development, calibration


def encode(model, tokenizer, texts, limit, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=limit, return_special_tokens_mask=True)
    special = enc.pop("special_tokens_mask").to(device).bool()
    enc = {name: tensor.to(device) for name, tensor in enc.items()}
    attn = enc["attention_mask"].bool()
    hidden = model(**enc).last_hidden_state
    pooled = F.normalize((hidden * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1).clamp(min=1).unsqueeze(-1), dim=-1)
    token = F.normalize(hidden, dim=-1)
    evidence_mask = attn & ~special
    return pooled, token, evidence_mask


def wilson(hits, total):
    z = statistics.NormalDist().inv_cdf(0.975)
    p = hits / total
    mid = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [mid - half, mid + half]


def main():
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("No Kaggle CUDA GPU; do not emit a CPU model claim")
    torch.manual_seed(271)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    dataset, data_sha256 = fetch_data()
    labels, development, calibration = split(dataset)
    if data_sha256 != "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0":
        raise ValueError("Pinned dataset SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True, trust_remote_code=False).to(device).eval()
    loaded_seconds = time.perf_counter() - started
    print(f"Loaded pinned {MODEL} on {torch.cuda.get_device_name(device)} in {loaded_seconds:.1f}s", flush=True)
    label_texts = [label.replace("_", " ") for label in labels]
    rows = development + calibration
    pooled_scores, evidence_scores = [], []
    pooled_score_ms, evidence_score_ms = [], []
    state_truncated = 0
    with torch.inference_mode():
        label_pool, label_tok, label_mask = encode(model, tokenizer, label_texts, MAX_LABEL_TOKENS, device)
        if not label_mask.any(dim=1).all():
            raise ValueError("A label tokenized without evidence tokens")
        infer_start = time.perf_counter()
        for batch_index, start in enumerate(range(0, len(rows), BATCH_SIZE)):
            texts = [text for text, _ in rows[start : start + BATCH_SIZE]]
            # Record overflow rather than silently passing long text as if complete.
            raw_lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
            state_truncated += sum(length > MAX_STATE_TOKENS for length in raw_lengths)
            state_pool, state_tok, state_mask = encode(model, tokenizer, texts, MAX_STATE_TOKENS, device)
            values = {}
            for method in (("pooled", "token_maxsim") if batch_index % 2 == 0 else ("token_maxsim", "pooled")):
                torch.cuda.synchronize()
                score_start = time.perf_counter()
                if method == "pooled":
                    values[method] = state_pool @ label_pool.T
                else:
                    sims = torch.einsum("bld,kmd->bklm", state_tok, label_tok)
                    sims = sims.masked_fill(~state_mask[:, None, :, None], -1e4)
                    evidence = sims.max(dim=2).values
                    values[method] = (evidence * label_mask[None, :, :]).sum(dim=-1) / label_mask.sum(dim=-1)[None, :]
                torch.cuda.synchronize()
                (pooled_score_ms if method == "pooled" else evidence_score_ms).append((time.perf_counter() - score_start) * 1000)
            pooled, evidence = values["pooled"], values["token_maxsim"]
            pooled_scores.append(pooled.cpu().numpy())
            evidence_scores.append(evidence.cpu().numpy())
            if time.perf_counter() - started > 330:
                raise TimeoutError("Frozen pilot exceeded 330-second internal budget before all folds")
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - infer_start
    pooled_scores = np.concatenate(pooled_scores)
    evidence_scores = np.concatenate(evidence_scores)
    ndev = len(development)
    truth = [label for _, label in development]
    known = [i for i, label in enumerate(truth) if label != "oos"]
    unknown = [i for i, label in enumerate(truth) if label == "oos"]
    cal_known = np.array([label != "oos" for _, label in calibration])
    result = {}
    predicted = {}
    for method, scores in (("pooled", pooled_scores), ("token_maxsim", evidence_scores)):
        cal = scores[ndev:]
        threshold = float(np.quantile(cal[cal_known].max(axis=1), 0.05, method="higher"))
        dev = scores[:ndev]
        winners = dev.argmax(axis=1)
        maximum = dev.max(axis=1)
        selected = ["oos" if float(maximum[i]) < threshold else labels[int(winners[i])] for i in range(ndev)]
        closed = sum(labels[int(winners[i])] == truth[i] for i in known)
        known_hits = sum(selected[i] == truth[i] for i in known)
        oos_hits = sum(selected[i] == "oos" for i in unknown)
        known_deferred = sum(selected[i] == "oos" for i in known)
        result[method] = {
            "threshold_calibration_known_5th_percentile": threshold,
            "known_closed_accuracy": closed / len(known),
            "known_closed_wilson95": wilson(closed, len(known)),
            "known_accuracy_after_defer": known_hits / len(known),
            "known_defer_rate": known_deferred / len(known),
            "oos_recall": oos_hits / len(unknown),
            "oos_wilson95": wilson(oos_hits, len(unknown)),
        }
        predicted[method] = {"closed": [labels[int(i)] for i in winners], "selected": selected, "maximum": maximum}
    paired = [int(predicted["token_maxsim"]["closed"][i] == truth[i]) - int(predicted["pooled"]["closed"][i] == truth[i]) for i in known]
    by_label = defaultdict(list)
    for index, effect in zip(known, paired):
        by_label[truth[index]].append(effect)
    random = np.random.default_rng(271)
    keys = sorted(by_label)
    boot = [np.mean([v for key in random.choice(keys, size=len(keys), replace=True) for v in by_label[key]]) for _ in range(1000)]
    predictions = [
        {
            "example_sha256": hashlib.sha256(normal(text).encode()).hexdigest(),
            "gold": label,
            "pooled": predicted["pooled"]["selected"][i],
            "token_maxsim": predicted["token_maxsim"]["selected"][i],
            "pooled_max_score": float(predicted["pooled"]["maximum"][i]),
            "token_maxsim_max_score": float(predicted["token_maxsim"]["maximum"][i]),
        }
        for i, (text, label) in enumerate(development)
    ]
    report = {
        "kind": "frozen_pretrained_clinc_matching_pilot_not_decision_model_win",
        "model": MODEL,
        "model_revision": REVISION,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "data_commit": DATA_COMMIT,
        "data_sha256": data_sha256,
        "split_seed": SPLIT_SEED,
        "development_rows": ndev,
        "calibration_rows": len(calibration),
        "development_oos": len(unknown),
        "calibration_oos": int((~cal_known).sum()),
        "state_max_tokens": MAX_STATE_TOKENS,
        "state_rows_truncated": state_truncated,
        "label_max_tokens": MAX_LABEL_TOKENS,
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "peak_torch_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "loaded_seconds": loaded_seconds,
        "inference_seconds_for_development_and_calibration_batches": inference_seconds,
        "pooled_score_batch_ms": pooled_score_ms,
        "token_maxsim_score_batch_ms": evidence_score_ms,
        "known_closed_paired_token_minus_pool": float(np.mean(paired)),
        "known_closed_paired_cluster_bootstrap95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "methods": result,
        "predictions": predictions,
        "limitations": "Frozen pretrained sentence model on untrained underscore-derived label descriptions; no trained cross encoder, probability calibration, local CPU RSS/latency, long-state or multi-question evidence, or locked test. GPU time includes tokenization in batches, not local CPU request time."
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen pilot complete: pooled {result['pooled']['known_closed_accuracy']:.3f}, MaxSim {result['token_maxsim']['known_closed_accuracy']:.3f}; OOS {result['pooled']['oos_recall']:.3f}/{result['token_maxsim']['oos_recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
