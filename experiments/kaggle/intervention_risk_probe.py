"""Screen intervention-risk routing against confidence routing on synthetic edits.

This is a controlled diagnostic, not a public benchmark or a model-win claim.
The sentence and NLI backbones stay frozen; only two small logistic readouts fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from intervention_probe import generate_state


SEED = 20260928
N_STATES = 1024
COMPOSITION_GROUP_SIZE = 8
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_REVISION = "c4d86af4493123990d7762712de9ed730c876161"
LABELS = ("true", "false", "unknown")
NLI_LABEL_MAP = {0: "false", 1: "true", 2: "unknown"}
TOP_K = (1, 3, 5, 10)
GATE_RATES = (0.10, 0.25, 0.50)
ENCODE_BATCH = 64
NLI_BATCH = 32
SENTENCE_MAX_TOKENS = 128
NLI_MAX_TOKENS = 256
OUT = Path("/kaggle/working/revv-intervention-risk-probe.json")


def normalize(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def encode_texts(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), ENCODE_BATCH):
            batch = texts[start:start + ENCODE_BATCH]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=SENTENCE_MAX_TOKENS, return_tensors="pt")
            enc = {key: value.to(device) for key, value in enc.items()}
            hidden = model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(normalize(pooled.cpu().numpy()))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def score_nli(premises: list[str], hypotheses: list[str], tokenizer, model, device: str) -> tuple[np.ndarray, int, float]:
    outputs = []
    truncated = 0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(premises), NLI_BATCH):
            p = premises[start:start + NLI_BATCH]
            h = hypotheses[start:start + NLI_BATCH]
            raw = tokenizer(p, h, truncation=False)["input_ids"]
            truncated += sum(len(ids) > NLI_MAX_TOKENS for ids in raw)
            enc = tokenizer(p, h, padding=True, truncation=True,
                            max_length=NLI_MAX_TOKENS, return_tensors="pt")
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            outputs.append(probs[:, [1, 0, 2]])
    return (np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 3)),
            truncated, time.monotonic() - started)


def labels_from_probs(probs: np.ndarray) -> list[str]:
    return [LABELS[int(index)] for index in probs.argmax(axis=1)]


def feature_rows(state_vectors: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
    return np.concatenate((state_vectors, query_vectors,
                           np.abs(state_vectors - query_vectors),
                           state_vectors * query_vectors), axis=1)


def risk_feature_rows(state_vectors: np.ndarray, query_vectors: np.ndarray,
                      edit_vectors: np.ndarray) -> np.ndarray:
    return np.concatenate((state_vectors, query_vectors, edit_vectors,
                           np.abs(edit_vectors - query_vectors),
                           edit_vectors * query_vectors), axis=1)


def _split_group(group_id: str) -> str:
    if int(group_id[:8], 16) % 4 == 0:
        return "heldout_composition"
    if int(group_id[8:16], 16) % 5 == 0:
        return "calibration_composition"
    return "fit_composition"


def edit_text(edit: dict) -> str:
    action = "Add" if edit["operation"] == "add" else "Remove"
    return f"{action} this fact: {edit['text']}"


def post_edit_statements(record: dict) -> list[dict]:
    edit = record["interventions"]["uniform_candidate_sample"]["edit"]
    if edit["operation"] == "remove":
        result = [item for item in record["statements"] if item["id"] != edit["fact_id"]]
        if len(result) == len(record["statements"]):
            raise ValueError(f"Removal target missing in {record['state_id']}")
        return result
    if any(item["id"] == edit["fact_id"] for item in record["statements"]):
        raise ValueError(f"Addition target already present in {record['state_id']}")
    return record["statements"] + [{"id": edit["fact_id"], "kind": "fact", "text": edit["text"]}]


def build_transition_examples(records: list[dict], statement_embeddings: list[np.ndarray],
                               query_embeddings: np.ndarray, edit_embeddings: np.ndarray,
                               added_fact_embeddings: dict[str, np.ndarray]) -> tuple:
    examples = []
    pre_vectors, post_vectors, queries, edits = [], [], [], []
    original_statement_vectors_by_id = {}
    post_texts_by_id = {}
    cursor = 0
    for record_index, record in enumerate(records):
        record_id = record["state_id"]
        statements = record["statements"]
        base_vectors = statement_embeddings[record_index]
        intervention = record["interventions"]["uniform_candidate_sample"]
        edit = intervention["edit"]
        pre_state_vector = normalize(base_vectors.mean(axis=0, keepdims=True))[0]
        if edit["operation"] == "remove":
            keep = [i for i, statement in enumerate(statements) if statement["id"] != edit["fact_id"]]
            post_statement_vectors = base_vectors[keep]
        else:
            post_statement_vectors = np.vstack((base_vectors, added_fact_embeddings[record_id]))
        if len(post_statement_vectors) == 0:
            raise ValueError(f"Empty post-edit state for {record_id}")
        post_state_vector = normalize(post_statement_vectors.mean(axis=0, keepdims=True))[0]
        post_statements = post_edit_statements(record)
        if len(post_statements) != len(post_statement_vectors):
            raise ValueError(f"Post-edit statement/vector mismatch for {record_id}")
        original_statement_vectors_by_id[record_id] = base_vectors
        post_texts_by_id[record_id] = [item["text"] for item in post_statements]
        after_fields = {item["field_id"]: item for item in intervention["after_fields"]}
        changed_mask = {item["field_id"]: bool(item["should_change"])
                        for item in intervention["affected_field_mask"]}
        edit_vector = edit_embeddings[record_index]
        for field in record["fields"]:
            field_id = field["field_id"]
            after = after_fields[field_id]
            query_vector = query_embeddings[cursor]
            scores = post_statement_vectors @ query_vector
            ranking_indices = np.argsort(-scores, kind="stable")
            examples.append({
                "state_id": record_id,
                "composition_group": record["composition_group"],
                "split": _split_group(record["composition_group"]),
                "field_id": field_id,
                "label": after["label"],
                "before_label": field["label"],
                "changed": changed_mask[field_id],
                "hypothesis": f"{field['target']['entity']} is {field['target']['predicate']}.",
                "gold_evidence": set(after["proof_evidence"]),
                "ranked_ids": [post_statements[int(i)]["id"] for i in ranking_indices],
                "ranked_texts": [post_statements[int(i)]["text"] for i in ranking_indices],
                "bundles": {q: record["question_bundles"][str(q)] for q in (1, 5, 20)},
            })
            pre_vectors.append(pre_state_vector)
            post_vectors.append(post_state_vector)
            queries.append(query_vector)
            edits.append(edit_vector)
            cursor += 1
    return (examples, np.asarray(pre_vectors, dtype=np.float32),
            np.asarray(post_vectors, dtype=np.float32), np.asarray(queries, dtype=np.float32),
            np.asarray(edits, dtype=np.float32), original_statement_vectors_by_id, post_texts_by_id)


def route_tie_key(example: dict) -> str:
    return hashlib.sha256((example["state_id"] + example["field_id"]).encode()).hexdigest()


def deterministic_random_score(example: dict) -> float:
    key = f"{SEED}|{example['state_id']}|{example['field_id']}|random_control"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / float(16**16 - 1)


def fit_route_cuts(calibration_examples: list[dict], scores: np.ndarray) -> dict:
    index_by_key = {(item["state_id"], item["field_id"]): i
                    for i, item in enumerate(calibration_examples)}
    cuts = {}
    for q in (1, 5, 20):
        rows = [item for item in calibration_examples if item["field_id"] in item["bundles"][q]]
        row_indices = [index_by_key[(item["state_id"], item["field_id"])] for item in rows]
        order = sorted(row_indices, key=lambda i: (-float(scores[i]), route_tie_key(calibration_examples[i])))
        if not order:
            raise ValueError(f"No calibration examples for Q={q}")
        cuts[str(q)] = {}
        for rate in GATE_RATES:
            count = max(1, int(math.ceil(rate * len(order))))
            threshold_i = order[count - 1]
            cuts[str(q)][str(rate)] = {
                "score": float(scores[threshold_i]),
                "tie_key": route_tie_key(calibration_examples[threshold_i]),
                "calibration_route_fraction": count / len(order),
                "calibration_examples": len(order),
            }
    return cuts


def route_by_rank(examples: list[dict], scores: np.ndarray, cuts: dict,
                  q: int, rate: float) -> np.ndarray:
    threshold = cuts[str(q)][str(rate)]
    boundary = (-threshold["score"], threshold["tie_key"])
    route = np.zeros(len(examples), dtype=bool)
    for i, item in enumerate(examples):
        route[i] = (-float(scores[i]), route_tie_key(item)) <= boundary
    return route


def metrics(labels: list[str], probs: np.ndarray, state_ids: list[str],
            route_mask: np.ndarray | None = None) -> dict:
    predictions = labels_from_probs(probs)
    per_class = {}
    f1s = []
    for label in LABELS:
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        support = sum(y == label for y in labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    exact_by_state: dict[str, list[bool]] = defaultdict(list)
    for state_id, gold, pred in zip(state_ids, labels, predictions):
        exact_by_state[state_id].append(gold == pred)
    indices = np.asarray([LABELS.index(label) for label in labels], dtype=int)
    one_hot = np.eye(len(LABELS))[indices]
    out = {
        "n": len(labels),
        "accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))) if labels else None,
        "macro_f1": float(np.mean(f1s)) if labels else None,
        "multiclass_brier_sum": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))) if labels else None,
        "bundle_exact_accuracy": float(np.mean([all(x) for x in exact_by_state.values()])) if exact_by_state else None,
        "bundle_any_error_rate": float(np.mean([not all(x) for x in exact_by_state.values()])) if exact_by_state else None,
        "per_class": per_class,
    }
    if route_mask is not None:
        out["route_fraction"] = float(np.mean(route_mask)) if len(route_mask) else 0.0
    return out


def selection_metrics(examples: list[dict], scores: np.ndarray,
                     cuts: dict, q: int, rate: float) -> dict:
    route = route_by_rank(examples, scores, cuts, q, rate)
    changed = np.asarray([item["changed"] for item in examples], dtype=bool)
    selected_changed = int(np.sum(route & changed))
    total_changed = int(np.sum(changed))
    selected = int(np.sum(route))
    labels = changed.astype(int)
    ap = (float(average_precision_score(labels, scores))
          if np.any(labels) and np.any(labels == 0) else None)
    return {
        "route_fraction": float(np.mean(route)) if len(route) else 0.0,
        "selected_fields": selected,
        "changed_fields": total_changed,
        "changed_rule_groups": len({item["composition_group"] for item in examples if item["changed"]}),
        "selected_changed_fields": selected_changed,
        "change_recall": selected_changed / total_changed if total_changed else None,
        "change_precision": selected_changed / selected if selected else None,
        "average_precision": ap,
    }


def cluster_bootstrap_recall_delta(examples: list[dict], route_a: np.ndarray,
                                  route_b: np.ndarray, seed: int = SEED,
                                  replicates: int = 2000) -> dict:
    """Paired bootstrap of changed-field recall, resampling rule groups as units."""
    groups = sorted({item["composition_group"] for item in examples})
    counts = {group: [0, 0, 0] for group in groups}
    for i, item in enumerate(examples):
        if item["changed"]:
            row = counts[item["composition_group"]]
            row[0] += 1
            row[1] += int(route_a[i])
            row[2] += int(route_b[i])
    total_changed = sum(value[0] for value in counts.values())
    if not total_changed:
        return {"changed_fields": 0, "groups": len(groups), "delta_recall_a_minus_b": None,
                "cluster_bootstrap_95pct_interval": None}
    observed_a = sum(value[1] for value in counts.values()) / total_changed
    observed_b = sum(value[2] for value in counts.values()) / total_changed
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        positives = sum(counts[group][0] for group in selected)
        if positives:
            hit_a = sum(counts[group][1] for group in selected)
            hit_b = sum(counts[group][2] for group in selected)
            samples.append(hit_a / positives - hit_b / positives)
    interval = np.quantile(np.asarray(samples), [0.025, 0.975]).tolist() if samples else None
    return {
        "changed_fields": total_changed,
        "groups": len(groups),
        "delta_recall_a_minus_b": observed_a - observed_b,
        "cluster_bootstrap_95pct_interval": interval,
        "bootstrap_replicates": len(samples),
        "unit": "rule-composition group",
    }


def cluster_bootstrap_accuracy_delta(examples: list[dict], labels: list[str],
                                     probs_a: np.ndarray, probs_b: np.ndarray,
                                     seed: int = SEED, replicates: int = 2000) -> dict:
    """Paired bootstrap of accuracy difference, resampling rule groups as units."""
    pred_a, pred_b = labels_from_probs(probs_a), labels_from_probs(probs_b)
    groups = sorted({item["composition_group"] for item in examples})
    counts = {group: [0, 0, 0] for group in groups}
    for item, gold, a, b in zip(examples, labels, pred_a, pred_b):
        row = counts[item["composition_group"]]
        row[0] += 1
        row[1] += int(gold == a)
        row[2] += int(gold == b)
    n = sum(value[0] for value in counts.values())
    correct_a = sum(value[1] for value in counts.values())
    correct_b = sum(value[2] for value in counts.values())
    observed = (correct_a - correct_b) / n if n else None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        total_n = sum(counts[group][0] for group in selected)
        if total_n:
            hit_a = sum(counts[group][1] for group in selected)
            hit_b = sum(counts[group][2] for group in selected)
            samples.append((hit_a - hit_b) / total_n)
    interval = np.quantile(np.asarray(samples), [0.025, 0.975]).tolist() if samples else None
    return {
        "delta_accuracy_a_minus_b": observed,
        "cluster_bootstrap_95pct_interval": interval,
        "bootstrap_replicates": len(samples),
        "groups": len(groups),
        "unit": "rule-composition group",
    }


def evidence_metrics(examples: list[dict], q: int) -> dict:
    result = {}
    for k in TOP_K:
        for label in ("true", "false", "all_supported"):
            subset = [e for e in examples if e["field_id"] in e["bundles"][q]
                      and e["label"] in ("true", "false")
                      and (label == "all_supported" or e["label"] == label)]
            any_hit = [bool(e["gold_evidence"] & set(e["ranked_ids"][:k])) for e in subset]
            all_hit = [e["gold_evidence"] <= set(e["ranked_ids"][:k]) for e in subset]
            result[f"{label}@{k}"] = {
                "n": len(subset),
                "any_gold": float(np.mean(any_hit)) if subset else None,
                "all_gold": float(np.mean(all_hit)) if subset else None,
            }
    return result


def score_subset_nli(examples: list[dict], post_texts_by_id: dict[str, list[str]],
                     tokenizer, model, device: str) -> tuple[np.ndarray, np.ndarray, dict]:
    full_premises, topk_premises, hypotheses = [], [], []
    for item in examples:
        full_premises.append(" ".join(post_texts_by_id[item["state_id"]]))
        topk_premises.append(" ".join(item["ranked_texts"][:5]))
        hypotheses.append(item["hypothesis"])
    full_probs, full_truncated, full_seconds = score_nli(full_premises, hypotheses, tokenizer, model, device)
    topk_probs, topk_truncated, topk_seconds = score_nli(topk_premises, hypotheses, tokenizer, model, device)
    return full_probs, topk_probs, {
        "pairs_per_route": len(examples),
        "direct_full_post_edit_truncated_pairs": full_truncated,
        "top5_post_edit_truncated_pairs": topk_truncated,
        "direct_full_batch_seconds": full_seconds,
        "top5_batch_seconds": topk_seconds,
    }


def heldout_report(examples: list[dict], old_probs: np.ndarray, post_probs: np.ndarray,
                   confidence_scores: np.ndarray, risk_scores: np.ndarray,
                   full_probs: np.ndarray, top5_probs: np.ndarray,
                   confidence_cuts: dict, risk_cuts: dict,
                   similarity_scores: np.ndarray, similarity_cuts: dict,
                   random_scores: np.ndarray, random_cuts: dict) -> dict:
    report = {"evidence": {}, "selection": {}, "update_quality": {}, "paired_accuracy_deltas": {}}
    for q in (1, 5, 20):
        idx = [i for i, item in enumerate(examples) if item["field_id"] in item["bundles"][q]]
        subset = [examples[i] for i in idx]
        labels = [item["label"] for item in subset]
        state_ids = [item["state_id"] for item in subset]
        changed = np.asarray([item["changed"] for item in subset], dtype=bool)
        old, post = old_probs[idx], post_probs[idx]
        full, top5 = full_probs[idx], top5_probs[idx]
        conf_score, risk_score = confidence_scores[idx], risk_scores[idx]
        similarity_score = similarity_scores[idx]
        random_score = random_scores[idx]
        report["evidence"][f"q{q}"] = evidence_metrics(subset, q)
        report["selection"][f"q{q}"] = {
            "uniform_edit_changed_field_fraction": float(np.mean(changed)) if len(changed) else None,
            "changed_fields": int(np.sum(changed)),
            "confidence_only": {str(rate): selection_metrics(subset, conf_score, confidence_cuts, q, rate)
                                for rate in GATE_RATES},
            "intervention_risk": {str(rate): selection_metrics(subset, risk_score, risk_cuts, q, rate)
                                  for rate in GATE_RATES},
            "edit_query_cosine_baseline": {str(rate): selection_metrics(subset, similarity_score, similarity_cuts, q, rate)
                                            for rate in GATE_RATES},
            "deterministic_random_reference": {str(rate): selection_metrics(subset, random_score, random_cuts, q, rate)
                                               for rate in GATE_RATES},
        }
        policies = {
            "cached_pre_edit_prediction": (old, None),
            "recompute_shared_head_after_edit": (post, None),
            "direct_full_state_nli_after_edit": (full, None),
            "top5_nli_after_edit_all_fields": (top5, np.ones(len(idx), dtype=bool)),
        }
        exact = np.eye(len(LABELS))[[LABELS.index(label) for label in labels]]
        every_changed = changed.copy()
        policies["oracle_change_mask_exact_update_non_deployable"] = (
            np.where(every_changed[:, None], exact, old), every_changed)
        for route_name, score, cuts in (("confidence_only", conf_score, confidence_cuts),
                                        ("intervention_risk", risk_score, risk_cuts),
                                        ("edit_query_cosine", similarity_score, similarity_cuts),
                                        ("deterministic_random", random_score, random_cuts)):
            for rate in GATE_RATES:
                route = route_by_rank(subset, score, cuts, q, rate)
                combined = np.where(route[:, None], top5, old)
                policies[f"{route_name}_gate_{int(rate * 100)}pct"] = (combined, route)
                budget = int(math.ceil(rate * len(idx)))
                changed_order = [i for i in range(len(idx)) if changed[i]]
                oracle_route = np.zeros(len(idx), dtype=bool)
                oracle_route[changed_order[:budget]] = True
                oracle = np.where(oracle_route[:, None], exact, old)
                policies[f"oracle_change_first_{int(rate * 100)}pct_exact_non_deployable"] = (oracle, oracle_route)
        report["update_quality"][f"q{q}"] = {
            name: metrics(labels, probs, state_ids, route)
            for name, (probs, route) in policies.items()
        }
        if q == 20:
            confidence_route = route_by_rank(subset, conf_score, confidence_cuts, q, 0.25)
            risk_route = route_by_rank(subset, risk_score, risk_cuts, q, 0.25)
            similarity_route = route_by_rank(subset, similarity_score, similarity_cuts, q, 0.25)
            random_route = route_by_rank(subset, random_score, random_cuts, q, 0.25)
            report["selection"]["q20"]["cluster_bootstrap_recall_comparisons_at_25pct"] = {
                "intervention_risk_minus_confidence": cluster_bootstrap_recall_delta(
                    subset, risk_route, confidence_route),
                "intervention_risk_minus_edit_query_cosine": cluster_bootstrap_recall_delta(
                    subset, risk_route, similarity_route),
                "intervention_risk_minus_random": cluster_bootstrap_recall_delta(
                    subset, risk_route, random_route),
            }
            risk_update = np.where(risk_route[:, None], top5, old)
            report["paired_accuracy_deltas"]["q20_risk_gate25_minus_cached"] = \
                cluster_bootstrap_accuracy_delta(subset, labels, risk_update, old)
    return report


def _cpu_profile(records: list[dict], examples: list[dict],
                 statement_vectors_by_id: dict[str, np.ndarray],
                 pre_state_by_id: dict[str, np.ndarray], query_by_key: dict[tuple[str, str], np.ndarray],
                 old_probs_by_key: dict[tuple[str, str], np.ndarray],
                 linear, risk_model, sent_tok, sent_model, nli_tok, nli_model,
                 confidence_cuts: dict, risk_cuts: dict, similarity_cuts: dict,
                 sample_states: int = 24) -> dict:
    """Request profile assuming the previous state and question slate are cached."""
    sent_model.to("cpu").eval()
    nli_model.to("cpu").eval()
    torch.cuda.empty_cache()
    torch.set_num_threads(2)
    record_map = {record["state_id"]: record for record in records}
    test_ids = sorted({e["state_id"] for e in examples if e["split"] == "heldout_composition"})
    if len(test_ids) > sample_states:
        positions = np.linspace(0, len(test_ids) - 1, sample_states).round().astype(int)
        selected_states = [test_ids[int(i)] for i in positions]
    else:
        selected_states = test_ids
    times = {route: {str(q): [] for q in (1, 5, 20)}
             for route in ("direct_full", "confidence_gate25", "edit_similarity_gate25",
                           "intervention_risk_gate25")}
    route_counts = {route: {str(q): [] for q in (1, 5, 20)}
                    for route in ("confidence_gate25", "edit_similarity_gate25",
                                  "intervention_risk_gate25")}
    with torch.inference_mode():
        for state_id in selected_states:
            record = record_map[state_id]
            edit = record["interventions"]["uniform_candidate_sample"]["edit"]
            fields_by_id = {field["field_id"]: field for field in record["fields"]}
            original_vectors = statement_vectors_by_id[state_id]
            for q in (1, 5, 20):
                field_ids = record["question_bundles"][str(q)]
                fields = [fields_by_id[field_id] for field_id in field_ids]
                hypotheses = [f"{field['target']['entity']} is {field['target']['predicate']}." for field in fields]
                keys = [(state_id, field_id) for field_id in field_ids]
                q_vectors = np.stack([query_by_key[key] for key in keys])
                old = np.stack([old_probs_by_key[key] for key in keys])
                # Same edited state, same question set, full long-premise NLI reference.
                start = time.perf_counter()
                post_statements = post_edit_statements(record)
                premise = " ".join(item["text"] for item in post_statements)
                score_nli([premise] * q, hypotheses, nli_tok, nli_model, "cpu")
                times["direct_full"][str(q)].append(time.perf_counter() - start)
                for route_name, cuts, score_kind in (
                    ("confidence_gate25", confidence_cuts, "confidence"),
                    ("edit_similarity_gate25", similarity_cuts, "similarity"),
                    ("intervention_risk_gate25", risk_cuts, "risk"),
                ):
                    start = time.perf_counter()
                    post_statements = post_edit_statements(record)
                    if score_kind in ("risk", "similarity"):
                        texts_to_encode = [edit_text(edit)]
                        if edit["operation"] == "add":
                            texts_to_encode.append(edit["text"])
                        encoded = encode_texts(texts_to_encode, sent_tok, sent_model, "cpu")
                        edit_vectors = encoded[:1]
                        if score_kind == "risk":
                            ev = np.repeat(edit_vectors, q, axis=0)
                            pre = np.repeat(pre_state_by_id[state_id][None, :], q, axis=0)
                            risk_x = risk_feature_rows(pre, q_vectors, ev)
                            classes = list(risk_model.classes_)
                            scores = risk_model.predict_proba(risk_x)[:, classes.index(1)]
                        else:
                            scores = np.sum(np.repeat(edit_vectors, q, axis=0) * q_vectors, axis=1)
                        added = encoded[1:] if edit["operation"] == "add" else None
                    else:
                        scores = 1.0 - old.max(axis=1)
                        added = (encode_texts([edit["text"]], sent_tok, sent_model, "cpu")
                                 if edit["operation"] == "add" else None)
                    examples_q = [{"state_id": state_id, "field_id": field_id} for field_id in field_ids]
                    # Routing ties use the same stable identity key as calibration.
                    route = route_by_rank(examples_q, scores, cuts, q, 0.25)
                    if edit["operation"] == "add":
                        post_vectors = np.vstack((original_vectors, added))
                    else:
                        keep = [i for i, statement in enumerate(record["statements"])
                                if statement["id"] != edit["fact_id"]]
                        post_vectors = original_vectors[keep]
                    if np.any(route):
                        rankings = np.argsort(-(q_vectors[route] @ post_vectors.T), axis=1, kind="stable")
                        selected_texts = [[post_statements[int(i)]["text"]
                                           for i in ranking[:5]] for ranking in rankings]
                        premises = [" ".join(texts) for texts in selected_texts]
                        score_nli(premises, [hypotheses[i] for i in np.flatnonzero(route)],
                                  nli_tok, nli_model, "cpu")
                    times[route_name][str(q)].append(time.perf_counter() - start)
                    route_counts[route_name][str(q)].append(float(np.mean(route)))

    def summarize_times(values: list[float]) -> dict:
        ms = np.asarray(values) * 1000.0
        return {"n_requests": len(values), "p50_ms": float(np.quantile(ms, 0.50)),
                "p95_ms": float(np.quantile(ms, 0.95)), "max_ms": float(ms.max())}

    cpu_name = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "machine": "Kaggle CPU reference; not user's target device",
        "cpu_model": cpu_name,
        "torch_threads": torch.get_num_threads(),
        "sample_states": len(selected_states),
        "request_latency": {route: {q: summarize_times(values) for q, values in by_q.items()}
                            for route, by_q in times.items()},
        "actual_route_fraction": {
            route: {q: float(np.mean(values)) if values else 0.0 for q, values in by_q.items()}
            for route, by_q in route_counts.items()},
        "whole_process_peak_rss_mib": float(peak_rss_mb),
        "assumptions_and_limits": [
            "The prior state, per-statement embeddings, previous field outputs, and question embeddings are cached.",
            "A repeated question slate is required to reuse prior field outputs; newly arriving questions are outside this update path.",
            "Includes edit/query routing and selected verification; excludes model download/load and initial state encoding.",
            "Only 24 held-out state requests per Q stratum; p95 is descriptive and unstable.",
            "Kaggle CPU is a reference host, not the user's target device or a universal CPU result.",
        ],
    }


def main() -> None:
    global torch, transformers, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Expected Kaggle T4 GPU is unavailable")
    torch.cuda.init()
    torch.cuda.set_device(0)
    warmup = torch.empty(1, device="cuda:0")
    del warmup
    torch.cuda.synchronize(0)
    print(f"CUDA preflight passed: {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    records = [generate_state(i, SEED, composition_group_size=COMPOSITION_GROUP_SIZE)
               for i in range(N_STATES)]
    serialized = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode()
    generator_path = Path(__import__("intervention_probe").__file__)
    generator_sha = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    records_sha = hashlib.sha256(serialized).hexdigest()
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record["composition_group"]].add(_split_group(record["composition_group"]))
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("A rule-composition group crossed split boundaries")

    sent_tok = AutoTokenizer.from_pretrained(SENTENCE_MODEL, revision=SENTENCE_REVISION,
                                              use_fast=True, trust_remote_code=False)
    sent_model = AutoModel.from_pretrained(SENTENCE_MODEL, revision=SENTENCE_REVISION,
                                           use_safetensors=True, trust_remote_code=False).to("cuda").eval()
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION,
                                             use_fast=True, trust_remote_code=False)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, revision=NLI_REVISION, use_safetensors=True, trust_remote_code=False).to("cuda").eval()
    for model in (sent_model, nli_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    mapping = {int(k): str(v).lower() for k, v in nli_model.config.id2label.items()}
    if mapping != {0: "contradiction", 1: "entailment", 2: "neutral"}:
        raise ValueError(f"Unexpected NLI label mapping: {mapping}")
    if any(parameter.requires_grad for parameter in sent_model.parameters()) or any(
            parameter.requires_grad for parameter in nli_model.parameters()):
        raise ValueError("Pretrained backbones must remain frozen")

    statement_texts, hypothesis_texts, edit_texts, add_texts, state_ranges = [], [], [], [], []
    add_state_ids = []
    for record in records:
        start = len(statement_texts)
        statement_texts.extend(statement["text"] for statement in record["statements"])
        state_ranges.append((start, len(statement_texts)))
        for field in record["fields"]:
            target = field["target"]
            hypothesis_texts.append(f"{target['entity']} is {target['predicate']}.")
        edit = record["interventions"]["uniform_candidate_sample"]["edit"]
        edit_texts.append(edit_text(edit))
        if edit["operation"] == "add":
            add_texts.append(edit["text"])
            add_state_ids.append(record["state_id"])
    encode_started = time.monotonic()
    all_statement_vectors = encode_texts(statement_texts, sent_tok, sent_model, "cuda")
    query_embeddings = encode_texts(hypothesis_texts, sent_tok, sent_model, "cuda")
    edit_embeddings = encode_texts(edit_texts, sent_tok, sent_model, "cuda")
    added_rows = encode_texts(add_texts, sent_tok, sent_model, "cuda")
    added_fact_embeddings = {state_id: added_rows[i] for i, state_id in enumerate(add_state_ids)}
    encode_seconds = time.monotonic() - encode_started
    statement_embeddings = [all_statement_vectors[a:b] for a, b in state_ranges]
    (examples, pre_vectors, post_vectors, query_vectors, edit_vectors,
     statement_vectors_by_id, post_texts_by_id) = build_transition_examples(
        records, statement_embeddings, query_embeddings, edit_embeddings, added_fact_embeddings)

    pre_labels = [item["before_label"] for item in examples]
    post_labels = [item["label"] for item in examples]
    changed = np.asarray([item["changed"] for item in examples], dtype=int)
    pre_x = feature_rows(pre_vectors, query_vectors)
    post_x = feature_rows(post_vectors, query_vectors)
    risk_x = risk_feature_rows(pre_vectors, query_vectors, edit_vectors)
    fit_idx = [i for i, item in enumerate(examples) if item["split"] == "fit_composition"]
    cal_idx = [i for i, item in enumerate(examples) if item["split"] == "calibration_composition"]
    test_idx = [i for i, item in enumerate(examples) if item["split"] == "heldout_composition"]
    unique_groups = {split: {examples[i]["composition_group"] for i in indices}
                     for split, indices in (("fit", fit_idx), ("calibration", cal_idx), ("heldout", test_idx))}
    if min(len(unique_groups[name]) for name in unique_groups) < 2:
        raise ValueError("Too few independent rule groups in fit/calibration/heldout split")
    if len(set(changed[fit_idx])) < 2:
        raise ValueError("Intervention-risk fit groups do not contain both unchanged and changed fields")

    linear = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=SEED)
    linear.fit(pre_x[fit_idx], np.asarray(pre_labels)[fit_idx])
    if int(linear.n_iter_[0]) >= 500 or set(linear.classes_) != set(LABELS):
        raise RuntimeError("Shared readout did not converge or lacks a ternary class")
    class_order = list(linear.classes_)
    old_raw = linear.predict_proba(pre_x)
    post_raw = linear.predict_proba(post_x)
    old_probs = old_raw[:, [class_order.index(label) for label in LABELS]]
    post_probs = post_raw[:, [class_order.index(label) for label in LABELS]]

    risk_model = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs",
                                    class_weight="balanced", random_state=SEED)
    risk_model.fit(risk_x[fit_idx], changed[fit_idx])
    if int(risk_model.n_iter_[0]) >= 500 or set(risk_model.classes_) != {0, 1}:
        raise RuntimeError("Intervention-risk readout did not converge or lacks both mask classes")
    risk_order = list(risk_model.classes_)
    risk_scores = risk_model.predict_proba(risk_x)[:, risk_order.index(1)]
    confidence_scores = 1.0 - old_probs.max(axis=1)
    similarity_scores = np.sum(edit_vectors * query_vectors, axis=1)
    random_scores = np.asarray([deterministic_random_score(item) for item in examples], dtype=np.float64)
    calibration_examples = [examples[i] for i in cal_idx]
    confidence_cuts = fit_route_cuts(calibration_examples, confidence_scores[cal_idx])
    risk_cuts = fit_route_cuts(calibration_examples, risk_scores[cal_idx])
    similarity_cuts = fit_route_cuts(calibration_examples, similarity_scores[cal_idx])
    random_cuts = fit_route_cuts(calibration_examples, random_scores[cal_idx])

    heldout_examples = [examples[i] for i in test_idx]
    full_probs, top5_probs, nli_timing = score_subset_nli(
        heldout_examples, post_texts_by_id, nli_tok, nli_model, "cuda")
    quality = heldout_report(heldout_examples, old_probs[test_idx], post_probs[test_idx],
                             confidence_scores[test_idx], risk_scores[test_idx],
                             full_probs, top5_probs, confidence_cuts, risk_cuts,
                             similarity_scores[test_idx], similarity_cuts,
                             random_scores[test_idx], random_cuts)
    torch.cuda.synchronize()
    peak_cuda = int(torch.cuda.max_memory_allocated(0))
    # Build lookup tables for the CPU profile from the already prepared rows.
    query_by_key = {(item["state_id"], item["field_id"]): query_vectors[i]
                    for i, item in enumerate(examples)}
    old_probs_by_key = {(item["state_id"], item["field_id"]): old_probs[i]
                        for i, item in enumerate(examples)}
    cpu_profile = _cpu_profile(
        records, heldout_examples, statement_vectors_by_id,
        {record["state_id"]: normalize(statement_embeddings[i].mean(axis=0, keepdims=True))[0]
         for i, record in enumerate(records)}, query_by_key, old_probs_by_key,
        linear, risk_model, sent_tok, sent_model, nli_tok, nli_model,
        confidence_cuts, risk_cuts, similarity_cuts)

    split_counts = Counter(_split_group(record["composition_group"]) for record in records)
    heldout_positive_states = len({item["state_id"] for item in heldout_examples if item["changed"]})
    report = {
        "kind": "frozen_synthetic_intervention_risk_vs_confidence_screen_not_benchmark_or_model_win",
        "seed": SEED,
        "states": N_STATES,
        "composition_group_size": COMPOSITION_GROUP_SIZE,
        "composition_groups": len(groups),
        "groups_by_split": {name: len(values) for name, values in unique_groups.items()},
        "states_by_split": dict(split_counts),
        "heldout_states_with_any_changed_field": heldout_positive_states,
        "generator_sha256": generator_sha,
        "generated_records_sha256": records_sha,
        "sentence_model": SENTENCE_MODEL,
        "sentence_revision": SENTENCE_REVISION,
        "nli_model": NLI_MODEL,
        "nli_revision": NLI_REVISION,
        "nli_label_mapping": mapping,
        "shared_readout": "logistic regression on frozen mean state embedding, query embedding, absolute difference, and elementwise product; fit groups only",
        "intervention_risk_readout": "class-balanced logistic ranker on frozen pre-edit state, query, operation-prefixed edit embedding, edit-query absolute difference, and edit-query product; fit groups only; scores rank fields and are not claimed calibrated probabilities",
        "edit_similarity_baseline": "cosine similarity between frozen query and operation-prefixed edit embeddings; thresholded on calibration groups",
        "readout_iterations": {"shared": int(linear.n_iter_[0]), "intervention_risk": int(risk_model.n_iter_[0])},
        "confidence_route_cuts_from_calibration_groups": confidence_cuts,
        "intervention_risk_route_cuts_from_calibration_groups": risk_cuts,
        "edit_query_cosine_route_cuts_from_calibration_groups": similarity_cuts,
        "deterministic_random_route_cuts_from_calibration_groups": random_cuts,
        "edit_sample": "one uniformly sampled add/remove edit per generated state; no-effect edits retained; effective-edit positive-control sample excluded from fitting and prevalence metrics",
        "route_budgets": list(GATE_RATES),
        "sentence_max_tokens": SENTENCE_MAX_TOKENS,
        "nli_max_tokens": NLI_MAX_TOKENS,
        "nli_heldout_timing": nli_timing,
        "frozen_sentence_encoding_seconds_gpu": encode_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "peak_cuda_allocated_bytes": peak_cuda,
        "heldout_results": quality,
        "kaggle_cpu_reference_profile": cpu_profile,
        "total_kernel_seconds": time.monotonic() - started,
        "test_set_examined": "held-out rule-composition groups only; synthetic and diagnostic",
        "limitations": [
            "Synthetic templated logic is not natural-language transfer or a published benchmark.",
            "The intervention-risk target is whether a ternary answer changes after one synthetic fact edit; it may not transfer to real edits or multi-edit updates.",
            "The same prior question slate and its outputs must be cached; new questions cannot use the cached-answer update path.",
            "Risk ranking is learned and thresholded on synthetic groups; its scores are not calibrated probabilities.",
            "The NLI verifier was pretrained on natural-language inference, not this exact rule grammar; verifier errors can limit update quality.",
            "Exact-answer oracle policies are non-deployable upper bounds. Stored proof metrics use one deterministic proof per entailed field.",
            "Kaggle GPU/CPU timing and memory do not establish the user's target-device speed or 8 GiB/4 GiB profile.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"kind": report["kind"], "groups_by_split": report["groups_by_split"],
                      "heldout_states_with_change": heldout_positive_states,
                      "q20_primary_gate_comparison": quality["selection"]["q20"]["cluster_bootstrap_recall_comparisons_at_25pct"],
                      "q20_selection": quality["selection"]["q20"],
                      "q20_update": quality["update_quality"]["q20"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
