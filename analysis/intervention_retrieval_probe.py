#!/usr/bin/env python3
"""Measure a cheap lexical evidence-retrieval floor on generated probe states.

This is a synthetic retrieval diagnostic. It does not evaluate a decision model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


TOP_K = (1, 3, 5, 10)
QUESTION_COUNTS = (1, 5, 20)


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _records_by_split(records: list[dict]) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_split[record["composition_split"]].append(record)
    return {"all": records, **by_split}


def score_split(records: list[dict], vectorizer: TfidfVectorizer) -> dict:
    rows = []
    for record in records:
        statements = record["statements"]
        statement_ids = [statement["id"] for statement in statements]
        statement_texts = [statement["text"] for statement in statements]
        matrix = vectorizer.transform(statement_texts)
        fields = {field["field_id"]: field for field in record["fields"]}
        for q in QUESTION_COUNTS:
            for field_id in record["question_bundles"][str(q)]:
                field = fields[field_id]
                # Declarative form is used as a retrieval query; the original
                # question and all evidence text remain unchanged in the data.
                target = field["target"]
                query = f"{target['entity']} is {target['predicate']}."
                query_vector = vectorizer.transform([query])
                scores = np.asarray((matrix @ query_vector.T).todense()).reshape(-1)
                ranking = np.argsort(-scores, kind="stable")
                gold = set(field["proof_evidence"])
                rows.append({
                    "composition_group": record["composition_group"],
                    "composition_split": record["composition_split"],
                    "q": q,
                    "label": field["label"],
                    "gold": gold,
                    "ranking": [statement_ids[int(i)] for i in ranking],
                })
    result = {}
    for q in QUESTION_COUNTS:
        for label in ("true", "false", "unknown", "all_labels"):
            subset = [row for row in rows if row["q"] == q and (label == "all_labels" or row["label"] == label)]
            positive = [row for row in subset if row["label"] in ("true", "false") and row["gold"]]
            metrics = {"examples": len(subset), "proof_supported_examples": len(positive)}
            for k in TOP_K:
                any_count = sum(bool(row["gold"] & set(row["ranking"][:k])) for row in positive)
                all_count = sum(row["gold"] <= set(row["ranking"][:k]) for row in positive)
                metrics[f"any_gold_at_{k}"] = any_count / len(positive) if positive else None
                metrics[f"all_gold_at_{k}"] = all_count / len(positive) if positive else None
            result[f"q{q}/{label}"] = metrics
    return result


def run(input_path: Path, output_path: Path) -> dict:
    records = load_records(input_path)
    statement_texts = [
        statement["text"] for record in records
        if record["composition_split"] == "seen_composition"
        for statement in record["statements"]
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm="l2")
    vectorizer.fit(statement_texts)
    splits = _records_by_split(records)
    report = {
        "kind": "synthetic_tfidf_evidence_retrieval_floor_not_decision_model_result",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "retrieval_fit_composition_split": "seen_composition_only",
        "states": len(records),
        "composition_groups": len({record["composition_group"] for record in records}),
        "states_by_composition_split": {
            split: len(items) for split, items in splits.items() if split != "all"
        },
        "retriever": "scikit-learn TfidfVectorizer(word unigrams+bigrams, sublinear_tf, l2 normalization)",
        "scikit_learn_version": sklearn.__version__,
        "metrics_by_composition_split": {
            split: score_split(items, vectorizer) for split, items in splits.items()
        },
        "limits": [
            "Synthetic templated rules only; this is not natural-language transfer evidence.",
            "Only one recorded proof path per entailed field is scored.",
            "Any/all evidence recall is not answer accuracy or calibrated risk.",
            "No neural backbone, verifier, router, CPU serving benchmark, or training is included.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/data/intervention-probe/intervention-bundles.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("experiments/data/2026-09-27-intervention-retrieval-floor.json"))
    args = parser.parse_args()
    report = run(args.input, args.output)
    heldout = report["metrics_by_composition_split"].get("heldout_composition", {})
    print(json.dumps({"kind": report["kind"], "states": report["states"], "heldout_q20": heldout.get("q20/all_labels")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
