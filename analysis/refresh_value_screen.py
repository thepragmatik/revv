#!/usr/bin/env python3
"""Screen cheap selectors for stale-field errors on controlled paired edits.

This is a model-free symbolic mechanism test. A one-round reasoner plays the
limited cached predictor; full closure is an optimistic exact refresher. It
does not measure natural-language quality, CPU latency, or memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .intervention_probe import (
        ENTITIES,
        Fact,
        Literal,
        Proof,
        Rule,
        _derive,
        _label,
        generate_state,
    )
except ImportError:
    from intervention_probe import (
        ENTITIES,
        Fact,
        Literal,
        Proof,
        Rule,
        _derive,
        _label,
        generate_state,
    )


BUNDLE_SIZES = (1, 5, 20)
EDIT_STRATA = ("uniform_candidate_sample", "effective_edit_sample")
POLICY_NAMES = (
    "cheap_flip",
    "unproved_after",
    "edit_query_jaccard",
    "random_expected",
    "oracle_error",
)


def derive_bounded(facts: list[Fact], rules: list[Rule], max_rounds: int) -> dict[Literal, Proof]:
    """Forward-chain for exactly max_rounds simultaneous rule rounds."""
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")
    proofs: dict[Literal, Proof] = {fact.literal: Proof(fact.fact_id) for fact in facts}
    for _ in range(max_rounds):
        additions: dict[Literal, Proof] = {}
        for rule in rules:
            for entity in ENTITIES:
                premises = tuple(Literal(predicate, entity, rule.positive) for predicate in rule.body)
                if all(premise in proofs for premise in premises):
                    head = Literal(rule.head, entity, rule.positive)
                    if head not in proofs:
                        additions.setdefault(head, Proof(rule.rule_id, premises))
        if not additions:
            break
        proofs.update(additions)
    return proofs


def _literal_from_key(value: str) -> Literal:
    entity, polarity, predicate = value.split("|")
    return Literal(predicate, entity, polarity == "+")


def _state_logic(record: dict[str, Any]) -> tuple[list[Fact], list[Rule]]:
    facts: list[Fact] = []
    rules: list[Rule] = []
    for item in record["statements"]:
        if item["kind"] == "fact":
            facts.append(Fact(item["id"], _literal_from_key(item["literal"])))
        elif item["kind"] == "rule":
            rules.append(Rule(
                item["id"],
                tuple(item["body"]),
                item["head"],
                bool(item["positive"]),
            ))
        else:
            raise ValueError(f"Unsupported statement kind: {item['kind']}")
    return facts, rules


def _apply_edit(facts: list[Fact], edit: dict[str, str]) -> list[Fact]:
    if edit["operation"] == "remove":
        updated = [fact for fact in facts if fact.fact_id != edit["fact_id"]]
        if len(updated) == len(facts):
            raise ValueError(f"Remove edit references absent fact {edit['fact_id']}")
        return updated
    if edit["operation"] != "add":
        raise ValueError(f"Unsupported edit operation: {edit['operation']}")
    words = re.findall(r"[a-z]+", edit["text"].lower())
    entity = next((name for name in ENTITIES if name.lower() in words), None)
    if entity is None:
        raise ValueError(f"Cannot parse entity from edit: {edit['text']}")
    predicate = words[-1] if words else ""
    positive = "not" not in words
    if any(fact.fact_id == edit["fact_id"] for fact in facts):
        raise ValueError(f"Add edit duplicates fact ID {edit['fact_id']}")
    return facts + [Fact(edit["fact_id"], Literal(predicate, entity, positive))]


def _field_labels(fields: list[dict[str, Any]], proofs: dict[Literal, Proof]) -> dict[str, str]:
    return {
        field["field_id"]: _label(
            field["target"]["predicate"],
            field["target"]["entity"],
            proofs,
        )[0]
        for field in fields
    }


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token != "is"}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _score_state_edit(
    record: dict[str, Any],
    stratum: str,
    shallow_rounds: int = 1,
) -> list[dict[str, Any]]:
    facts, rules = _state_logic(record)
    before_full = _derive(facts, rules)
    before_shallow = derive_bounded(facts, rules, shallow_rounds)
    if not set(before_shallow).issubset(before_full):
        raise AssertionError("Bounded inference produced a fact outside exact closure")

    before_gold = _field_labels(record["fields"], before_full)
    if before_gold != {field["field_id"]: field["label"] for field in record["fields"]}:
        raise AssertionError("Generated base labels disagree with exact closure")
    stale = _field_labels(record["fields"], before_shallow)
    sample = record["interventions"][stratum]
    edited_facts = _apply_edit(facts, sample["edit"])
    after_full = _derive(edited_facts, rules)
    after_shallow = derive_bounded(edited_facts, rules, shallow_rounds)
    if not set(after_shallow).issubset(after_full):
        raise AssertionError("Edited bounded inference produced a fact outside exact closure")

    after_gold = _field_labels(record["fields"], after_full)
    recorded_after = {field["field_id"]: field["label"] for field in sample["after_fields"]}
    if after_gold != recorded_after:
        raise AssertionError("Reconstructed post-edit labels disagree with generator record")
    masks = {item["field_id"]: item["should_change"] for item in sample["affected_field_mask"]}
    if set(masks) != set(before_gold) or set(after_gold) != set(before_gold):
        raise AssertionError("Field IDs differ across paired worlds")
    if any(masks[field_id] != (before_gold[field_id] != after_gold[field_id]) for field_id in masks):
        raise AssertionError("Recorded affected-field mask disagrees with exact labels")

    edit_text = sample["edit"]["text"]
    edit_id = sample["edit"]["fact_id"]
    rows = []
    for field in record["fields"]:
        field_id = field["field_id"]
        after_cheap = _label(
            field["target"]["predicate"],
            field["target"]["entity"],
            after_shallow,
        )[0]
        stale_label = stale[field_id]
        gold_label = after_gold[field_id]
        rows.append({
            "state_id": record["state_id"],
            "composition_group": record["composition_group"],
            "composition_split": record["composition_split"],
            "stratum": stratum,
            "edit_id": edit_id,
            "edit_operation": sample["edit"]["operation"],
            "edit_text": edit_text,
            "field_id": field_id,
            "query": field["query"],
            "target": field["target"],
            "pre_edit_gold": before_gold[field_id],
            "stale_prediction": stale_label,
            "post_edit_cheap_prediction": after_cheap,
            "post_edit_gold": gold_label,
            "affected": before_gold[field_id] != gold_label,
            "stale_error": stale_label != gold_label,
            "signals": {
                "cheap_flip": int(stale_label != after_cheap),
                "unproved_after": int(after_cheap == "unknown"),
                "edit_query_jaccard": _jaccard(edit_text, field["query"]),
            },
        })
    return rows


def _budgets(q: int) -> list[int]:
    values = (0, (q + 3) // 4, (q + 1) // 2, (3 * q + 3) // 4, q)
    return sorted(set(values))


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_state_stratum: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["state_id"], row["stratum"])
        if row["field_id"] in by_state_stratum[key]:
            raise AssertionError(f"Duplicate field {row['field_id']} in state/edit")
        by_state_stratum[key][row["field_id"]] = row

    for (state_id, stratum), field_map in by_state_stratum.items():
        sample = next(row for row in field_map.values())
        for q in BUNDLE_SIZES:
            selected_ids = sorted(field_map, key=lambda fid: int(fid[1:]))[:q]
            if len(selected_ids) != q:
                raise AssertionError(f"State {state_id} has fewer than {q} fields")
            grouped[(sample["composition_split"], stratum, q, state_id)].extend(
                field_map[field_id] for field_id in selected_ids
            )
    return grouped


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_rows(rows)
    group_splits: dict[str, str] = {}
    for split, _stratum, _q, state_id in grouped:
        row = grouped[(split, _stratum, _q, state_id)][0]
        previous = group_splits.setdefault(row["composition_group"], split)
        if previous != split:
            raise AssertionError("Composition group crosses seen/held-out partitions")

    accum: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    for (split, stratum, q, state_id), bundle in grouped.items():
        base_errors = sum(int(row["stale_error"]) for row in bundle)
        state_group = bundle[0]["composition_group"]
        for budget in _budgets(q):
            random_expected = budget * base_errors / q
            score_orders: dict[str, list[dict[str, Any]]] = {}
            for policy in ("cheap_flip", "unproved_after", "edit_query_jaccard", "oracle_error"):
                if policy == "oracle_error":
                    score = lambda row: int(row["stale_error"])
                else:
                    score = lambda row, p=policy: row["signals"][p]
                score_orders[policy] = sorted(
                    bundle,
                    key=lambda row: (-score(row), row["field_id"]),
                )
            policy_saved = {
                policy: sum(int(row["stale_error"]) for row in order[:budget])
                for policy, order in score_orders.items()
            }
            policy_saved["random_expected"] = random_expected
            for policy, saved in policy_saved.items():
                key = (split, stratum, q, policy, budget)
                item = accum.setdefault(key, {
                    "composition_split": split,
                    "edit_stratum": stratum,
                    "bundle_size": q,
                    "policy": policy,
                    "refresh_budget_per_bundle": budget,
                    "bundles": 0,
                    "fields": 0,
                    "stale_errors": 0,
                    "refresh_calls": 0,
                    "errors_avoided": 0.0,
                    "composition_groups": set(),
                })
                item["bundles"] += 1
                item["fields"] += q
                item["stale_errors"] += base_errors
                item["refresh_calls"] += budget
                item["errors_avoided"] += saved
                item["composition_groups"].add(state_group)

    aggregates = []
    for key in sorted(accum):
        item = accum[key]
        remaining = item["stale_errors"] - item["errors_avoided"]
        avoided = item["errors_avoided"]
        aggregates.append({
            "composition_split": item["composition_split"],
            "edit_stratum": item["edit_stratum"],
            "bundle_size": item["bundle_size"],
            "policy": item["policy"],
            "refresh_budget_per_bundle": item["refresh_budget_per_bundle"],
            "bundles": item["bundles"],
            "composition_groups": len(item["composition_groups"]),
            "fields": item["fields"],
            "refresh_calls": item["refresh_calls"],
            "baseline_stale_errors": item["stale_errors"],
            "errors_avoided": round(avoided, 8),
            "remaining_errors": round(remaining, 8),
            "fraction_of_stale_errors_avoided": (
                round(avoided / item["stale_errors"], 8) if item["stale_errors"] else None
            ),
            "post_policy_error_rate": round(remaining / item["fields"], 8) if item["fields"] else None,
        })
    return {"aggregates": aggregates}


def run_screen(
    seed: int = 20260928,
    states: int = 256,
    composition_group_size: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if states < composition_group_size:
        raise ValueError("states must include at least one full composition group")
    records = [
        generate_state(
            state_index,
            seed,
            max_fields=20,
            composition_group_size=composition_group_size,
        )
        for state_index in range(states)
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        for stratum in EDIT_STRATA:
            rows.extend(_score_state_edit(record, stratum))

    group_to_split: dict[str, str] = {}
    for record in records:
        prior = group_to_split.setdefault(record["composition_group"], record["composition_split"])
        if prior != record["composition_split"]:
            raise AssertionError("One composition graph received multiple split labels")
    eval_result = evaluate_rows(rows)

    config = {
        "seed": seed,
        "states": states,
        "composition_group_size": composition_group_size,
        "composition_groups": len(group_to_split),
        "states_by_composition_split": {
            split: sum(record["composition_split"] == split for record in records)
            for split in ("seen_composition", "heldout_composition")
        },
        "question_bundle_sizes": list(BUNDLE_SIZES),
        "edit_strata": list(EDIT_STRATA),
        "shallow_reasoning_rounds": 1,
        "refresher": "exact full forward-chaining closure (optimistic ceiling)",
        "policies": list(POLICY_NAMES),
        "budgets_by_bundle_size": {str(q): _budgets(q) for q in BUNDLE_SIZES},
        "cost_assumption": "one equal-cost exact refresh call per selected field; no wall-clock claim",
        "diagnostic_only": True,
        "model_free": True,
        "external_data_or_weights": False,
    }
    return {"config": config, **eval_result}, rows


def write_screen(
    output_dir: Path,
    seed: int = 20260928,
    states: int = 256,
    composition_group_size: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result, rows = run_screen(seed, states, composition_group_size)
    raw_path = output_dir / "refresh-value-scores.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    script_path = Path(__file__).resolve()
    generator_path = script_path.with_name("intervention_probe.py")
    result["config"]["screen_script_sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    result["config"]["generator_sha256"] = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    result["config"]["score_rows"] = len(rows)
    result["config"]["score_rows_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    result["config"]["unique_composition_groups"] = len({
        item["composition_group"] for item in rows
    })
    out_path = output_dir / "refresh-value-aggregate.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--composition-group-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/data/refresh-value-screen"),
    )
    args = parser.parse_args()
    result = write_screen(
        args.output_dir,
        seed=args.seed,
        states=args.states,
        composition_group_size=args.composition_group_size,
    )
    print(json.dumps(result["config"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

