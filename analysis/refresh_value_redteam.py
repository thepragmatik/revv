#!/usr/bin/env python3
"""Post-hoc decomposition of base errors and errors newly induced by an edit.

Run this only after the preregistered value-of-refresh screen. The decomposition
is exploratory and does not alter or replace its primary metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .refresh_value_screen import _budgets, run_screen
except ImportError:
    from refresh_value_screen import _budgets, run_screen


def _new_edit_error(row: dict[str, Any]) -> bool:
    return (
        row["stale_prediction"] == row["pre_edit_gold"]
        and row["stale_prediction"] != row["post_edit_gold"]
    )


def _score(row: dict[str, Any], policy: str) -> float:
    if policy == "oracle_error":
        return float(row["stale_error"])
    if policy == "oracle_new_regression":
        return float(_new_edit_error(row))
    if policy == "random_expected":
        raise ValueError("Random is handled analytically")
    return float(row["signals"][policy])


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    states: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        states[(row["composition_split"], row["stratum"], row["state_id"])].append(row)

    decomposition: dict[tuple[str, str, int], dict[str, Any]] = {}
    policy_results: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}

    for (split, stratum, state_id), all_fields in states.items():
        fields_by_index = {
            int(row["field_id"][1:]): row for row in all_fields
        }
        for q in (1, 5, 20):
            bundle = [fields_by_index[i] for i in range(q)]
            dkey = (split, stratum, q)
            d = decomposition.setdefault(dkey, {
                "composition_split": split,
                "edit_stratum": stratum,
                "bundle_size": q,
                "bundles": 0,
                "fields": 0,
                "pre_edit_stale_errors": 0,
                "post_edit_stale_errors": 0,
                "preexisting_errors_persisting": 0,
                "new_edit_induced_errors": 0,
                "preexisting_errors_fixed_by_edit": 0,
                "gold_label_flips": 0,
            })
            d["bundles"] += 1
            d["fields"] += q
            d["pre_edit_stale_errors"] += sum(
                row["stale_prediction"] != row["pre_edit_gold"] for row in bundle
            )
            d["post_edit_stale_errors"] += sum(row["stale_error"] for row in bundle)
            d["preexisting_errors_persisting"] += sum(
                row["stale_prediction"] != row["pre_edit_gold"]
                and row["stale_prediction"] != row["post_edit_gold"]
                for row in bundle
            )
            d["new_edit_induced_errors"] += sum(_new_edit_error(row) for row in bundle)
            d["preexisting_errors_fixed_by_edit"] += sum(
                row["stale_prediction"] != row["pre_edit_gold"]
                and row["stale_prediction"] == row["post_edit_gold"]
                for row in bundle
            )
            d["gold_label_flips"] += sum(
                row["pre_edit_gold"] != row["post_edit_gold"] for row in bundle
            )

            stale_errors = sum(row["stale_error"] for row in bundle)
            new_errors = sum(_new_edit_error(row) for row in bundle)
            methods = ("cheap_flip", "unproved_after", "edit_query_jaccard",
                       "random_expected", "oracle_error", "oracle_new_regression")
            for budget in _budgets(q):
                for policy in methods:
                    if policy == "random_expected":
                        total_saved = budget * stale_errors / q
                        new_saved = budget * new_errors / q
                    else:
                        order = sorted(
                            bundle,
                            key=lambda row: (-_score(row, policy), row["field_id"]),
                        )
                        chosen = order[:budget]
                        total_saved = sum(row["stale_error"] for row in chosen)
                        new_saved = sum(_new_edit_error(row) for row in chosen)

                    pkey = (split, stratum, q, policy, budget)
                    p = policy_results.setdefault(pkey, {
                        "composition_split": split,
                        "edit_stratum": stratum,
                        "bundle_size": q,
                        "policy": policy,
                        "refresh_budget_per_bundle": budget,
                        "bundles": 0,
                        "refresh_calls": 0,
                        "post_edit_errors_avoided": 0.0,
                        "new_edit_errors_avoided": 0.0,
                    })
                    p["bundles"] += 1
                    p["refresh_calls"] += budget
                    p["post_edit_errors_avoided"] += total_saved
                    p["new_edit_errors_avoided"] += new_saved

    out_decomposition = []
    for key in sorted(decomposition):
        out_decomposition.append(decomposition[key])

    out_policy = []
    for key in sorted(policy_results):
        item = policy_results[key]
        d = decomposition[(item["composition_split"], item["edit_stratum"], item["bundle_size"])]
        p = dict(item)
        p["post_edit_errors_avoided"] = round(p["post_edit_errors_avoided"], 8)
        p["new_edit_errors_avoided"] = round(p["new_edit_errors_avoided"], 8)
        p["post_edit_error_recall"] = (
            round(p["post_edit_errors_avoided"] / d["post_edit_stale_errors"], 8)
            if d["post_edit_stale_errors"] else None
        )
        p["new_edit_error_recall"] = (
            round(p["new_edit_errors_avoided"] / d["new_edit_induced_errors"], 8)
            if d["new_edit_induced_errors"] else None
        )
        out_policy.append(p)

    return {
        "posthoc_red_team": True,
        "decomposition_definition": {
            "new_edit_induced_error": "cached output matched pre-edit gold but not post-edit gold",
            "preexisting_persisting_error": "cached output differed from both pre-edit and post-edit gold",
            "preexisting_error_fixed_by_edit": "cached output differed from pre-edit gold but matched post-edit gold",
        },
        "error_decomposition": out_decomposition,
        "policy_capture": out_policy,
        "limits": [
            "This is a post-hoc diagnostic decomposition, not a preregistered primary endpoint.",
            "The shallow cached predictor is a symbolic one-round reasoner and is intentionally not a trained language model.",
            "The exact refresher is perfect by construction; counts are an optimistic upper-bound screen.",
            "No natural-language, calibration, latency, memory, or novelty claim follows.",
        ],
    }


def write_audit(output_path: Path, seed: int = 20260928, states: int = 256,
                composition_group_size: int = 8) -> dict[str, Any]:
    _result, rows = run_screen(seed, states, composition_group_size)
    result = audit_rows(rows)
    screen_path = Path(__file__).resolve().with_name("refresh_value_screen.py")
    result["config"] = {
        "seed": seed,
        "states": states,
        "composition_group_size": composition_group_size,
        "screen_script_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
        "score_rows": len(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--composition-group-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/data/2026-09-28-refresh-value-screen-redteam.json"),
    )
    args = parser.parse_args()
    result = write_audit(args.output, args.seed, args.states, args.composition_group_size)
    print(json.dumps(result["config"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

