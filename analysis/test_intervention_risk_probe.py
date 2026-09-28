"""Fast CPU checks for the intervention-risk experiment's routing mechanics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments" / "kaggle"))

import intervention_risk_probe as probe  # noqa: E402
from intervention_probe import generate_state  # noqa: E402


def calibration_rows(count: int = 20) -> list[dict]:
    return [
        {
            "state_id": f"s{i:04d}",
            "field_id": "q00",
            "composition_group": f"g{i // 2}",
            "bundles": {1: ["q00"], 5: ["q00"], 20: ["q00"]},
            "changed": i % 4 == 0,
        }
        for i in range(count)
    ]


class InterventionRiskRoutingTests(unittest.TestCase):
    def test_calibrated_budget_is_exact_and_prefers_largest_risk(self) -> None:
        rows = calibration_rows()
        scores = list(range(len(rows)))
        cuts = probe.fit_route_cuts(rows, np.asarray(scores, dtype=float))
        route = probe.route_by_rank(rows, np.asarray(scores, dtype=float), cuts, 20, 0.25)
        self.assertEqual(int(route.sum()), 5)
        self.assertEqual(set(__import__("numpy").flatnonzero(route)), {15, 16, 17, 18, 19})

    def test_tied_scores_use_stable_identity_order(self) -> None:
        rows = calibration_rows()
        scores = np.ones(len(rows), dtype=float)
        cuts = probe.fit_route_cuts(rows, scores)
        first = probe.route_by_rank(rows, scores, cuts, 5, 0.25)
        second = probe.route_by_rank(rows, scores, cuts, 5, 0.25)
        self.assertEqual(int(first.sum()), 5)
        self.assertTrue(np.array_equal(first, second))

    def test_group_bootstrap_detects_perfect_routing_gap(self) -> None:
        rows = calibration_rows(20)
        risk_route = np.ones(len(rows), dtype=bool)
        confidence_route = np.zeros(len(rows), dtype=bool)
        result = probe.cluster_bootstrap_recall_delta(rows, risk_route, confidence_route)
        self.assertEqual(result["delta_recall_a_minus_b"], 1.0)
        self.assertEqual(result["cluster_bootstrap_95pct_interval"], [1.0, 1.0])

    def test_paired_accuracy_bootstrap_uses_rule_groups(self) -> None:
        rows = calibration_rows(20)
        labels = ["true"] * len(rows)
        correct = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (len(rows), 1))
        wrong = np.tile(np.asarray([[0.0, 1.0, 0.0]]), (len(rows), 1))
        result = probe.cluster_bootstrap_accuracy_delta(rows, labels, correct, wrong)
        self.assertEqual(result["delta_accuracy_a_minus_b"], 1.0)
        self.assertEqual(result["cluster_bootstrap_95pct_interval"], [1.0, 1.0])

    def test_applies_only_the_sampled_single_fact_edit(self) -> None:
        record = generate_state(3, 20260927, composition_group_size=8)
        post = probe.post_edit_statements(record)
        edit = record["interventions"]["uniform_candidate_sample"]["edit"]
        expected_delta = -1 if edit["operation"] == "remove" else 1
        self.assertEqual(len(post) - len(record["statements"]), expected_delta)
        self.assertEqual(sum(item["kind"] == "fact" for item in post) - len(record["facts"]), expected_delta)


if __name__ == "__main__":
    unittest.main()
