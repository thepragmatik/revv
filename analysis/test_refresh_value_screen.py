import unittest

from intervention_probe import _derive, generate_state
from refresh_value_screen import (
    _apply_edit,
    _score_state_edit,
    derive_bounded,
    evaluate_rows,
    run_screen,
    _state_logic,
)


class RefreshValueScreenTests(unittest.TestCase):
    def test_bounded_proofs_are_subsets_and_converge_to_exact_closure(self):
        record = generate_state(2, 20260928)
        facts, rules = _state_logic(record)
        exact = _derive(facts, rules)
        shallow = derive_bounded(facts, rules, 1)
        converged = derive_bounded(facts, rules, 20)
        self.assertTrue(set(shallow).issubset(exact))
        self.assertEqual(converged.keys(), exact.keys())

    def test_reconstructed_edits_preserve_exact_labels_and_masks(self):
        record = generate_state(9, 20260928)
        for stratum in ("uniform_candidate_sample", "effective_edit_sample"):
            rows = _score_state_edit(record, stratum)
            self.assertEqual(len(rows), 20)
            for row in rows:
                self.assertEqual(row["affected"], row["pre_edit_gold"] != row["post_edit_gold"])
                self.assertEqual(
                    row["stale_error"],
                    row["stale_prediction"] != row["post_edit_gold"],
                )

    def test_oracle_is_upper_bound_and_exact_full_budget_clears_errors(self):
        record = generate_state(11, 20260928)
        rows = _score_state_edit(record, "uniform_candidate_sample")
        evaluated = evaluate_rows(rows)["aggregates"]
        held = [
            row for row in evaluated
            if row["bundle_size"] == 20
            and row["refresh_budget_per_bundle"] == 10
        ]
        saved = {row["policy"]: row["errors_avoided"] for row in held}
        self.assertGreaterEqual(saved["oracle_error"], saved["cheap_flip"])
        self.assertGreaterEqual(saved["oracle_error"], saved["unproved_after"])
        self.assertGreaterEqual(saved["oracle_error"], saved["edit_query_jaccard"])
        self.assertEqual(saved["oracle_error"], 10.0)
        full = [
            row for row in evaluated
            if row["bundle_size"] == 20
            and row["refresh_budget_per_bundle"] == 20
        ]
        for row in full:
            self.assertEqual(row["errors_avoided"], row["baseline_stale_errors"])
            self.assertEqual(row["remaining_errors"], 0.0)

    def test_screen_is_deterministic_and_composition_groups_do_not_leak(self):
        first, rows_a = run_screen(seed=20260928, states=16, composition_group_size=8)
        second, rows_b = run_screen(seed=20260928, states=16, composition_group_size=8)
        self.assertEqual(first, second)
        self.assertEqual(rows_a, rows_b)
        by_group = {}
        for row in rows_a:
            old = by_group.setdefault(row["composition_group"], row["composition_split"])
            self.assertEqual(old, row["composition_split"])


if __name__ == "__main__":
    unittest.main()

