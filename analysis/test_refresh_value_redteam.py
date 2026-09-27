import unittest

from refresh_value_screen import run_screen
from refresh_value_redteam import audit_rows


class RefreshValueRedTeamTests(unittest.TestCase):
    def test_error_decomposition_and_policy_budget_are_consistent(self):
        _screen, rows = run_screen(seed=20260928, states=8, composition_group_size=8)
        result = audit_rows(rows)
        q20 = [
            item for item in result["error_decomposition"]
            if item["bundle_size"] == 20
        ]
        self.assertEqual(len(q20), 2)
        for item in q20:
            self.assertEqual(
                item["post_edit_stale_errors"],
                item["preexisting_errors_persisting"] + item["new_edit_induced_errors"],
            )
            self.assertLessEqual(item["preexisting_errors_fixed_by_edit"], item["pre_edit_stale_errors"])
        full = [
            item for item in result["policy_capture"]
            if item["bundle_size"] == 20 and item["refresh_budget_per_bundle"] == 20
        ]
        decomposition = {
            (item["composition_split"], item["edit_stratum"]): item
            for item in q20
        }
        for item in full:
            self.assertEqual(item["post_edit_error_recall"], 1.0)
            errors = decomposition[(item["composition_split"], item["edit_stratum"])]["new_edit_induced_errors"]
            if errors:
                self.assertEqual(item["new_edit_error_recall"], 1.0)
            else:
                self.assertIsNone(item["new_edit_error_recall"])


if __name__ == "__main__":
    unittest.main()

