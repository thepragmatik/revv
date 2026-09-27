import copy
import unittest

import kaggle_intervention_risk_probe as collector


def valid_report():
    return {
        "kind": collector.EXPECTED_KIND,
        "seed": 20260928,
        "states": 1024,
        "composition_group_size": 8,
        "composition_groups": 128,
        "generator_sha256": collector.EXPECTED_GENERATOR_SHA256,
        "generated_records_sha256": collector.EXPECTED_RECORDS_SHA256,
        "groups_by_split": {"fit": 81, "calibration": 17, "heldout": 30},
        "states_by_split": {"heldout": 240},
        "sentence_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sentence_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
        "nli_revision": "c4d86af4493123990d7762712de9ed730c876161",
        "heldout_results": {
            "selection": {
                "q20": {
                    "intervention_risk": {"0.25": {"changed_fields": 74, "changed_rule_groups": 22}},
                    "confidence_only": {"0.25": {"changed_fields": 74, "changed_rule_groups": 22}},
                    "cluster_bootstrap_recall_comparisons_at_25pct": {
                        "intervention_risk_minus_confidence": {
                            "changed_fields": 74,
                            "delta_recall_a_minus_b": 0.12,
                            "cluster_bootstrap_95pct_interval": [0.02, 0.20],
                            "bootstrap_replicates": 2000,
                        }
                    },
                }
            },
            "update_quality": {"q20": {"cached_pre_edit_prediction": {"n": 4800}}},
            "paired_accuracy_deltas": {
                "q20_risk_gate25_minus_cached": {
                    "delta_accuracy_a_minus_b": 0.01,
                    "cluster_bootstrap_95pct_interval": [-0.01, 0.03],
                }
            },
        },
        "nli_heldout_timing": {"pairs_per_route": 4800},
        "kaggle_cpu_reference_profile": {
            "request_latency": {
                "direct_full": {"20": {"p50_ms": 150.0}},
                "intervention_risk_gate25": {"20": {"p50_ms": 100.0}},
            }
        },
    }


class KaggleCollectorValidationTests(unittest.TestCase):
    def test_accepts_complete_screen_and_applies_preregistered_gate(self):
        result = collector.validate_report(valid_report(), collector.EXPECTED_GENERATOR_SHA256)
        self.assertTrue(result["validated"])
        self.assertEqual(result["primary_gate_status"], "pass")
        self.assertEqual(result["heldout_q20_changed_fields"], 74)

    def test_preserves_underpowered_run_as_inconclusive(self):
        report = valid_report()
        report["heldout_results"]["selection"]["q20"]["intervention_risk"]["0.25"]["changed_fields"] = 49
        report["heldout_results"]["selection"]["q20"]["cluster_bootstrap_recall_comparisons_at_25pct"][
            "intervention_risk_minus_confidence"]["changed_fields"] = 49
        result = collector.validate_report(report, collector.EXPECTED_GENERATOR_SHA256)
        self.assertFalse(result["primary_sample_size_gate_pass"])
        self.assertIsNone(result["primary_recall_gate_pass"])
        self.assertEqual(result["primary_gate_status"], "inconclusive_underpowered")

    def test_rejects_stale_generator_or_incomplete_quality_output(self):
        with self.assertRaisesRegex(ValueError, "different intervention generator"):
            collector.validate_report(valid_report(), "stale")
        incomplete = copy.deepcopy(valid_report())
        del incomplete["heldout_results"]["paired_accuracy_deltas"]
        with self.assertRaisesRegex(ValueError, "paired update-quality interval"):
            collector.validate_report(incomplete, collector.EXPECTED_GENERATOR_SHA256)


if __name__ == "__main__":
    unittest.main()
