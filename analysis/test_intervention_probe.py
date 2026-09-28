import unittest

from intervention_probe import Fact, Literal, Rule, _derive, _label, generate_state


class InterventionProbeTests(unittest.TestCase):
    def test_positive_chain_returns_exact_proof_evidence(self):
        facts = [Fact("f0", Literal("blue", "Ari", True))]
        rules = [
            Rule("r0", ("blue",), "warm", True),
            Rule("r1", ("warm",), "kind", True),
        ]
        proofs = _derive(facts, rules)
        label, evidence, depth = _label("kind", "Ari", proofs)
        self.assertEqual(label, "true")
        self.assertEqual(evidence, ["f0", "r0", "r1"])
        self.assertEqual(depth, 2)

    def test_explicit_negative_chain_is_false_and_unproved_is_unknown(self):
        facts = [Fact("f0", Literal("blue", "Ari", False))]
        rules = [Rule("r0", ("blue",), "warm", False)]
        proofs = _derive(facts, rules)
        self.assertEqual(_label("warm", "Ari", proofs)[0], "false")
        self.assertEqual(_label("friendly", "Ari", proofs), ("unknown", [], None))

    def test_generated_state_is_reproducible_and_masks_match_labels(self):
        first = generate_state(7, 20260927)
        second = generate_state(7, 20260927)
        self.assertEqual(first, second)
        before = {item["field_id"]: item["label"] for item in first["fields"]}
        for sample in first["interventions"].values():
            after = {item["field_id"]: item["label"] for item in sample["after_fields"]}
            masks = {item["field_id"]: item["should_change"] for item in sample["affected_field_mask"]}
            self.assertEqual(set(before), set(after))
            self.assertEqual(set(before), set(masks))
            for field_id in before:
                self.assertEqual(masks[field_id], before[field_id] != after[field_id])
            self.assertEqual(sample["changed_field_count"], sum(masks.values()))
        self.assertEqual(len(first["question_bundles"]["1"]), 1)
        self.assertEqual(len(first["question_bundles"]["5"]), 5)
        self.assertEqual(len(first["question_bundles"]["20"]), 20)

    def test_intervention_edits_exactly_one_fact(self):
        record = generate_state(3, 11)
        for sample in record["interventions"].values():
            edit = sample["edit"]
            self.assertIn(edit["operation"], ("add", "remove"))
            self.assertTrue(edit["fact_id"])
        self.assertTrue(record["diagnostic_only"])

    def test_composition_groups_are_reused_and_split_as_units(self):
        groups = [
            (generate_state(group * 8, 91), generate_state(group * 8 + 1, 91))
            for group in range(12)
        ]
        splits = set()
        for first, second in groups:
            self.assertEqual(first["composition_group"], second["composition_group"])
            self.assertEqual(first["composition_split"], second["composition_split"])
            splits.add(first["composition_split"])
        self.assertEqual(splits, {"seen_composition", "heldout_composition"})


if __name__ == "__main__":
    unittest.main()
