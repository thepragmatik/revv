#!/usr/bin/env python3
"""Generate a tiny, deterministic intervention dataset for pipeline checks.

This is a synthetic diagnostic, not a benchmark and not a model evaluation.
It creates shared rule states, typed question fields, exact labels/proof traces,
and one-fact counterfactual edits with exact field-level change masks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PREDICATES = (
    "blue", "cold", "warm", "rough", "young", "furry",
    "kind", "round", "quiet", "soft", "playful", "friendly",
)
ENTITIES = ("Ari", "Bo", "Cy", "Dee", "Eli", "Flo")
BASE_PREDICATES = PREDICATES[:4]


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    entity: str
    positive: bool = True

    def key(self) -> str:
        return f"{self.entity}|{'+' if self.positive else '-'}|{self.predicate}"

    def natural_language(self) -> str:
        negation = "" if self.positive else "not "
        return f"{self.entity} is {negation}{self.predicate}."


@dataclass(frozen=True)
class Fact:
    fact_id: str
    literal: Literal


@dataclass(frozen=True)
class Rule:
    rule_id: str
    body: tuple[str, ...]
    head: str
    positive: bool

    def natural_language(self) -> str:
        body = " and ".join(
            ("" if self.positive else "not ") + predicate for predicate in self.body
        )
        head = ("" if self.positive else "not ") + self.head
        return f"If something is {body}, then it is {head}."


@dataclass(frozen=True)
class Proof:
    ref_id: str
    premises: tuple[Literal, ...] = ()


def _rule_graph(rng: random.Random, count: int = 16) -> list[tuple[tuple[str, ...], str]]:
    """Create an acyclic rule graph so exact forward chaining is finite."""
    # The backbone makes multi-hop proofs common enough to inspect. Extra local
    # conjunctions add alternate compositions without collapsing everything to
    # one-hop rules.
    chain = [((PREDICATES[i],), PREDICATES[i + 1]) for i in range(len(PREDICATES) - 1)]
    possible: list[tuple[tuple[str, ...], str]] = []
    for head_i in range(1, len(PREDICATES)):
        prior = PREDICATES[:head_i]
        for body in itertools.combinations(prior, 2):
            # Keep shortcut rules local so the chain still yields deeper proofs.
            min_body_i = min(PREDICATES.index(p) for p in body)
            if head_i - min_body_i <= 3:
                possible.append((body, PREDICATES[head_i]))
    rng.shuffle(possible)
    return chain[:count] + possible[:max(0, count - len(chain))]


def _render_rules(graph: Iterable[tuple[tuple[str, ...], str]]) -> list[Rule]:
    rules: list[Rule] = []
    index = 0
    for body, head in graph:
        for positive in (True, False):
            rules.append(Rule(f"r{index:03d}", body, head, positive))
            index += 1
    return rules


def _derive(facts: list[Fact], rules: list[Rule]) -> dict[Literal, Proof]:
    proofs: dict[Literal, Proof] = {fact.literal: Proof(fact.fact_id) for fact in facts}
    # Rules are acyclic in predicate order. Repeated relaxation also makes the
    # result independent of source ordering and easy to inspect.
    changed = True
    while changed:
        changed = False
        for rule in rules:
            for entity in ENTITIES:
                premises = tuple(Literal(p, entity, rule.positive) for p in rule.body)
                if all(premise in proofs for premise in premises):
                    head = Literal(rule.head, entity, rule.positive)
                    if head not in proofs:
                        proofs[head] = Proof(rule.rule_id, premises)
                        changed = True
    # This probe intentionally uses consistent, explicit-polarity theories.
    for entity in ENTITIES:
        for predicate in PREDICATES:
            if (Literal(predicate, entity, True) in proofs and
                    Literal(predicate, entity, False) in proofs):
                raise ValueError(f"Generated inconsistent theory at {entity}/{predicate}")
    return proofs


def _proof_evidence(literal: Literal, proofs: dict[Literal, Proof]) -> list[str]:
    node = proofs.get(literal)
    if node is None:
        return []
    evidence = {node.ref_id}
    for premise in node.premises:
        evidence.update(_proof_evidence(premise, proofs))
    return sorted(evidence)


def _label(predicate: str, entity: str, proofs: dict[Literal, Proof]) -> tuple[str, list[str], int | None]:
    positive = Literal(predicate, entity, True)
    negative = Literal(predicate, entity, False)
    if positive in proofs:
        return "true", _proof_evidence(positive, proofs), _proof_depth(positive, proofs)
    if negative in proofs:
        return "false", _proof_evidence(negative, proofs), _proof_depth(negative, proofs)
    return "unknown", [], None


def _proof_depth(literal: Literal, proofs: dict[Literal, Proof]) -> int:
    node = proofs[literal]
    if not node.premises:
        return 0
    return 1 + max(_proof_depth(premise, proofs) for premise in node.premises)


def _field(field_id: str, predicate: str, entity: str, proofs: dict[Literal, Proof]) -> dict:
    label, evidence, depth = _label(predicate, entity, proofs)
    return {
        "field_id": field_id,
        "query": f"Is {entity} {predicate}?",
        "target": {"entity": entity, "predicate": predicate},
        "label": label,
        "proof_evidence": evidence,
        "proof_depth": depth,
    }


def _choose_intervention(
    facts: list[Fact], rules: list[Rule], fields: list[tuple[str, str, str]], rng: random.Random
) -> tuple[dict, list[dict]]:
    """Return an all-edits random sample and a separately tagged effective sample."""
    before_proofs = _derive(facts, rules)
    before = {fid: _field(fid, pred, ent, before_proofs) for fid, pred, ent in fields}
    candidates: list[tuple[int, str, list[Fact], dict, dict]] = []
    for fact in facts:
        edited = [other for other in facts if other.fact_id != fact.fact_id]
        after_proofs = _derive(edited, rules)
        after = {fid: _field(fid, pred, ent, after_proofs) for fid, pred, ent in fields}
        changed = sum(before[fid]["label"] != after[fid]["label"] for fid, _, _ in fields)
        candidates.append((changed, f"remove:{fact.fact_id}", edited, after, {"operation": "remove", "fact_id": fact.fact_id, "text": fact.literal.natural_language()}))
    present = {fact.literal for fact in facts}
    for entity in ENTITIES:
        sign = next((fact.literal.positive for fact in facts if fact.literal.entity == entity), True)
        for predicate in BASE_PREDICATES:
            lit = Literal(predicate, entity, sign)
            if lit in present:
                continue
            fact_id = f"f_add_{entity}_{predicate}"
            edited = facts + [Fact(fact_id, lit)]
            after_proofs = _derive(edited, rules)
            after = {fid: _field(fid, pred, ent, after_proofs) for fid, pred, ent in fields}
            changed = sum(before[fid]["label"] != after[fid]["label"] for fid, _, _ in fields)
            candidates.append((changed, f"add:{fact_id}", edited, after, {"operation": "add", "fact_id": fact_id, "text": lit.natural_language()}))
    def materialize(candidate: tuple[int, str, list[Fact], dict, dict]) -> dict:
        changed_count, _, _, after_fields, edit = candidate
        masks = [
            {
                "field_id": fid,
                "before": before[fid]["label"],
                "after": after_fields[fid]["label"],
                "should_change": before[fid]["label"] != after_fields[fid]["label"],
            }
            for fid, _, _ in fields
        ]
        return {
            "edit": edit,
            "changed_field_count": changed_count,
            "affected_field_mask": masks,
            "after_fields": [after_fields[fid] for fid, _, _ in fields],
        }

    random_candidate = rng.choice(candidates)
    effective_candidates = [candidate for candidate in candidates if candidate[0] > 0]
    effective_candidate = rng.choice(effective_candidates) if effective_candidates else random_candidate
    return {
        "uniform_candidate_sample": materialize(random_candidate),
        # This stratum is conditioned on changing at least one queried label.
        # It is a stress/control sample, not a prevalence estimate.
        "effective_edit_sample": materialize(effective_candidate),
    }, [before[fid] for fid, _, _ in fields]


def generate_state(state_index: int, seed: int, max_fields: int = 20, composition_group_size: int = 8) -> dict:
    if composition_group_size < 1:
        raise ValueError("composition_group_size must be at least 1")
    if not 1 <= max_fields <= len(ENTITIES) * len(PREDICATES):
        raise ValueError("max_fields must be in [1, entity_count * predicate_count]")
    composition_group_index = state_index // composition_group_size
    graph_rng = random.Random(seed + composition_group_index * 1009)
    rng = random.Random(seed + state_index * 2003 + 37)
    graph = _rule_graph(graph_rng)
    rules = _render_rules(graph)
    facts: list[Fact] = []
    for entity_i, entity in enumerate(ENTITIES):
        positive = (rng.randrange(2) == 0)
        sampled = rng.sample(BASE_PREDICATES, k=rng.randint(1, 3))
        for predicate_i, predicate in enumerate(sorted(sampled)):
            literal = Literal(predicate, entity, positive)
            facts.append(Fact(f"f{entity_i:02d}_{predicate_i:02d}", literal))
    proofs = _derive(facts, rules)
    all_queries = [(predicate, entity) for entity in ENTITIES for predicate in PREDICATES]
    rng.shuffle(all_queries)
    selected = all_queries[:max_fields]
    fields = [(f"q{i:02d}", predicate, entity) for i, (predicate, entity) in enumerate(selected)]
    base_fields = [_field(fid, pred, ent, proofs) for fid, pred, ent in fields]
    interventions, before_fields = _choose_intervention(facts, rules, fields, rng)
    signature = "|".join(f"{','.join(body)}->{head}" for body, head in sorted(graph))
    composition_group = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    composition_split = "heldout_composition" if int(composition_group[:8], 16) % 4 == 0 else "seen_composition"
    statements = ([
        {"id": fact.fact_id, "kind": "fact", "literal": fact.literal.key(), "text": fact.literal.natural_language()}
        for fact in facts
    ] + [
        {"id": rule.rule_id, "kind": "rule", "text": rule.natural_language(), "body": list(rule.body), "head": rule.head, "positive": rule.positive}
        for rule in rules
    ])
    bundles = {}
    for q in (1, 5, 20):
        ids = [fid for fid, _, _ in fields[:min(q, len(fields))]]
        bundles[str(q)] = ids
    return {
        "state_id": f"s{state_index:05d}",
        "composition_group": composition_group,
        "composition_group_index": composition_group_index,
        "composition_split": composition_split,
        "statements": statements,
        "facts": [fact.fact_id for fact in facts],
        "rules": [rule.rule_id for rule in rules],
        "fields": base_fields,
        "question_bundles": bundles,
        "interventions": interventions,
        "diagnostic_only": True,
    }


def summarize(records: list[dict]) -> dict:
    proof_depths: list[int] = []
    intervention_stats = {}
    for record in records:
        proof_depths.extend(field["proof_depth"] for field in record["fields"] if field["proof_depth"] is not None)
    for intervention_name in ("uniform_candidate_sample", "effective_edit_sample"):
        changed_per_q = {"1": [], "5": [], "20": []}
        nonempty_per_q = {"1": 0, "5": 0, "20": 0}
        mask_histogram: dict[str, int] = {}
        empty_masks = 0
        for record in records:
            selected = record["interventions"][intervention_name]
            masks = {item["field_id"]: item["should_change"] for item in selected["affected_field_mask"]}
            changed = sum(masks.values())
            mask_histogram[str(changed)] = mask_histogram.get(str(changed), 0) + 1
            empty_masks += int(changed == 0)
            for q, ids in record["question_bundles"].items():
                changed_count = sum(masks[fid] for fid in ids)
                changed_per_q[q].append(changed_count)
                nonempty_per_q[q] += int(changed_count > 0)
        intervention_stats[intervention_name] = {
            "empty_masks_over_20_fields": empty_masks,
            "affected_field_count_histogram_over_20": dict(sorted(mask_histogram.items(), key=lambda pair: int(pair[0]))),
            "mean_affected_fields_by_bundle": {
                q: statistics.mean(values) if values else 0.0 for q, values in changed_per_q.items()
            },
            "fraction_of_bundles_with_any_affected_field": {
                q: nonempty_per_q[q] / len(records) if records else 0.0 for q in nonempty_per_q
            },
        }
    return {
        "states": len(records),
        "composition_groups": len({record["composition_group"] for record in records}),
        "states_by_composition_split": {
            split: sum(record["composition_split"] == split for record in records)
            for split in ("seen_composition", "heldout_composition")
        },
        "diagnostic_only": True,
        "question_counts": [1, 5, 20],
        "median_nonempty_proof_depth": statistics.median(proof_depths) if proof_depths else None,
        "interventions": intervention_stats,
        "limits": [
            "No learned model is included.",
            "The data are templated synthetic logic and do not measure natural-language transfer.",
            "No latency claim follows from symbolic proof generation or operation counts.",
        ],
    }


def run(seed: int, states: int, output_dir: Path, composition_group_size: int = 8) -> dict:
    if composition_group_size < 1:
        raise ValueError("composition_group_size must be at least 1")
    records = [generate_state(i, seed, composition_group_size=composition_group_size) for i in range(states)]
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "intervention-bundles.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = summarize(records)
    summary.update({
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        "seed": seed,
        "composition_group_size": composition_group_size,
        "jsonl": jsonl.name,
    })
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--composition-group-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/data/intervention-probe"))
    args = parser.parse_args()
    if args.states < 1:
        parser.error("--states must be at least 1")
    print(json.dumps(run(args.seed, args.states, args.output_dir, args.composition_group_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
