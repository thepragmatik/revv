"""Check the shared-trunk depth-allocation reduction against exhaustive search.

This verifies only optimizer algebra for additive expected task loss and costs.
It says nothing about model accuracy, calibration, or measured latency.
"""
from __future__ import annotations

from itertools import product
import random
from typing import Sequence


def objective(
    depths: Sequence[int],
    losses: Sequence[Sequence[float]],
    branch_costs: Sequence[Sequence[float]],
    shared_costs: Sequence[float],
    cost_weight: float,
) -> float:
    deepest = max(depths)
    return (
        sum(losses[q][d] for q, d in enumerate(depths))
        + cost_weight
        * (
            shared_costs[deepest]
            + sum(branch_costs[q][d] for q, d in enumerate(depths))
        )
    )


def solve_exact(
    losses: Sequence[Sequence[float]],
    branch_costs: Sequence[Sequence[float]],
    shared_costs: Sequence[float],
    cost_weight: float,
) -> tuple[float, tuple[int, ...]]:
    """Return the minimum objective and depth vector in O(Q * L)."""
    question_count = len(losses)
    max_depth = len(shared_costs) - 1
    if question_count == 0:
        raise ValueError("at least one question is required")
    if len(branch_costs) != question_count:
        raise ValueError("one branch-cost profile is required per question")
    if any(len(row) != max_depth + 1 for row in losses):
        raise ValueError("loss profiles must have one entry per depth")
    if any(len(row) != max_depth + 1 for row in branch_costs):
        raise ValueError("branch-cost profiles must have one entry per depth")

    weighted = [
        [
            losses[q][d] + cost_weight * branch_costs[q][d]
            for d in range(max_depth + 1)
        ]
        for q in range(question_count)
    ]

    prefix_argmin: list[list[int]] = []
    for row in weighted:
        best_at_depth = 0
        prefix = []
        for depth, value in enumerate(row):
            if value < row[best_at_depth]:
                best_at_depth = depth
            prefix.append(best_at_depth)
        prefix_argmin.append(prefix)

    best_value = float("inf")
    best_depths: tuple[int, ...] = ()
    for deepest in range(max_depth + 1):
        minima = [prefix_argmin[q][deepest] for q in range(question_count)]
        regrets = [
            weighted[q][deepest] - weighted[q][minima[q]]
            for q in range(question_count)
        ]
        forced_question = min(range(question_count), key=lambda q: regrets[q])
        depths = list(minima)
        depths[forced_question] = deepest
        value = (
            cost_weight * shared_costs[deepest]
            + sum(weighted[q][minima[q]] for q in range(question_count))
            + regrets[forced_question]
        )
        if value < best_value:
            best_value = value
            best_depths = tuple(depths)
    return best_value, best_depths


def solve_brute_force(
    losses: Sequence[Sequence[float]],
    branch_costs: Sequence[Sequence[float]],
    shared_costs: Sequence[float],
    cost_weight: float,
) -> tuple[float, tuple[int, ...]]:
    max_depth = len(shared_costs) - 1
    candidates = product(range(max_depth + 1), repeat=len(losses))
    return min(
        (
            objective(depths, losses, branch_costs, shared_costs, cost_weight),
            tuple(depths),
        )
        for depths in candidates
    )


def main() -> None:
    rng = random.Random(20260927)
    cases = 0
    maximum_difference = 0.0
    for _ in range(3000):
        question_count = rng.randint(1, 4)
        max_depth = rng.randint(1, 5)

        losses = [
            [rng.random() * 4 for _ in range(max_depth + 1)]
            for _ in range(question_count)
        ]

        def cumulative_cost() -> list[float]:
            values = [0.0]
            for _depth in range(max_depth):
                values.append(values[-1] + rng.random() * 2)
            return values

        branch_costs = [cumulative_cost() for _ in range(question_count)]
        shared_costs = cumulative_cost()
        cost_weight = rng.random() * 2

        exact = solve_exact(losses, branch_costs, shared_costs, cost_weight)
        brute = solve_brute_force(losses, branch_costs, shared_costs, cost_weight)
        difference = abs(exact[0] - brute[0])
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-10:
            raise AssertionError(
                f"case {cases}: solver {exact[0]} != brute force {brute[0]}"
            )
        cases += 1

    print(
        f"PASS: {cases} deterministic randomized problems matched exhaustive search; "
        f"maximum objective difference {maximum_difference:.17g}."
    )


if __name__ == "__main__":
    main()
