"""
Unit tests for Association Rule Generation and Metric Calculations.
"""

import math
import os
import sys

# Ensure root directory is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.metrics import calculate_rule_metrics
from core.association_rules import generate_association_rules, verify_rules_equivalence
from algorithms.fpgrowth import fpgrowth
from algorithms.apriori import apriori
from algorithms.eclat import eclat
from algorithms.hmine import hmine


def test_calculate_rule_metrics_exact():
    metrics = calculate_rule_metrics(
        count_antecedent=50,
        count_consequent=40,
        count_both=25,
        num_transactions=100
    )

    assert math.isclose(metrics["support"], 0.25)
    assert math.isclose(metrics["confidence"], 0.50)
    assert math.isclose(metrics["lift"], 1.25)
    assert math.isclose(metrics["leverage"], 0.05)
    assert math.isclose(metrics["conviction"], 1.20)


def test_rule_generation_on_manual_dataset():
    dataset = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
        ["A", "B", "C"]
    ]
    frequent = fpgrowth(dataset, min_support=0.40)
    # The manual dataset items A, B, C have 80% individual support each,
    # so pairwise joint support (60%) results in lift = 0.6 / 0.64 = 0.9375.
    # With min_lift=0.8, 12 valid rules are generated.
    rules = generate_association_rules(
        frequent_itemsets=frequent,
        num_transactions=5,
        min_confidence=0.5,
        min_lift=0.8
    )

    assert len(rules) == 12
    for r in rules:
        assert r["confidence"] >= 0.5
        assert r["lift"] >= 0.8
        assert r["support"] >= 0.40
        assert len(r["antecedent"]) >= 1
        assert len(r["consequent"]) >= 1
        assert set(r["antecedent"]).isdisjoint(set(r["consequent"]))


def test_rule_generation_high_lift():
    # Positively correlated dataset
    dataset = [
        ["Rice", "Clay", "Flood"],
        ["Rice", "Clay", "Flood"],
        ["Wheat", "Sandy", "Drip"],
        ["Wheat", "Sandy", "Drip"],
        ["Maize", "Loamy", "Rainfed"],
        ["Rice", "Clay", "Flood"],
        ["Wheat", "Sandy", "Drip"],
        ["Maize", "Loamy", "Rainfed"],
        ["Rice", "Clay", "Flood"],
        ["Wheat", "Sandy", "Drip"],
    ]
    frequent = fpgrowth(dataset, min_support=0.30)
    rules = generate_association_rules(
        frequent_itemsets=frequent,
        num_transactions=10,
        min_confidence=0.8,
        min_lift=1.5
    )

    assert len(rules) > 0
    for r in rules:
        assert r["confidence"] >= 0.8
        assert r["lift"] >= 1.5


def test_rules_identical_across_algorithms():
    dataset = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
        ["A", "B", "C"]
    ]
    itemsets_a = apriori(dataset, 0.40)
    itemsets_f = fpgrowth(dataset, 0.40)
    itemsets_e = eclat(dataset, 0.40)
    itemsets_h = hmine(dataset, 0.40)

    rules_a = generate_association_rules(itemsets_a, num_transactions=5, min_confidence=0.5, min_lift=0.8)
    rules_f = generate_association_rules(itemsets_f, num_transactions=5, min_confidence=0.5, min_lift=0.8)
    rules_e = generate_association_rules(itemsets_e, num_transactions=5, min_confidence=0.5, min_lift=0.8)
    rules_h = generate_association_rules(itemsets_h, num_transactions=5, min_confidence=0.5, min_lift=0.8)

    assert verify_rules_equivalence(rules_a, rules_f)
    assert verify_rules_equivalence(rules_a, rules_e)
    assert verify_rules_equivalence(rules_a, rules_h)


if __name__ == "__main__":
    test_calculate_rule_metrics_exact()
    test_rule_generation_on_manual_dataset()
    test_rule_generation_high_lift()
    test_rules_identical_across_algorithms()
    print("test_association_rules.py: ALL TESTS PASSED")
