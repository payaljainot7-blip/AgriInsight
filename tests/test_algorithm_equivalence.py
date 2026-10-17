"""
Tests for Algorithm Equivalence:
Apriori == FP-Growth == ECLAT == H-Mine.
Confirms identical complete frequent itemsets and exact support counts across diverse test datasets.
"""

import os
import random
from algorithms.apriori import apriori
from algorithms.fpgrowth import fpgrowth
from algorithms.eclat import eclat
from algorithms.hmine import hmine
from core.verification import verify_all_four, compare_two_algorithms, recalculate_exact_counts
from core.preprocessing import load_transactions_from_csv


def test_equivalence_manual_dataset():
    # Dataset 1: The canonical 5-transaction academic specification
    dataset = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
        ["A", "B", "C"]
    ]
    min_support = 0.40
    min_count = 2

    res_a = apriori(dataset, min_support)
    res_f = fpgrowth(dataset, min_support)
    res_e = eclat(dataset, min_support)
    res_h = hmine(dataset, min_support)

    assert res_a == res_f == res_e == res_h

    # Independent recount check
    recalc = recalculate_exact_counts(dataset, list(res_a.keys()))
    for iset, cnt in res_a.items():
        assert recalc[tuple(sorted(iset))] == cnt

    verification = verify_all_four(
        {"Apriori": res_a, "FP-Growth": res_f, "ECLAT": res_e, "H-Mine": res_h},
        min_count=min_count
    )
    assert verification["overall_status"] == "PASS"
    assert verification["all_pairs_pass"] is True
    assert len(verification["pairwise_comparisons"]) == 6
    for pair in verification["pairwise_comparisons"]:
        assert pair["status"] == "PASS"


def test_equivalence_market_basket_dataset():
    # Dataset 2: Retail Market Basket transactions
    dataset = [
        ["Milk", "Bread", "Butter"],
        ["Bread", "Butter"],
        ["Milk", "Eggs"],
        ["Bread", "Eggs", "Milk", "Butter"],
        ["Bread", "Butter"],
        ["Milk", "Bread", "Eggs"],
        ["Coke", "Chips"],
        ["Coke", "Chips", "Bread"],
        ["Milk", "Bread", "Butter", "Eggs"],
        ["Milk", "Butter"]
    ]
    min_support = 0.30

    res_a = apriori(dataset, min_support)
    res_f = fpgrowth(dataset, min_support)
    res_e = eclat(dataset, min_support)
    res_h = hmine(dataset, min_support)

    assert res_a == res_f == res_e == res_h
    assert len(res_a) > 0

    recalc = recalculate_exact_counts(dataset, list(res_a.keys()))
    for iset, cnt in res_a.items():
        assert recalc[tuple(sorted(iset))] == cnt

    verification = verify_all_four(
        {"Apriori": res_a, "FP-Growth": res_f, "ECLAT": res_e, "H-Mine": res_h},
        min_count=3
    )
    assert verification["overall_status"] == "PASS"


def test_equivalence_dense_agricultural_patterns():
    # Dataset 3: Dense agricultural cluster patterns
    dataset = [
        ["Crop=Rice", "Soil=Clay", "Water=High", "Season=Kharif", "Fertilizer=Urea"],
        ["Crop=Rice", "Soil=Clay", "Water=High", "Season=Kharif"],
        ["Crop=Rice", "Soil=Clay", "Water=High"],
        ["Crop=Wheat", "Soil=Alluvial", "Water=Medium", "Season=Rabi", "Fertilizer=DAP"],
        ["Crop=Wheat", "Soil=Alluvial", "Water=Medium", "Season=Rabi"],
        ["Crop=Tomato", "Irrigation=Drip", "Fertilizer=NPK", "Yield=High"],
        ["Crop=Tomato", "Irrigation=Drip", "Fertilizer=NPK", "Yield=High", "Soil=Sandy"],
        ["Crop=Tomato", "Irrigation=Drip", "Yield=High"],
        ["Crop=Rice", "Soil=Clay", "Season=Kharif", "Water=High"],
        ["Crop=Wheat", "Soil=Alluvial", "Season=Rabi"],
        ["Crop=Rice", "Soil=Clay", "Water=High", "Yield=High"],
        ["Crop=Tomato", "Irrigation=Drip", "Fertilizer=NPK"],
    ]
    min_support = 0.25  # min_count = ceil(0.25 * 12) = 3

    res_a = apriori(dataset, min_support)
    res_f = fpgrowth(dataset, min_support)
    res_e = eclat(dataset, min_support)
    res_h = hmine(dataset, min_support)

    assert res_a == res_f == res_e == res_h
    assert len(res_a) > 0

    recalc = recalculate_exact_counts(dataset, list(res_a.keys()))
    for iset, cnt in res_a.items():
        assert recalc[tuple(sorted(iset))] == cnt

    verification = verify_all_four(
        {"Apriori": res_a, "FP-Growth": res_f, "ECLAT": res_e, "H-Mine": res_h},
        min_count=3
    )
    assert verification["overall_status"] == "PASS"


def test_equivalence_sparse_high_diversity():
    # Dataset 4: Sparse transactions with high vocabulary diversity
    dataset = [
        ["I1", "I2", "I5"],
        ["I2", "I4"],
        ["I2", "I3"],
        ["I1", "I2", "I4"],
        ["I1", "I3"],
        ["I2", "I3"],
        ["I1", "I3"],
        ["I1", "I2", "I3", "I5"],
        ["I1", "I2", "I3"],
        ["I6", "I7"],
        ["I6", "I8", "I1"],
        ["I7", "I8", "I2"],
        ["I1", "I2", "I3", "I4", "I5"]
    ]
    min_support = 0.20  # min_count = ceil(0.20 * 13) = 3

    res_a = apriori(dataset, min_support)
    res_f = fpgrowth(dataset, min_support)
    res_e = eclat(dataset, min_support)
    res_h = hmine(dataset, min_support)

    assert res_a == res_f == res_e == res_h

    recalc = recalculate_exact_counts(dataset, list(res_a.keys()))
    for iset, cnt in res_a.items():
        assert recalc[tuple(sorted(iset))] == cnt

    verification = verify_all_four(
        {"Apriori": res_a, "FP-Growth": res_f, "ECLAT": res_e, "H-Mine": res_h},
        min_count=3
    )
    assert verification["overall_status"] == "PASS"


def test_equivalence_random_reproducible_dataset():
    # Dataset 5: Random synthetic transactions generated with fixed seed
    rng = random.Random(42)
    items_pool = [f"Item_{x}" for x in range(12)]
    dataset = []
    for _ in range(50):
        length = rng.randint(2, 6)
        tx = rng.sample(items_pool, length)
        dataset.append(tx)

    min_support = 0.10  # min_count = ceil(0.10 * 50) = 5
    res_a = apriori(dataset, min_support)
    res_f = fpgrowth(dataset, min_support)
    res_e = eclat(dataset, min_support)
    res_h = hmine(dataset, min_support)

    assert res_a == res_f == res_e == res_h
    assert len(res_a) > 0

    recalc = recalculate_exact_counts(dataset, list(res_a.keys()))
    for iset, cnt in res_a.items():
        assert recalc[tuple(sorted(iset))] == cnt

    verification = verify_all_four(
        {"Apriori": res_a, "FP-Growth": res_f, "ECLAT": res_e, "H-Mine": res_h},
        min_count=5
    )
    assert verification["overall_status"] == "PASS"


def test_equivalence_with_max_length():
    dataset = [
        ["Crop=Rice", "Soil=Clay", "Water=High", "Season=Kharif"],
        ["Crop=Rice", "Soil=Clay", "Water=High"],
        ["Crop=Wheat", "Soil=Clay", "Water=High"],
        ["Crop=Rice", "Soil=Clay", "Water=High", "Season=Kharif", "Fertilizer=Urea"],
        ["Crop=Rice", "Season=Kharif"],
    ]
    min_support = 0.40

    res_a = apriori(dataset, min_support, max_length=2)
    res_f = fpgrowth(dataset, min_support, max_length=2)
    res_e = eclat(dataset, min_support, max_length=2)
    res_h = hmine(dataset, min_support, max_length=2)

    assert res_a == res_f == res_e == res_h
    assert all(len(itemset) <= 2 for itemset in res_a.keys())


def test_equivalence_on_agricultural_transactions_sample():
    tx_file = os.path.join("data", "agriinsight_synthetic_100k_transactions.csv")
    if not os.path.exists(tx_file):
        tx_file = "agriinsight_synthetic_100k_transactions.csv"

    if os.path.exists(tx_file):
        transactions = load_transactions_from_csv(tx_file, max_rows=500)
        assert len(transactions) == 500

        min_support = 0.15
        res_a = apriori(transactions, min_support, max_length=3)
        res_f = fpgrowth(transactions, min_support, max_length=3)
        res_e = eclat(transactions, min_support, max_length=3)
        res_h = hmine(transactions, min_support, max_length=3)

        assert res_a == res_f == res_e == res_h
        assert len(res_a) > 0

        verification = verify_all_four(
            {"Apriori": res_a, "FP-Growth": res_f, "ECLAT": res_e, "H-Mine": res_h},
            min_count=int(500 * min_support)
        )
        assert verification["overall_status"] == "PASS"


if __name__ == "__main__":
    test_equivalence_manual_dataset()
    test_equivalence_market_basket_dataset()
    test_equivalence_dense_agricultural_patterns()
    test_equivalence_sparse_high_diversity()
    test_equivalence_random_reproducible_dataset()
    test_equivalence_with_max_length()
    test_equivalence_on_agricultural_transactions_sample()
    print("test_algorithm_equivalence.py: ALL TESTS PASSED")
