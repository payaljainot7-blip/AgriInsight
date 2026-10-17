"""
Unit tests for Data Preprocessing and Transaction Transformation.
"""

import os
from core.preprocessing import (
    load_transactions_from_csv,
    load_raw_dataset_records,
    transform_raw_to_transactions,
    discretize_value,
    get_transaction_statistics,
    DEFAULT_NUMERIC_BINS
)


def test_discretize_value():
    bins = [("Low", float("-inf"), 50.0), ("Medium", 50.0, 100.0), ("High", 100.0, float("inf"))]
    assert discretize_value(25.0, bins) == "Low"
    assert discretize_value(50.0, bins) == "Medium"
    assert discretize_value(75.0, bins) == "Medium"
    assert discretize_value(100.0, bins) == "High"
    assert discretize_value(150.0, bins) == "High"


def test_transform_raw_to_transactions():
    raw_records = [
        {
            "transaction_id": "AGR001",
            "region": "Vidarbha",
            "crop": "Cotton",
            "nitrogen_kg_ha": "150.0",  # High (>=130)
            "soil_ph": "7.0",           # Neutral (6.2-7.5)
            "yield_category": "High"
        },
        {
            "transaction_id": "AGR002",
            "region": "Punjab",
            "crop": "Wheat",
            "nitrogen_kg_ha": "40.0",   # Low (<80)
            "soil_ph": "5.5",           # Acidic (<6.2)
            "yield_category": "Medium"
        }
    ]

    transactions = transform_raw_to_transactions(raw_records)
    assert len(transactions) == 2

    t1 = transactions[0]
    assert not any("transaction_id" in item.lower() for item in t1)
    assert "Region=Vidarbha" in t1
    assert "Crop=Cotton" in t1
    assert "Nitrogen=High" in t1
    assert "Soil_pH=Neutral" in t1
    assert "Yield=High" in t1

    t2 = transactions[1]
    assert "Region=Punjab" in t2
    assert "Crop=Wheat" in t2
    assert "Nitrogen=Low" in t2
    assert "Soil_pH=Acidic" in t2
    assert "Yield=Medium" in t2


def test_transaction_statistics():
    transactions = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A"]
    ]
    stats = get_transaction_statistics(transactions)
    assert stats["num_transactions"] == 3
    assert stats["num_distinct_items"] == 3
    assert stats["min_length"] == 1
    assert stats["max_length"] == 3
    assert stats["mean_length"] == 2.0
    assert stats["median_length"] == 2.0


def test_load_builtin_transaction_file():
    filepath = os.path.join("data", "agriinsight_synthetic_100k_transactions.csv")
    if not os.path.exists(filepath):
        filepath = "agriinsight_synthetic_100k_transactions.csv"

    if os.path.exists(filepath):
        tx = load_transactions_from_csv(filepath, max_rows=50)
        assert len(tx) == 50
        assert len(tx[0]) > 5
        for t in tx:
            assert len(t) == len(set(t))


if __name__ == "__main__":
    test_discretize_value()
    test_transform_raw_to_transactions()
    test_transaction_statistics()
    test_load_builtin_transaction_file()
    print("test_preprocessing.py: ALL TESTS PASSED")
