"""
Unit tests for Synthetic Dataset Generator.
Tests reproducibility, schema conformity, and pattern sanity.
"""

import os
import tempfile
from data.generate_dataset import generate_agricultural_dataset
from core.preprocessing import load_raw_dataset_records, load_transactions_from_csv


def test_generator_reproducibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        res1 = generate_agricultural_dataset(num_records=50, seed=123, output_dir=tmpdir, file_prefix="run1")
        res2 = generate_agricultural_dataset(num_records=50, seed=123, output_dir=tmpdir, file_prefix="run2")

        with open(res1["raw_path"], "r", encoding="utf-8") as f1, open(res2["raw_path"], "r", encoding="utf-8") as f2:
            assert f1.read() == f2.read()

        with open(res1["transaction_path"], "r", encoding="utf-8") as f1, open(res2["transaction_path"], "r", encoding="utf-8") as f2:
            assert f1.read() == f2.read()


def test_generator_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = generate_agricultural_dataset(num_records=100, seed=42, output_dir=tmpdir, file_prefix="test100")
        records, summary = load_raw_dataset_records(res["raw_path"])
        assert len(records) == 100
        assert summary["total_columns"] == 18
        for col, missing in summary["missing_counts"].items():
            assert missing == 0

        transactions = load_transactions_from_csv(res["transaction_path"])
        assert len(transactions) == 100
        for t in transactions:
            assert len(t) == 17


if __name__ == "__main__":
    test_generator_reproducibility()
    test_generator_schema()
    print("test_dataset_generator.py: ALL TESTS PASSED")
