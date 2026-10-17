"""
AgriInsight — Preprocessing and Transaction Transformation Module
Converts raw agricultural records into clean, validated, transaction-oriented datasets 
suitable for association rule mining.

Key Functions:
- Load and parse raw agricultural CSV datasets.
- Load pre-formatted transaction CSV datasets.
- Discretize continuous agricultural metrics (pH, nutrients, temperature, water use) into meaningful bins.
- Remove identifiers (transaction_id, farm_id) to prevent data leakage.
- Deduplicate items within transactions (strict set semantics).
- Flexible attribute selection and configurable bin boundaries.
- Compute comprehensive preprocessing summary statistics.
"""

import csv
import io
import math
from typing import List, Dict, Set, Tuple, Any, Optional, Union

# Standard academic agricultural bin definitions
DEFAULT_NUMERIC_BINS = {
    "nitrogen_kg_ha": {
        "attr_name": "Nitrogen",
        "bins": [
            ("Low", float("-inf"), 80.0),
            ("Medium", 80.0, 130.0),
            ("High", 130.0, float("inf")),
        ]
    },
    "phosphorus_kg_ha": {
        "attr_name": "Phosphorus",
        "bins": [
            ("Low", float("-inf"), 35.0),
            ("Medium", 35.0, 65.0),
            ("High", 65.0, float("inf")),
        ]
    },
    "potassium_kg_ha": {
        "attr_name": "Potassium",
        "bins": [
            ("Low", float("-inf"), 35.0),
            ("Medium", 35.0, 75.0),
            ("High", 75.0, float("inf")),
        ]
    },
    "soil_ph": {
        "attr_name": "Soil_pH",
        "bins": [
            ("Acidic", float("-inf"), 6.2),
            ("Neutral", 6.2, 7.5),
            ("Alkaline", 7.5, float("inf")),
        ]
    },
    "temperature_c": {
        "attr_name": "Temperature",
        "bins": [
            ("Low", float("-inf"), 20.0),
            ("Moderate", 20.0, 30.0),
            ("High", 30.0, float("inf")),
        ]
    },
    "humidity_percent": {
        "attr_name": "Humidity",
        "bins": [
            ("Low", float("-inf"), 45.0),
            ("Moderate", 45.0, 70.0),
            ("High", 70.0, float("inf")),
        ]
    },
    "fertilizer_kg_ha": {
        "attr_name": "Fertilizer_Quantity",
        "bins": [
            ("Low", float("-inf"), 120.0),
            ("Medium", 120.0, 180.0),
            ("High", 180.0, float("inf")),
        ]
    },
    "water_use_m3_ha": {
        "attr_name": "Water_Use",
        "bins": [
            ("Low", float("-inf"), 2500.0),
            ("Medium", 2500.0, 4500.0),
            ("High", 4500.0, float("inf")),
        ]
    }
}

CATEGORICAL_COLUMN_MAP = {
    "region": "Region",
    "season": "Season",
    "crop": "Crop",
    "soil_type": "Soil",
    "irrigation_type": "Irrigation",
    "fertilizer_type": "Fertilizer",
    "pesticide_intensity": "Pesticide",
    "rainfall_category": "Rainfall",
    "yield_category": "Yield",
    "farming_method": "Farming_Method",
    "soil_moisture_category": "Soil_Moisture",
    "drainage_category": "Drainage"
}

IDENTIFIER_COLUMNS = {"transaction_id", "farm_id", "id", "index"}


def discretize_value(val: float, bin_definitions: List[Tuple[str, float, float]]) -> str:
    """Discretizes a numeric value according to intervals [lower, upper)."""
    for label, lower, upper in bin_definitions:
        if lower <= val < upper:
            return label
        # Upper bound inclusive for maximum endpoint
        if math.isinf(upper) and val >= lower:
            return label
    return bin_definitions[-1][0]


def load_transactions_from_csv(
    filepath_or_buffer: Union[str, io.StringIO, io.TextIOBase],
    max_rows: Optional[int] = None
) -> List[List[str]]:
    """
    Loads transactions directly from a transaction CSV file (columns: transaction_id, items).
    Each row in 'items' is semicolon-separated (Item1;Item2;...).
    """
    transactions: List[List[str]] = []
    
    if isinstance(filepath_or_buffer, str):
        f = open(filepath_or_buffer, mode="r", encoding="utf-8", errors="replace")
        should_close = True
    else:
        f = filepath_or_buffer
        should_close = False

    try:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            items_str = row.get("items", "")
            if not items_str:
                continue
            # Split items on semicolon, strip, remove empty, and deduplicate preserving order
            raw_items = [it.strip() for it in items_str.split(";") if it.strip()]
            unique_items = list(dict.fromkeys(raw_items))
            if unique_items:
                transactions.append(unique_items)
    finally:
        if should_close:
            f.close()

    return transactions


def load_raw_dataset_records(
    filepath_or_buffer: Union[str, io.StringIO, io.TextIOBase],
    max_rows: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads raw CSV records and computes missing values and data overview.
    """
    records: List[Dict[str, Any]] = []
    missing_counts: Dict[str, int] = {}
    total_rows = 0

    if isinstance(filepath_or_buffer, str):
        f = open(filepath_or_buffer, mode="r", encoding="utf-8", errors="replace")
        should_close = True
    else:
        f = filepath_or_buffer
        should_close = False

    try:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in fieldnames:
            missing_counts[col] = 0

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            total_rows += 1
            for col, val in row.items():
                if val is None or val.strip() == "" or val.lower() in ("nan", "null", "none"):
                    missing_counts[col] = missing_counts.get(col, 0) + 1
            records.append(row)
    finally:
        if should_close:
            f.close()

    summary = {
        "total_records_loaded": len(records),
        "total_columns": len(fieldnames),
        "fieldnames": fieldnames,
        "missing_counts": missing_counts
    }
    return records, summary


def transform_raw_to_transactions(
    records: List[Dict[str, Any]],
    selected_attributes: Optional[Set[str]] = None,
    numeric_bins: Optional[Dict[str, Any]] = None,
    drop_missing: bool = True
) -> List[List[str]]:
    """
    Transforms raw agricultural records into transaction item lists.
    
    Handles:
    - Exclusion of identifiers (e.g. transaction_id, farm_id)
    - Dynamic binning of numerical features
    - Categorical formatting (Attribute=Value)
    - Itemset deduplication within each transaction
    """
    if numeric_bins is None:
        numeric_bins = DEFAULT_NUMERIC_BINS

    transactions: List[List[str]] = []

    for row in records:
        items: List[str] = []
        skip_record = False

        for col, val in row.items():
            col_lower = col.lower()
            if col_lower in IDENTIFIER_COLUMNS:
                continue

            if selected_attributes is not None and col not in selected_attributes and col_lower not in selected_attributes:
                continue

            if val is None or val.strip() == "" or val.lower() in ("nan", "null", "none"):
                if drop_missing:
                    skip_record = True
                    break
                continue

            # Check if column is a continuous numeric column configured for binning
            if col in numeric_bins or col_lower in numeric_bins:
                cfg = numeric_bins.get(col, numeric_bins.get(col_lower))
                try:
                    num_val = float(val)
                    cat_label = discretize_value(num_val, cfg["bins"])
                    items.append(f"{cfg['attr_name']}={cat_label}")
                except ValueError:
                    if drop_missing:
                        skip_record = True
                        break
            else:
                # Standard categorical attribute
                attr_label = CATEGORICAL_COLUMN_MAP.get(col_lower, col.replace("_", " ").title().replace(" ", "_"))
                clean_val = val.strip()
                items.append(f"{attr_label}={clean_val}")

        if not skip_record and items:
            # Enforce unique items per transaction (set semantics)
            unique_items = list(dict.fromkeys(items))
            transactions.append(unique_items)

    return transactions


def get_transaction_statistics(transactions: List[List[str]]) -> Dict[str, Any]:
    """
    Computes summary metrics on a transaction dataset:
    - Number of transactions
    - Number of distinct items
    - Length distribution (min, max, mean, median)
    - Item frequency distribution
    """
    n = len(transactions)
    if n == 0:
        return {
            "num_transactions": 0,
            "num_distinct_items": 0,
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0.0,
            "median_length": 0.0,
            "item_frequencies": {}
        }

    lengths = [len(t) for t in transactions]
    item_counts: Dict[str, int] = {}

    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    sorted_lengths = sorted(lengths)
    median_len = (
        sorted_lengths[n // 2]
        if n % 2 != 0
        else (sorted_lengths[(n // 2) - 1] + sorted_lengths[n // 2]) / 2.0
    )

    return {
        "num_transactions": n,
        "num_distinct_items": len(item_counts),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / float(n),
        "median_length": median_len,
        "item_frequencies": item_counts
    }
