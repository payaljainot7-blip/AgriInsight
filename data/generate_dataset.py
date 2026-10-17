"""
AgriInsight — Reproducible Synthetic Agricultural Dataset Generator
Generates realistic agricultural transaction datasets with intentionally planted patterns 
for Association Rule Mining benchmarking.

DISCLAIMER:
This dataset is synthetic and generated strictly for educational, research, and 
algorithm benchmarking purposes. It does not represent observed farm surveys and must not 
be interpreted as genuine agronomic advice or verified field outcomes.

Planted Pattern Dependencies (for mining validation):
1. Rice is strongly correlated with Clay soil, Kharif season, High rainfall, and High water use.
2. Tomato & Vegetables are correlated with Drip irrigation, High potassium, Water-Soluble NPK, and High yield.
3. Wheat is correlated with Rabi season, Alluvial/Loamy soil, Urea+DAP, and Moderate temperature.
4. Chickpea is correlated with Rainfed/Sprinkler, Low water use, and DAP+Potash.
5. Sugarcane is correlated with High nitrogen, Flood irrigation, and Long water use.
"""

import argparse
import csv
import math
import os
import random
import sys
from typing import Dict, List, Tuple


CROP_PROFILES = {
    "Rice": {
        "soils": [("Clay", 0.65), ("Alluvial", 0.25), ("Black", 0.08), ("Sandy", 0.02)],
        "seasons": [("Kharif", 0.90), ("Rabi", 0.10)],
        "irrigation": [("Flood", 0.75), ("Sprinkler", 0.15), ("Drip", 0.05), ("Rainfed", 0.05)],
        "fertilizer": [("Urea+DAP", 0.50), ("NPK_Complex", 0.30), ("Water_Soluble_NPK", 0.10), ("DAP+Potash", 0.10)],
        "water_use_range": (4200.0, 7500.0),
        "rainfall": [("High", 0.70), ("Medium", 0.25), ("Low", 0.05)],
        "n_range": (90.0, 160.0),
        "p_range": (30.0, 60.0),
        "k_range": (30.0, 55.0),
        "temp_range": (24.0, 36.0),
        "humidity_range": (65.0, 95.0),
        "yield_weights": [("High", 0.50), ("Medium", 0.35), ("Low", 0.15)]
    },
    "Wheat": {
        "soils": [("Alluvial", 0.50), ("Clay", 0.25), ("Loamy", 0.20), ("Black", 0.05)],
        "seasons": [("Rabi", 0.95), ("Kharif", 0.05)],
        "irrigation": [("Sprinkler", 0.45), ("Flood", 0.35), ("Drip", 0.15), ("Rainfed", 0.05)],
        "fertilizer": [("Urea+DAP", 0.60), ("NPK_Complex", 0.25), ("Urea+Potash", 0.15)],
        "water_use_range": (2600.0, 4800.0),
        "rainfall": [("Medium", 0.55), ("Low", 0.35), ("High", 0.10)],
        "n_range": (80.0, 140.0),
        "p_range": (40.0, 70.0),
        "k_range": (25.0, 50.0),
        "temp_range": (12.0, 26.0),
        "humidity_range": (40.0, 75.0),
        "yield_weights": [("High", 0.45), ("Medium", 0.40), ("Low", 0.15)]
    },
    "Tomato": {
        "soils": [("Loamy", 0.45), ("Sandy", 0.30), ("Black", 0.15), ("Clay", 0.10)],
        "seasons": [("Rabi", 0.55), ("Kharif", 0.35), ("Zaid", 0.10)],
        "irrigation": [("Drip", 0.75), ("Sprinkler", 0.15), ("Flood", 0.10)],
        "fertilizer": [("Water_Soluble_NPK", 0.60), ("NPK_Complex", 0.25), ("Urea+DAP", 0.15)],
        "water_use_range": (2000.0, 3800.0),
        "rainfall": [("Medium", 0.50), ("Low", 0.35), ("High", 0.15)],
        "n_range": (70.0, 120.0),
        "p_range": (35.0, 65.0),
        "k_range": (60.0, 120.0),
        "temp_range": (18.0, 32.0),
        "humidity_range": (50.0, 85.0),
        "yield_weights": [("High", 0.55), ("Medium", 0.30), ("Low", 0.15)]
    },
    "Maize": {
        "soils": [("Black", 0.40), ("Loamy", 0.35), ("Alluvial", 0.15), ("Sandy", 0.10)],
        "seasons": [("Kharif", 0.75), ("Rabi", 0.25)],
        "irrigation": [("Flood", 0.45), ("Rainfed", 0.35), ("Sprinkler", 0.20)],
        "fertilizer": [("Urea+DAP", 0.50), ("NPK_Complex", 0.35), ("DAP+Potash", 0.15)],
        "water_use_range": (2800.0, 5200.0),
        "rainfall": [("Medium", 0.60), ("Low", 0.25), ("High", 0.15)],
        "n_range": (90.0, 150.0),
        "p_range": (40.0, 75.0),
        "k_range": (30.0, 60.0),
        "temp_range": (20.0, 34.0),
        "humidity_range": (45.0, 80.0),
        "yield_weights": [("High", 0.35), ("Medium", 0.50), ("Low", 0.15)]
    },
    "Chickpea": {
        "soils": [("Black", 0.55), ("Loamy", 0.25), ("Sandy", 0.20)],
        "seasons": [("Rabi", 0.90), ("Kharif", 0.10)],
        "irrigation": [("Rainfed", 0.60), ("Sprinkler", 0.30), ("Drip", 0.10)],
        "fertilizer": [("DAP+Potash", 0.65), ("NPK_Complex", 0.25), ("Urea+DAP", 0.10)],
        "water_use_range": (1200.0, 2400.0),
        "rainfall": [("Low", 0.65), ("Medium", 0.30), ("High", 0.05)],
        "n_range": (20.0, 50.0),  # Legume fixates N
        "p_range": (45.0, 80.0),
        "k_range": (35.0, 70.0),
        "temp_range": (14.0, 28.0),
        "humidity_range": (35.0, 65.0),
        "yield_weights": [("Medium", 0.50), ("High", 0.35), ("Low", 0.15)]
    },
    "Sugarcane": {
        "soils": [("Alluvial", 0.45), ("Clay", 0.35), ("Black", 0.20)],
        "seasons": [("Kharif", 0.85), ("Rabi", 0.15)],
        "irrigation": [("Flood", 0.60), ("Drip", 0.30), ("Sprinkler", 0.10)],
        "fertilizer": [("NPK_Complex", 0.45), ("Urea+DAP", 0.40), ("Water_Soluble_NPK", 0.15)],
        "water_use_range": (5500.0, 9200.0),
        "rainfall": [("Medium", 0.50), ("High", 0.40), ("Low", 0.10)],
        "n_range": (120.0, 220.0),
        "p_range": (40.0, 85.0),
        "k_range": (50.0, 110.0),
        "temp_range": (24.0, 38.0),
        "humidity_range": (55.0, 90.0),
        "yield_weights": [("High", 0.45), ("Medium", 0.45), ("Low", 0.10)]
    },
    "Soybean": {
        "soils": [("Black", 0.60), ("Loamy", 0.25), ("Clay", 0.15)],
        "seasons": [("Kharif", 0.95), ("Rabi", 0.05)],
        "irrigation": [("Rainfed", 0.55), ("Sprinkler", 0.30), ("Drip", 0.15)],
        "fertilizer": [("DAP+Potash", 0.55), ("NPK_Complex", 0.35), ("Urea+DAP", 0.10)],
        "water_use_range": (2000.0, 3600.0),
        "rainfall": [("Medium", 0.60), ("High", 0.25), ("Low", 0.15)],
        "n_range": (25.0, 60.0),
        "p_range": (50.0, 85.0),
        "k_range": (35.0, 70.0),
        "temp_range": (22.0, 34.0),
        "humidity_range": (55.0, 85.0),
        "yield_weights": [("Medium", 0.45), ("High", 0.40), ("Low", 0.15)]
    },
    "Groundnut": {
        "soils": [("Sandy", 0.50), ("Loamy", 0.30), ("Black", 0.20)],
        "seasons": [("Kharif", 0.70), ("Rabi", 0.30)],
        "irrigation": [("Rainfed", 0.50), ("Sprinkler", 0.35), ("Flood", 0.15)],
        "fertilizer": [("NPK_Complex", 0.50), ("DAP+Potash", 0.35), ("Urea+Potash", 0.15)],
        "water_use_range": (2200.0, 4000.0),
        "rainfall": [("Low", 0.50), ("Medium", 0.40), ("High", 0.10)],
        "n_range": (25.0, 55.0),
        "p_range": (40.0, 75.0),
        "k_range": (40.0, 80.0),
        "temp_range": (24.0, 35.0),
        "humidity_range": (40.0, 70.0),
        "yield_weights": [("Medium", 0.50), ("High", 0.35), ("Low", 0.15)]
    },
    "Potato": {
        "soils": [("Loamy", 0.50), ("Sandy", 0.35), ("Alluvial", 0.15)],
        "seasons": [("Rabi", 0.90), ("Kharif", 0.10)],
        "irrigation": [("Drip", 0.55), ("Sprinkler", 0.30), ("Flood", 0.15)],
        "fertilizer": [("Urea+DAP", 0.45), ("NPK_Complex", 0.35), ("Water_Soluble_NPK", 0.20)],
        "water_use_range": (2400.0, 4200.0),
        "rainfall": [("Medium", 0.50), ("Low", 0.40), ("High", 0.10)],
        "n_range": (100.0, 160.0),
        "p_range": (50.0, 90.0),
        "k_range": (80.0, 140.0),
        "temp_range": (12.0, 24.0),
        "humidity_range": (55.0, 80.0),
        "yield_weights": [("High", 0.45), ("Medium", 0.40), ("Low", 0.15)]
    }
}

REGIONS = [
    "Madhya Pradesh", "North Maharashtra", "Vidarbha", "Marathwada",
    "Western Maharashtra", "Punjab", "Karnataka", "Konkan"
]


def weighted_choice(choices: List[Tuple[str, float]], rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for item, weight in choices:
        cumulative += weight
        if r <= cumulative:
            return item
    return choices[-1][0]


def discretize_num(val: float, bins: List[Tuple[str, float, float]]) -> str:
    for label, lower, upper in bins:
        if lower <= val < upper:
            return label
        if math.isinf(upper) and val >= lower:
            return label
    return bins[-1][0]


def generate_agricultural_dataset(
    num_records: int = 100000,
    seed: int = 42,
    output_dir: str = "data",
    file_prefix: str = "agriinsight_synthetic_100k"
) -> Dict[str, str]:
    """
    Generates synthetic agricultural raw, transaction, and categorical CSV files.
    """
    rng = random.Random(seed)
    crops = list(CROP_PROFILES.keys())

    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"{file_prefix}_raw.csv")
    tx_path = os.path.join(output_dir, f"{file_prefix}_transactions.csv")
    cat_path = os.path.join(output_dir, f"{file_prefix}.csv")

    raw_headers = [
        "transaction_id", "region", "season", "crop", "soil_type", "irrigation_type",
        "fertilizer_type", "pesticide_intensity", "rainfall_category", "nitrogen_kg_ha",
        "phosphorus_kg_ha", "potassium_kg_ha", "soil_ph", "temperature_c",
        "humidity_percent", "fertilizer_kg_ha", "water_use_m3_ha", "yield_category"
    ]

    cat_headers = [
        "transaction_id", "Region", "Season", "Crop", "Soil", "Irrigation",
        "Fertilizer", "Pesticide", "Rainfall", "Nitrogen", "Phosphorus",
        "Potassium", "Soil_pH", "Temperature", "Humidity", "Fertilizer_Quantity",
        "Water_Use", "Yield"
    ]

    # Numeric bin specifications matching discovery
    n_bins = [("Low", float("-inf"), 80.0), ("Medium", 80.0, 130.0), ("High", 130.0, float("inf"))]
    p_bins = [("Low", float("-inf"), 35.0), ("Medium", 35.0, 65.0), ("High", 65.0, float("inf"))]
    k_bins = [("Low", float("-inf"), 35.0), ("Medium", 35.0, 75.0), ("High", 75.0, float("inf"))]
    ph_bins = [("Acidic", float("-inf"), 6.2), ("Neutral", 6.2, 7.5), ("Alkaline", 7.5, float("inf"))]
    t_bins = [("Low", float("-inf"), 20.0), ("Moderate", 20.0, 30.0), ("High", 30.0, float("inf"))]
    h_bins = [("Low", float("-inf"), 45.0), ("Moderate", 45.0, 70.0), ("High", 70.0, float("inf"))]
    fert_bins = [("Low", float("-inf"), 120.0), ("Medium", 120.0, 180.0), ("High", 180.0, float("inf"))]
    water_bins = [("Low", float("-inf"), 2500.0), ("Medium", 2500.0, 4500.0), ("High", 4500.0, float("inf"))]

    with open(raw_path, "w", newline="", encoding="utf-8") as f_raw, \
         open(tx_path, "w", newline="", encoding="utf-8") as f_tx, \
         open(cat_path, "w", newline="", encoding="utf-8") as f_cat:

        w_raw = csv.writer(f_raw)
        w_tx = csv.writer(f_tx)
        w_cat = csv.writer(f_cat)

        w_raw.writerow(raw_headers)
        w_tx.writerow(["transaction_id", "items"])
        w_cat.writerow(cat_headers)

        for i in range(1, num_records + 1):
            tid = f"AGR{i:06d}"
            crop = rng.choice(crops)
            prof = CROP_PROFILES[crop]

            region = rng.choice(REGIONS)
            soil = weighted_choice(prof["soils"], rng)
            season = weighted_choice(prof["seasons"], rng)
            irrigation = weighted_choice(prof["irrigation"], rng)
            fertilizer = weighted_choice(prof["fertilizer"], rng)
            rainfall = weighted_choice(prof["rainfall"], rng)
            yield_cat = weighted_choice(prof["yield_weights"], rng)
            pesticide = weighted_choice([("Low", 0.40), ("Medium", 0.45), ("High", 0.15)], rng)

            # Continuous numeric features generated with Gaussian perturbation
            n_val = round(rng.uniform(*prof["n_range"]) + rng.gauss(0, 5), 1)
            p_val = round(rng.uniform(*prof["p_range"]) + rng.gauss(0, 4), 1)
            k_val = round(rng.uniform(*prof["k_range"]) + rng.gauss(0, 5), 1)
            ph_val = round(rng.gauss(6.8, 0.6), 2)
            temp_val = round(rng.uniform(*prof["temp_range"]) + rng.gauss(0, 2), 1)
            humidity_val = round(rng.uniform(*prof["humidity_range"]) + rng.gauss(0, 3), 1)
            water_val = round(rng.uniform(*prof["water_use_range"]) + rng.gauss(0, 150), 1)
            fert_val = round((n_val + p_val + k_val) * rng.uniform(0.7, 1.1), 1)

            # Raw record
            w_raw.writerow([
                tid, region, season, crop, soil, irrigation, fertilizer,
                pesticide, rainfall, n_val, p_val, k_val, ph_val,
                temp_val, humidity_val, fert_val, water_val, yield_cat
            ])

            # Discretize continuous columns into items
            n_cat = discretize_num(n_val, n_bins)
            p_cat = discretize_num(p_val, p_bins)
            k_cat = discretize_num(k_val, k_bins)
            ph_cat = discretize_num(ph_val, ph_bins)
            temp_cat = discretize_num(temp_val, t_bins)
            humidity_cat = discretize_num(humidity_val, h_bins)
            fert_cat = discretize_num(fert_val, fert_bins)
            water_cat = discretize_num(water_val, water_bins)

            # Categorical row
            w_cat.writerow([
                tid, region, season, crop, soil, irrigation, fertilizer,
                pesticide, rainfall, n_cat, p_cat, k_cat, ph_cat,
                temp_cat, humidity_cat, fert_cat, water_cat, yield_cat
            ])

            # Transaction items (semicolon separated)
            items = [
                f"Region={region}",
                f"Season={season}",
                f"Crop={crop}",
                f"Soil={soil}",
                f"Irrigation={irrigation}",
                f"Fertilizer={fertilizer}",
                f"Pesticide={pesticide}",
                f"Rainfall={rainfall}",
                f"Nitrogen={n_cat}",
                f"Phosphorus={p_cat}",
                f"Potassium={k_cat}",
                f"Soil_pH={ph_cat}",
                f"Temperature={temp_cat}",
                f"Humidity={humidity_cat}",
                f"Fertilizer_Quantity={fert_cat}",
                f"Water_Use={water_cat}",
                f"Yield={yield_cat}"
            ]
            w_tx.writerow([tid, ";".join(items)])

    return {
        "raw_path": raw_path,
        "transaction_path": tx_path,
        "categorical_path": cat_path,
        "num_records": num_records,
        "seed": seed
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic agricultural datasets for AgriInsight.")
    parser.add_argument("--num-records", type=int, default=100000, help="Number of records to generate (e.g. 100000, 250000, 500000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--prefix", type=str, default="agriinsight_synthetic", help="File prefix")

    args = parser.parse_args()
    prefix = f"{args.prefix}_{args.num_records // 1000}k" if args.num_records >= 1000 else f"{args.prefix}_{args.num_records}"

    print(f"Generating synthetic agricultural dataset with {args.num_records:,} records (seed={args.seed})...")
    res = generate_agricultural_dataset(
        num_records=args.num_records,
        seed=args.seed,
        output_dir=args.output_dir,
        file_prefix=prefix
    )
    print("Generation complete:")
    print(f"  Raw CSV:         {res['raw_path']}")
    print(f"  Transaction CSV: {res['transaction_path']}")
    print(f"  Categorical CSV: {res['categorical_path']}")


if __name__ == "__main__":
    main()
