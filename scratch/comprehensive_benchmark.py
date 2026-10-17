"""
AgriInsight — Comprehensive Honest Benchmark Script
Benchmarks Apriori, FP-Growth, ECLAT, and H-Mine on the actual provided dataset:
Sizes: 1,000, 5,000, 10,000, 50,000, and 100,000 transactions.
Captures exact runtime, memory via tracemalloc, frequent itemset count, and 6-way equivalence verification.
"""

import math
import os
import sys
import time
import tracemalloc

# Ensure project root
sys.path.insert(0, r"c:\Users\Payal\OneDrive\Desktop\AgriInsight")

from core.preprocessing import load_transactions_from_csv
from algorithms.apriori import apriori
from algorithms.fpgrowth import fpgrowth
from algorithms.eclat import eclat
from algorithms.hmine import hmine
from core.verification import verify_all_four

DATASET_FILE = r"c:\Users\Payal\OneDrive\Desktop\AgriInsight\agriinsight_synthetic_100k_transactions.csv"

def run_benchmark():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found.")
        return

    print("=" * 80)
    print("AGRIINSIGHT EMPIRICAL BENCHMARK SUITE (ACTUAL DATASET)")
    print("=" * 80)

    # We evaluate 1k, 5k, 10k, 50k, 100k transactions
    # Note: On 100k records, Apriori candidate generation takes significantly longer,
    # so we cap max_length at 3 or 4 for practical completion.
    sizes = [1000, 5000, 10000, 50000, 100000]
    min_support = 0.05  # 5%
    max_length = 3

    results = []

    for n in sizes:
        print(f"\n---> Benchmarking N = {n:,} transactions (min_support={min_support*100}%, max_length={max_length})...")
        transactions = load_transactions_from_csv(DATASET_FILE, max_rows=n)
        actual_n = len(transactions)
        min_count = math.ceil(min_support * actual_n)

        algs = [
            ("Apriori", apriori),
            ("FP-Growth", fpgrowth),
            ("ECLAT", eclat),
            ("H-Mine", hmine)
        ]

        run_outputs = {}
        row = {"N": actual_n, "Min_Count": min_count}

        for name, fn in algs:
            tracemalloc.start()
            t0 = time.perf_counter()
            try:
                res = fn(transactions, min_support=min_support, max_length=max_length)
                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                peak_mb = peak / (1024.0 * 1024.0)
                run_outputs[name] = res
                count_itemsets = len(res)
                row[f"{name}_Time"] = elapsed
                row[f"{name}_Memory_MB"] = peak_mb
                row[f"{name}_Itemsets"] = count_itemsets
                print(f"    {name:<10}: {elapsed:.4f}s | Peak Mem: {peak_mb:.2f} MB | Itemsets: {count_itemsets}")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                row[f"{name}_Time"] = elapsed
                row[f"{name}_Memory_MB"] = 0.0
                row[f"{name}_Itemsets"] = -1
                print(f"    {name:<10}: FAILED ({e})")
            finally:
                tracemalloc.stop()

        # Equivalence verification
        if len(run_outputs) == 4:
            ver = verify_all_four(run_outputs, min_count=min_count)
            row["Verification"] = ver["overall_status"]
            print(f"    Verification Matrix Status: {ver['overall_status']}")
        else:
            row["Verification"] = "INCOMPLETE"

        results.append(row)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'N':<8} | {'MinCnt':<7} | {'Apriori(s)':<11} | {'FP-Growth(s)':<12} | {'ECLAT(s)':<10} | {'H-Mine(s)':<10} | {'Itemsets':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        print(f"{r['N']:<8} | {r['Min_Count']:<7} | {r.get('Apriori_Time', 0):<11.4f} | {r.get('FP-Growth_Time', 0):<12.4f} | {r.get('ECLAT_Time', 0):<10.4f} | {r.get('H-Mine_Time', 0):<10.4f} | {r.get('FP-Growth_Itemsets', 0):<8} | {r.get('Verification')}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
