"""
AgriInsight — Benchmarking and Performance Evaluation Module
Measures execution time, memory usage, itemset scalability, and correctness 
across Apriori, FP-Growth, ECLAT, and H-Mine.

Performance Metrics:
- High-precision execution time using time.perf_counter()
- Peak memory allocation using Python's standard tracemalloc module
- Support count consistency and verification check
- Configurable dataset subsets (e.g. 1,000, 5,000, 10,000, 50,000, 100,000)
- Configurable minimum-support sweeps (e.g. 1%, 3%, 5%, 10%)
"""

import time
import tracemalloc
import math
from typing import Sequence, Collection, Dict, FrozenSet, List, Any, Optional, Callable

from algorithms.apriori import apriori
from algorithms.fpgrowth import fpgrowth
from algorithms.eclat import eclat
from algorithms.hmine import hmine
from core.verification import compare_two_algorithms

ALGORITHMS: Dict[str, Callable] = {
    "Apriori": apriori,
    "FP-Growth": fpgrowth,
    "ECLAT": eclat,
    "H-Mine": hmine
}


def run_single_benchmark(
    algorithm_name: str,
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None,
    track_memory: bool = True
) -> Dict[str, Any]:
    """
    Executes a single algorithm run and records execution metrics.
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(ALGORITHMS.keys())}")

    fn = ALGORITHMS[algorithm_name]
    num_trans = len(transactions)
    min_count = math.ceil(min_support * num_trans) if num_trans > 0 else 0

    peak_memory_mb = 0.0
    if track_memory:
        tracemalloc.start()

    start_time = time.perf_counter()
    try:
        frequent_itemsets = fn(transactions, min_support=min_support, max_length=max_length)
        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        status = "SUCCESS"
        error_msg = None
    except Exception as e:
        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        status = "FAILED"
        error_msg = str(e)
        frequent_itemsets = {}
    finally:
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak / (1024.0 * 1024.0)

    return {
        "algorithm": algorithm_name,
        "dataset_size": num_trans,
        "min_support": min_support,
        "min_count": min_count,
        "max_length": max_length,
        "execution_time_sec": elapsed_sec,
        "peak_memory_mb": peak_memory_mb,
        "frequent_itemsets_count": len(frequent_itemsets),
        "status": status,
        "error_message": error_msg,
        "frequent_itemsets": frequent_itemsets
    }


def benchmark_all_algorithms(
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None,
    trials: int = 1,
    selected_algorithms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Benchmarks multiple algorithms on the exact same dataset and verifies correctness.
    """
    if selected_algorithms is None:
        selected_algorithms = ["Apriori", "FP-Growth", "ECLAT", "H-Mine"]

    results_summary: List[Dict[str, Any]] = []
    itemsets_by_alg: Dict[str, Dict[FrozenSet[str], int]] = {}

    for alg_name in selected_algorithms:
        trial_times: List[float] = []
        last_run: Optional[Dict[str, Any]] = None

        for _ in range(trials):
            run_data = run_single_benchmark(
                alg_name,
                transactions,
                min_support=min_support,
                max_length=max_length,
                track_memory=True
            )
            trial_times.append(run_data["execution_time_sec"])
            last_run = run_data

        if last_run and last_run["status"] == "SUCCESS":
            itemsets_by_alg[alg_name] = last_run["frequent_itemsets"]
            median_time = sorted(trial_times)[len(trial_times) // 2]
            results_summary.append({
                "Algorithm": alg_name,
                "Dataset Size": len(transactions),
                "Min Support (%)": f"{min_support * 100:.1f}%",
                "Min Count": last_run["min_count"],
                "Itemsets Found": last_run["frequent_itemsets_count"],
                "Time (s)": round(median_time, 4),
                "Peak Memory (MB)": round(last_run["peak_memory_mb"], 2),
                "Status": "PASS"
            })
        else:
            results_summary.append({
                "Algorithm": alg_name,
                "Dataset Size": len(transactions),
                "Min Support (%)": f"{min_support * 100:.1f}%",
                "Min Count": 0,
                "Itemsets Found": 0,
                "Time (s)": round(last_run["execution_time_sec"] if last_run else 0.0, 4),
                "Peak Memory (MB)": 0.0,
                "Status": f"ERROR: {last_run['error_message'] if last_run else 'Unknown'}"
            })

    # Cross-verification
    equivalence_matrix: List[Dict[str, Any]] = []
    algs = list(itemsets_by_alg.keys())
    for i in range(len(algs)):
        for j in range(i + 1, len(algs)):
            cmp = compare_two_algorithms(algs[i], itemsets_by_alg[algs[i]], algs[j], itemsets_by_alg[algs[j]])
            equivalence_matrix.append(cmp)

    return {
        "summary_table": results_summary,
        "itemsets_by_algorithm": itemsets_by_alg,
        "equivalence_matrix": equivalence_matrix,
        "min_support": min_support,
        "dataset_size": len(transactions)
    }
