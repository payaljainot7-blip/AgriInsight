"""
AgriInsight Universal Test Runner
Executes the comprehensive test suite across all algorithms, verification, metrics,
and data preprocessing modules.
"""

import os
import sys
import time

def run_all_tests():
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    test_modules = [
        "tests.test_apriori",
        "tests.test_fpgrowth",
        "tests.test_eclat",
        "tests.test_hmine",
        "tests.test_algorithm_equivalence",
        "tests.test_association_rules",
        "tests.test_preprocessing",
        "tests.test_dataset_generator"
    ]

    print("=" * 70)
    print("AGRIINSIGHT TEST SUITE EXECUTION")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    start_total = time.perf_counter()

    for mod_name in test_modules:
        print(f"\nRunning {mod_name}...")
        try:
            mod = __import__(mod_name, fromlist=["*"])
            # Discover and execute all functions starting with test_
            test_funcs = [
                getattr(mod, name) for name in dir(mod)
                if name.startswith("test_") and callable(getattr(mod, name))
            ]
            for fn in test_funcs:
                try:
                    fn()
                    print(f"  [PASS] {fn.__name__}")
                    total_passed += 1
                except Exception as e:
                    print(f"  [FAIL] {fn.__name__}: {e}")
                    total_failed += 1
        except Exception as e:
            print(f"  [ERROR] Could not import {mod_name}: {e}")
            total_failed += 1

    duration = time.perf_counter() - start_total
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {total_passed} Passed, {total_failed} Failed in {duration:.3f}s")
    print("=" * 70)

    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
