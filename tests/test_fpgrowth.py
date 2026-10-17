"""
Unit tests for FP-Growth Algorithm.
Tests against manually derived ground truth, conditional tree edge cases, and independent recount.
"""

try:
    import pytest
except ImportError:
    class PytestShim:
        @staticmethod
        def fixture(f):
            return f
        @staticmethod
        def raises(exc):
            class Ctx:
                def __enter__(self): return self
                def __exit__(self, et, ev, tb):
                    if et is None:
                        raise AssertionError(f"Expected exception {exc} but none raised")
                    return issubclass(et, exc)
            return Ctx()
    pytest = PytestShim()

from algorithms.fpgrowth import fpgrowth
from core.verification import recalculate_exact_counts


def get_manual_dataset():
    return [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
        ["A", "B", "C"]
    ]


@pytest.fixture
def manual_dataset():
    return get_manual_dataset()


def test_fpgrowth_manual_dataset():
    ds = get_manual_dataset()
    expected = {
        frozenset(["A"]): 4,
        frozenset(["B"]): 4,
        frozenset(["C"]): 4,
        frozenset(["A", "B"]): 3,
        frozenset(["A", "C"]): 3,
        frozenset(["B", "C"]): 3,
        frozenset(["A", "B", "C"]): 2,
    }

    result = fpgrowth(ds, min_support=0.40)
    assert result == expected
    assert len(result) == 7


def test_fpgrowth_independent_recount():
    ds = get_manual_dataset()
    result = fpgrowth(ds, min_support=0.40)
    recalculated = recalculate_exact_counts(ds, list(result.keys()))
    for itemset, count in result.items():
        canonical_key = tuple(sorted(itemset))
        assert recalculated[canonical_key] == count


def test_fpgrowth_empty_and_edge_cases():
    assert fpgrowth([], 0.5) == {}
    assert fpgrowth([[], []], 0.5) == {}
    assert fpgrowth([["A"], ["A"], ["A"]], 0.5) == {frozenset(["A"]): 3}
    assert fpgrowth([["A"], ["B"], ["C"]], 0.8) == {}


def test_fpgrowth_duplicate_items():
    transactions = [["A", "A", "B"], ["A", "B", "B", "C"], ["A", "C"]]
    result = fpgrowth(transactions, min_support=0.5)
    expected = {
        frozenset(["A"]): 3,
        frozenset(["B"]): 2,
        frozenset(["C"]): 2,
        frozenset(["A", "B"]): 2,
        frozenset(["A", "C"]): 2,
    }
    assert result == expected


def test_fpgrowth_max_length():
    ds = get_manual_dataset()
    result_len1 = fpgrowth(ds, min_support=0.40, max_length=1)
    assert len(result_len1) == 3
    assert all(len(k) == 1 for k in result_len1.keys())

    result_len2 = fpgrowth(ds, min_support=0.40, max_length=2)
    assert len(result_len2) == 6
    assert all(len(k) <= 2 for k in result_len2.keys())


def test_fpgrowth_invalid_support():
    with pytest.raises(ValueError):
        fpgrowth([["A"]], min_support=0.0)
    with pytest.raises(ValueError):
        fpgrowth([["A"]], min_support=1.5)


if __name__ == "__main__":
    test_fpgrowth_manual_dataset()
    test_fpgrowth_independent_recount()
    test_fpgrowth_empty_and_edge_cases()
    test_fpgrowth_duplicate_items()
    test_fpgrowth_max_length()
    test_fpgrowth_invalid_support()
    print("test_fpgrowth.py: ALL TESTS PASSED")
