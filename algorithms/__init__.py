"""
AgriInsight Algorithms Package
Contains genuine, independent implementations of:
- Apriori (level-wise candidate generation and pruning)
- FP-Growth (FP-Tree with conditional pattern bases)
- ECLAT (vertical transaction ID set intersections)
- H-Mine (hyper-linked transaction structures)

All algorithms adhere to the same signature:
mine_frequent_itemsets(
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None
) -> Dict[frozenset[str], int]
"""

import math
from typing import Sequence, Collection, Dict, FrozenSet, Optional


def compute_min_support_count(num_transactions: int, min_support: float) -> int:
    """
    Computes integer minimum support count using ceil(min_support * N).
    
    Parameters
    ----------
    num_transactions : int
        Total number of transactions in the dataset.
    min_support : float
        Minimum support proportion in (0.0, 1.0].
        
    Returns
    -------
    int
        Exact minimum support count required for an itemset to be frequent.
    """
    if num_transactions <= 0:
        return 0
    if not (0.0 < min_support <= 1.0):
        raise ValueError(f"min_support must be in (0.0, 1.0], got {min_support}")
    return math.ceil(min_support * num_transactions)
