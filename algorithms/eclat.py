"""
AgriInsight — ECLAT Algorithm Implementation
Academic, genuine implementation of the ECLAT (Equivalence CLASS Transformation) algorithm (Zaki, 2000).

Key Steps:
1. Invert horizontal transaction database into vertical TID-sets (item -> set of transaction IDs).
2. Compute support count of each 1-item as |TID-set|. Filter by min_count.
3. Sort items deterministically.
4. Recursively generate candidate itemsets via TID-set intersections:
   - For items i and j sharing prefix P, the TID-set of P U {i, j} is TID(P U {i}) ∩ TID(P U {j}).
   - Exact support count is |TID-set|.
   - Retain candidates with support count >= min_count.
5. Terminate recursion when no candidates qualify or max_length is reached.
"""

import math
from typing import Sequence, Collection, Dict, FrozenSet, Optional, List, Set, Tuple


def eclat(
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None
) -> Dict[FrozenSet[str], int]:
    """
    Mine all frequent itemsets using the ECLAT algorithm.

    Parameters
    ----------
    transactions : Sequence[Collection[str]]
        A sequence of transaction records.
    min_support : float
        Minimum support proportion in (0.0, 1.0].
    max_length : Optional[int]
        Maximum itemset length.

    Returns
    -------
    Dict[FrozenSet[str], int]
        Mapping from frequent itemset to exact integer support count.
    """
    n_trans = len(transactions)
    if n_trans == 0:
        return {}

    if not (0.0 < min_support <= 1.0):
        raise ValueError(f"min_support must be in (0.0, 1.0], got {min_support}")

    if max_length is not None and max_length < 1:
        return {}

    min_count = math.ceil(min_support * n_trans)

    # Step 1: Vertical database creation (Item -> Set of Transaction IDs)
    vertical_db: Dict[str, Set[int]] = {}
    for tid, t in enumerate(transactions):
        if not t:
            continue
        for item in set(t):
            if item not in vertical_db:
                vertical_db[item] = set()
            vertical_db[item].add(tid)

    # Step 2: Filter frequent 1-items
    frequent_1: List[Tuple[str, Set[int]]] = []
    frequent_itemsets: Dict[FrozenSet[str], int] = {}

    for item, tids in vertical_db.items():
        count = len(tids)
        if count >= min_count:
            frequent_1.append((item, tids))
            frequent_itemsets[frozenset([item])] = count

    if not frequent_1 or max_length == 1:
        return frequent_itemsets

    # Deterministic ordering: descending by support count, then ascending by item string
    frequent_1.sort(key=lambda x: (-len(x[1]), x[0]))

    def eclat_recurse(
        prefix: Tuple[str, ...],
        items_with_tids: List[Tuple[str, Set[int]]]
    ):
        """
        Recursive depth-first extension of prefix itemsets.
        """
        n_items = len(items_with_tids)
        for i in range(n_items):
            item_i, tids_i = items_with_tids[i]
            new_prefix = prefix + (item_i,)
            
            # If max_length reached for extensions, don't generate deeper sets
            if max_length is not None and len(new_prefix) >= max_length:
                continue

            # Candidate extensions for new_prefix
            next_items: List[Tuple[str, Set[int]]] = []
            for j in range(i + 1, n_items):
                item_j, tids_j = items_with_tids[j]
                
                # Fast set intersection
                intersected_tids = tids_i & tids_j
                count = len(intersected_tids)

                if count >= min_count:
                    ext_itemset = new_prefix + (item_j,)
                    frequent_itemsets[frozenset(ext_itemset)] = count
                    next_items.append((item_j, intersected_tids))

            if next_items:
                eclat_recurse(new_prefix, next_items)

    # Convert initial frequent 1-items to recursive format
    eclat_recurse((), frequent_1)

    return frequent_itemsets
