"""
AgriInsight — Apriori Algorithm Implementation
Academic, genuine implementation of the Apriori association rule mining algorithm (Agrawal & Srikant, 1994).

Key Steps:
1. Frequent 1-itemset generation (L1) by scanning transactions.
2. Candidate generation (Ck) from L(k-1) using lexicographical join of matching prefixes.
3. Apriori pruning: verify all (k-1)-subsets of each candidate exist in L(k-1).
4. Transaction subset counting: count candidate occurrences across filtered transactions.
5. Filter by minimum support count to produce Lk.
6. Terminate when Lk is empty or max_length is reached.
"""

import math
from itertools import combinations
from typing import Sequence, Collection, Dict, FrozenSet, Optional, List, Set, Tuple


def apriori(
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None
) -> Dict[FrozenSet[str], int]:
    """
    Mine all frequent itemsets using the Apriori algorithm.

    Parameters
    ----------
    transactions : Sequence[Collection[str]]
        A sequence of transactions, where each transaction is a collection of item strings.
    min_support : float
        Minimum support threshold as a fraction in (0.0, 1.0].
    max_length : Optional[int]
        Maximum itemset length to discover. If None, mines all lengths.

    Returns
    -------
    Dict[FrozenSet[str], int]
        Mapping from frequent itemset (frozenset of strings) to exact integer support count.
    """
    n_trans = len(transactions)
    if n_trans == 0:
        return {}

    if not (0.0 < min_support <= 1.0):
        raise ValueError(f"min_support must be in (0.0, 1.0], got {min_support}")

    if max_length is not None and max_length < 1:
        return {}

    min_count = math.ceil(min_support * n_trans)

    # Convert transactions to sets to remove duplicate items within transactions
    # and retain non-empty transactions
    clean_transactions: List[Set[str]] = [set(t) for t in transactions if t]
    if not clean_transactions:
        return {}

    # Step 1: Count frequent 1-itemsets
    item_counts: Dict[str, int] = {}
    for t in clean_transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    frequent_1: Dict[Tuple[str, ...], int] = {
        (item,): count
        for item, count in item_counts.items()
        if count >= min_count
    }

    if not frequent_1:
        return {}

    frequent_itemsets: Dict[FrozenSet[str], int] = {
        frozenset(k): v for k, v in frequent_1.items()
    }

    if max_length == 1:
        return frequent_itemsets

    # Pre-filter transactions to only include frequent 1-items
    frequent_items_set = {k[0] for k in frequent_1.keys()}
    filtered_transactions: List[Set[str]] = [
        t & frequent_items_set for t in clean_transactions
    ]
    # Keep only transactions with at least 2 items
    filtered_transactions = [t for t in filtered_transactions if len(t) >= 2]

    # Current level frequent itemsets represented as sorted tuples of strings
    current_frequent: List[Tuple[str, ...]] = sorted(frequent_1.keys())

    k = 2
    while current_frequent:
        if max_length is not None and k > max_length:
            break

        # Candidate generation: join L(k-1) with L(k-1)
        candidates: List[Tuple[str, ...]] = []
        n_curr = len(current_frequent)

        for i in range(n_curr):
            itemset1 = current_frequent[i]
            for j in range(i + 1, n_curr):
                itemset2 = current_frequent[j]
                # Join condition: first k-2 items must match
                if itemset1[:k - 2] == itemset2[:k - 2]:
                    # Create candidate by appending the distinct last items
                    new_candidate = itemset1 + (itemset2[k - 2],)
                    candidates.append(new_candidate)
                else:
                    # Because current_frequent is sorted, once prefixes don't match,
                    # no subsequent itemset2 will match itemset1's prefix
                    break

        if not candidates:
            break

        # Apriori pruning: all (k-1) subsets of candidate must be in current_frequent
        current_set = set(current_frequent)
        pruned_candidates: List[Tuple[str, ...]] = []
        for cand in candidates:
            all_subsets_frequent = True
            for subset in combinations(cand, k - 1):
                if subset not in current_set:
                    all_subsets_frequent = False
                    break
            if all_subsets_frequent:
                pruned_candidates.append(cand)

        if not pruned_candidates:
            break

        # Candidate counting
        # Filter transactions that have at least k items
        filtered_transactions = [t for t in filtered_transactions if len(t) >= k]
        if not filtered_transactions:
            break

        # Convert pruned candidates to sets for fast issubset check
        cand_sets = [(cand, set(cand)) for cand in pruned_candidates]
        candidate_counts: Dict[Tuple[str, ...], int] = {cand: 0 for cand in pruned_candidates}

        for t in filtered_transactions:
            for cand, cand_set in cand_sets:
                if cand_set.issubset(t):
                    candidate_counts[cand] += 1

        # Select candidates meeting min_count
        next_frequent: List[Tuple[str, ...]] = []
        for cand, count in candidate_counts.items():
            if count >= min_count:
                next_frequent.append(cand)
                frequent_itemsets[frozenset(cand)] = count

        current_frequent = sorted(next_frequent)
        k += 1

    return frequent_itemsets
