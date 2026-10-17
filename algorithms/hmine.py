"""
AgriInsight — H-Mine Algorithm Implementation
Academic, genuine implementation of the H-Mine (Hyper-Structure Mining) algorithm 
(Pei, Han, Lu, Nishio, Tang, & Yang, ICDM 2001: "H-Mine: Hyper-Structure Mining of Frequent Patterns in Large Databases").

Key Principles & Data Structures:
1. H-struct (Hyper-Structure):
   - Transactions are stored in memory with items ordered by a global deterministic sequence.
   - For each item, a header table entry contains its global support count and a hyper-link pointer 
     to the first transaction containing the item.
   - Within each transaction, hyper-links chain each item occurrence to the next transaction 
     containing that same item.
2. Projected Database via Hyper-Link Adjustment:
   - To mine patterns with prefix P U {i}, H-Mine traverses the hyper-linked list of item i.
   - It counts frequencies of items appearing after i in the active transactions.
   - Frequent items are identified with exact support counts.
   - The hyper-links are dynamically projected for frequent extension items without physical database replication.
3. Recursive Mining:
   - Recursively mine projected hyper-structures depth-first until no frequent extensions exist 
     or max_length is reached.
"""

import math
from typing import Sequence, Collection, Dict, FrozenSet, Optional, List, Set, Tuple


class HNode:
    """A hyper-link node representing an item occurrence within a transaction."""
    __slots__ = ('item_idx', 'trans_idx', 'next_node')

    def __init__(self, item_idx: int, trans_idx: int):
        self.item_idx = item_idx
        self.trans_idx = trans_idx
        self.next_node: Optional['HNode'] = None


class HStruct:
    """
    Hyper-Structure containing transactions and hyper-linked item lists.
    """
    def __init__(self, num_items: int):
        # Header table: item_idx -> (first HNode, last HNode, support_count)
        self.headers: List[Optional[HNode]] = [None] * num_items
        self.tails: List[Optional[HNode]] = [None] * num_items
        self.counts: List[int] = [0] * num_items

    def add_node(self, item_idx: int, trans_idx: int):
        """Append an item occurrence in a transaction to the hyper-link chain."""
        node = HNode(item_idx, trans_idx)
        self.counts[item_idx] += 1
        if self.headers[item_idx] is None:
            self.headers[item_idx] = node
            self.tails[item_idx] = node
        else:
            self.tails[item_idx].next_node = node
            self.tails[item_idx] = node


def hmine(
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None
) -> Dict[FrozenSet[str], int]:
    """
    Mine all frequent itemsets using the H-Mine algorithm.

    Parameters
    ----------
    transactions : Sequence[Collection[str]]
        A sequence of transactions (collections of item strings).
    min_support : float
        Minimum support threshold fraction in (0.0, 1.0].
    max_length : Optional[int]
        Maximum allowed itemset length.

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

    # Pass 1: Count 1-item frequencies
    item_counts: Dict[str, int] = {}
    for t in transactions:
        if not t:
            continue
        for item in set(t):
            item_counts[item] = item_counts.get(item, 0) + 1

    # Filter frequent items
    frequent_items = {
        item: count for item, count in item_counts.items() if count >= min_count
    }

    if not frequent_items:
        return {}

    # Deterministic global item ordering:
    # Sorted by descending support count, then alphabetically by item name
    sorted_items = sorted(
        frequent_items.keys(),
        key=lambda x: (-frequent_items[x], x)
    )
    item_to_id = {item: idx for idx, item in enumerate(sorted_items)}
    id_to_item = sorted_items
    num_frequent = len(sorted_items)

    # Pass 2: Build in-memory transaction database and initial H-Struct
    # Clean and sort transactions using integer item IDs
    encoded_transactions: List[List[int]] = []
    hstruct = HStruct(num_frequent)

    for trans_idx, t in enumerate(transactions):
        if not t:
            continue
        # Deduplicate and filter to frequent items
        item_ids = [item_to_id[item] for item in set(t) if item in item_to_id]
        if item_ids:
            item_ids.sort()
            encoded_transactions.append(item_ids)
            curr_tid = len(encoded_transactions) - 1
            for i_id in item_ids:
                hstruct.add_node(i_id, curr_tid)

    frequent_itemsets: Dict[FrozenSet[str], int] = {}

    # Record 1-itemsets
    for i_id in range(num_frequent):
        cnt = hstruct.counts[i_id]
        if cnt >= min_count:
            frequent_itemsets[frozenset([id_to_item[i_id]])] = cnt

    if max_length == 1:
        return frequent_itemsets

    def mine_projected(
        prefix_ids: Tuple[int, ...],
        curr_head: Optional[HNode],
        last_item_id: int
    ):
        """
        Recursively mine frequent itemsets from hyper-linked projected database.
        
        Parameters
        ----------
        prefix_ids : Tuple[int, ...]
            The current itemset prefix as item IDs.
        curr_head : Optional[HNode]
            Head of the hyper-link chain of transactions containing prefix_ids.
        last_item_id : int
            The last item ID in the prefix. Only items with ID > last_item_id are evaluated.
        """
        if max_length is not None and len(prefix_ids) >= max_length:
            return

        # Count frequencies of items appearing after last_item_id in transactions linked from curr_head
        local_counts: Dict[int, int] = {}
        # Also store transaction indices containing each item for projected linking
        local_trans: Dict[int, List[int]] = {}

        curr = curr_head
        while curr is not None:
            t_idx = curr.trans_idx
            trans = encoded_transactions[t_idx]
            
            # Find items in trans strictly following last_item_id
            # Since items are sorted in ascending order of ID, we scan from after last_item_id
            for item_id in trans:
                if item_id > last_item_id:
                    local_counts[item_id] = local_counts.get(item_id, 0) + 1
                    if item_id not in local_trans:
                        local_trans[item_id] = []
                    local_trans[item_id].append(t_idx)

            curr = curr.next_node

        # Filter frequent extensions
        frequent_extensions = [
            item_id for item_id, count in local_counts.items() if count >= min_count
        ]
        # Sort extensions to maintain deterministic order
        frequent_extensions.sort()

        for ext_id in frequent_extensions:
            count = local_counts[ext_id]
            new_prefix = prefix_ids + (ext_id,)
            itemset_names = frozenset(id_to_item[idx] for idx in new_prefix)
            frequent_itemsets[itemset_names] = count

            if max_length is not None and len(new_prefix) >= max_length:
                continue

            # Build hyper-link chain for the projected database of new_prefix
            proj_tids = local_trans[ext_id]
            if not proj_tids:
                continue

            # Create head of linked list for projected transactions
            proj_head = HNode(ext_id, proj_tids[0])
            tail = proj_head
            for tid in proj_tids[1:]:
                next_node = HNode(ext_id, tid)
                tail.next_node = next_node
                tail = next_node

            # Recurse into the projected hyper-structure
            mine_projected(new_prefix, proj_head, ext_id)

    # Initial top-level mining for each frequent 1-item
    for i_id in range(num_frequent):
        head = hstruct.headers[i_id]
        if head is not None:
            mine_projected((i_id,), head, i_id)

    return frequent_itemsets
