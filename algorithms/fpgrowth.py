"""
AgriInsight — FP-Growth Algorithm Implementation
Academic, genuine implementation of the FP-Growth algorithm (Han, Pei, & Yin, 2000).

Key Steps:
1. Scan transactions to count 1-item frequencies and filter by min_support.
2. Build global deterministic item ordering (frequency descending, item name ascending).
3. Build FP-Tree with root, child branches, and header table with node-links.
4. Recursively mine FP-Tree bottom-up:
   - Form frequent itemset (prefix + item) with exact support count.
   - Extract conditional pattern base by traversing node links up to the root.
   - Construct conditional FP-Tree.
   - Recursively mine conditional tree until empty or max_length is reached.
"""

import math
from typing import Sequence, Collection, Dict, FrozenSet, Optional, List, Set, Tuple


class FPNode:
    """A node in an FP-Tree."""
    __slots__ = ('item', 'count', 'parent', 'children', 'node_link')

    def __init__(self, item: Optional[str], count: int, parent: Optional['FPNode'] = None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: Dict[str, 'FPNode'] = {}
        self.node_link: Optional['FPNode'] = None


class FPTree:
    """An FP-Tree containing root node and item header table."""

    def __init__(self):
        self.root = FPNode(None, 0, None)
        # Header table: item -> [support_count, head_node, tail_node]
        self.header_table: Dict[str, List] = {}

    def add_transaction(self, transaction: List[str], count: int = 1):
        """Insert a filtered, ordered transaction into the FP-Tree."""
        curr = self.root
        for item in transaction:
            if item in curr.children:
                curr.children[item].count += count
            else:
                new_node = FPNode(item, count, curr)
                curr.children[item] = new_node
                # Update header table node-links
                if item in self.header_table:
                    # Append new node at tail of linked list for O(1) insertion
                    tail = self.header_table[item][2]
                    tail.node_link = new_node
                    self.header_table[item][2] = new_node
                else:
                    self.header_table[item] = [count, new_node, new_node]
            curr = curr.children[item]


def fpgrowth(
    transactions: Sequence[Collection[str]],
    min_support: float,
    max_length: Optional[int] = None
) -> Dict[FrozenSet[str], int]:
    """
    Mine all frequent itemsets using FP-Growth.

    Parameters
    ----------
    transactions : Sequence[Collection[str]]
        Collection of transaction records.
    min_support : float
        Minimum support fraction in (0.0, 1.0].
    max_length : Optional[int]
        Maximum allowed itemset length.

    Returns
    -------
    Dict[FrozenSet[str], int]
        Mapping from frequent itemsets to exact integer support counts.
    """
    n_trans = len(transactions)
    if n_trans == 0:
        return {}

    if not (0.0 < min_support <= 1.0):
        raise ValueError(f"min_support must be in (0.0, 1.0], got {min_support}")

    if max_length is not None and max_length < 1:
        return {}

    min_count = math.ceil(min_support * n_trans)

    # First pass: count 1-item support
    item_counts: Dict[str, int] = {}
    for t in transactions:
        if not t:
            continue
        # Ensure unique items per transaction
        unique_items = set(t)
        for item in unique_items:
            item_counts[item] = item_counts.get(item, 0) + 1

    # Filter frequent 1-items
    frequent_items = {
        item: count for item, count in item_counts.items() if count >= min_count
    }

    if not frequent_items:
        return {}

    # Deterministic order: descending by count, then ascending alphabetically by item string
    # Sorting key: (-count, item)
    item_order = {
        item: idx
        for idx, (item, _) in enumerate(
            sorted(frequent_items.items(), key=lambda x: (-x[1], x[0]))
        )
    }

    # Second pass: construct FP-Tree
    tree = FPTree()
    for t in transactions:
        if not t:
            continue
        # Filter to frequent items and sort by global deterministic order
        filtered = [item for item in set(t) if item in frequent_items]
        if filtered:
            filtered.sort(key=lambda x: item_order[x])
            tree.add_transaction(filtered, count=1)

    frequent_itemsets: Dict[FrozenSet[str], int] = {}

    def mine_tree(curr_tree: FPTree, prefix: Tuple[str, ...]):
        """Recursively mine the FP-Tree."""
        # Process items in reverse of global order (least frequent first)
        sorted_items = sorted(
            curr_tree.header_table.keys(),
            key=lambda x: item_order.get(x, 0),
            reverse=True
        )

        for item in sorted_items:
            head_info = curr_tree.header_table[item]
            # Exact support is sum of counts across all nodes of this item in the tree
            item_support = 0
            node = head_info[1]
            while node:
                item_support += node.count
                node = node.node_link

            if item_support < min_count:
                continue

            new_prefix = prefix + (item,)
            frequent_itemsets[frozenset(new_prefix)] = item_support

            # Check max_length constraint
            if max_length is not None and len(new_prefix) >= max_length:
                continue

            # Build conditional pattern base
            cond_patterns: List[Tuple[List[str], int]] = []
            node = head_info[1]
            while node:
                path: List[str] = []
                parent = node.parent
                # Ascend to root (excluding root and the item itself)
                while parent and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    # Reverse because we ascended from bottom to top
                    path.reverse()
                    cond_patterns.append((path, node.count))
                node = node.node_link

            # Count item occurrences in conditional pattern base
            cond_counts: Dict[str, int] = {}
            for path, count in cond_patterns:
                for p_item in path:
                    cond_counts[p_item] = cond_counts.get(p_item, 0) + count

            # Filter frequent items in conditional tree
            cond_frequent = {
                p_item for p_item, p_count in cond_counts.items() if p_count >= min_count
            }

            if not cond_frequent:
                continue

            # Build conditional FP-tree
            cond_tree = FPTree()
            for path, count in cond_patterns:
                filtered_path = [p_item for p_item in path if p_item in cond_frequent]
                if filtered_path:
                    filtered_path.sort(key=lambda x: item_order[x])
                    cond_tree.add_transaction(filtered_path, count=count)

            # Recurse if tree is non-empty
            if cond_tree.header_table:
                mine_tree(cond_tree, new_prefix)

    mine_tree(tree, ())
    return frequent_itemsets
