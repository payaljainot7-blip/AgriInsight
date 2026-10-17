"""
AgriInsight — Association Rule Generation Module
Extracts valid association rules X -> Y from frequent itemsets and computes statistical metrics.

For each frequent itemset I of size >= 2:
- Generates all non-empty proper subsets X subseteq I as antecedents, and Y = I \\ X as consequents.
- Evaluates Support, Confidence, Lift, Leverage, and Conviction.
- Filters rules based on user-defined min_confidence and min_lift thresholds.
- Guarantees deterministic ordering for rule reproducibility.
"""

from itertools import combinations
from typing import Dict, FrozenSet, List, Any, Optional, Set, Tuple
from core.metrics import calculate_rule_metrics


def generate_association_rules(
    frequent_itemsets: Dict[FrozenSet[str], int],
    num_transactions: int,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    max_consequent_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Derives association rules from frequent itemsets.

    Parameters
    ----------
    frequent_itemsets : Dict[FrozenSet[str], int]
        Mapping of frequent itemset to exact integer support count.
    num_transactions : int
        Total number of transactions in the dataset.
    min_confidence : float
        Minimum confidence threshold in [0.0, 1.0].
    min_lift : float
        Minimum lift threshold (typically >= 1.0).
    max_consequent_size : Optional[int]
        Optional restriction on the number of items in the consequent.

    Returns
    -------
    List[Dict[str, Any]]
        List of generated rules sorted deterministically by (confidence desc, lift desc, support desc).
    """
    if num_transactions <= 0:
        return []

    rules: List[Dict[str, Any]] = []

    # Filter itemsets with at least 2 items
    multi_itemsets = [
        (itemset, count) for itemset, count in frequent_itemsets.items()
        if len(itemset) >= 2
    ]

    for itemset, count_both in multi_itemsets:
        items_list = sorted(list(itemset))
        k = len(items_list)

        # Generate all proper subsets of itemset as antecedent X
        for r in range(1, k):
            for antecedent_tuple in combinations(items_list, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                if max_consequent_size is not None and len(consequent) > max_consequent_size:
                    continue

                if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                    count_x = frequent_itemsets[antecedent]
                    count_y = frequent_itemsets[consequent]

                    metrics = calculate_rule_metrics(
                        count_antecedent=count_x,
                        count_consequent=count_y,
                        count_both=count_both,
                        num_transactions=num_transactions
                    )

                    if metrics["confidence"] >= min_confidence and metrics["lift"] >= min_lift:
                        rule_dict = {
                            "antecedent": tuple(sorted(antecedent)),
                            "consequent": tuple(sorted(consequent)),
                            "antecedent_str": ", ".join(sorted(antecedent)),
                            "consequent_str": ", ".join(sorted(consequent)),
                            "rule_str": f"{{{', '.join(sorted(antecedent))}}} => {{{', '.join(sorted(consequent))}}}",
                            "support": metrics["support"],
                            "confidence": metrics["confidence"],
                            "lift": metrics["lift"],
                            "leverage": metrics["leverage"],
                            "conviction": metrics["conviction"],
                            "count_rule": count_both,
                            "count_antecedent": count_x,
                            "count_consequent": count_y,
                            "num_transactions": num_transactions
                        }
                        rules.append(rule_dict)

    # Sort deterministically
    rules.sort(key=lambda r: (-r["confidence"], -r["lift"], -r["support"], r["rule_str"]))
    return rules


def verify_rules_equivalence(
    rules_a: List[Dict[str, Any]],
    rules_b: List[Dict[str, Any]]
) -> bool:
    """
    Verifies that two rule sets are identical in content, metrics, and size.
    """
    if len(rules_a) != len(rules_b):
        return False

    dict_a = {r["rule_str"]: r for r in rules_a}
    dict_b = {r["rule_str"]: r for r in rules_b}

    if set(dict_a.keys()) != set(dict_b.keys()):
        return False

    for rule_str, ra in dict_a.items():
        rb = dict_b[rule_str]
        if ra["count_rule"] != rb["count_rule"]:
            return False
        if abs(ra["confidence"] - rb["confidence"]) > 1e-7:
            return False
        if abs(ra["lift"] - rb["lift"]) > 1e-7:
            return False

    return True
