"""
AgriInsight — Core Metrics Module
Mathematical definitions and computation of Association Rule Mining metrics:
- Support
- Confidence
- Lift
- Leverage
- Conviction

All formulas strictly adhere to established academic definitions.
"""

from typing import Dict, Any, Optional
import math


def calculate_rule_metrics(
    count_antecedent: int,
    count_consequent: int,
    count_both: int,
    num_transactions: int
) -> Dict[str, float]:
    """
    Computes all standard rule evaluation metrics for a rule X -> Y.

    Parameters
    ----------
    count_antecedent : int
        Exact support count of X in dataset.
    count_consequent : int
        Exact support count of Y in dataset.
    count_both : int
        Exact support count of X U Y in dataset.
    num_transactions : int
        Total number of transactions N in the dataset.

    Returns
    -------
    dict
        Dictionary containing support, confidence, lift, leverage, and conviction.
    """
    if num_transactions <= 0:
        raise ValueError("num_transactions must be positive.")
    if count_antecedent <= 0:
        raise ValueError("count_antecedent must be positive.")
    if count_both < 0 or count_both > count_antecedent or count_both > count_consequent:
        raise ValueError("count_both is invalid relative to antecedent/consequent counts.")

    n = float(num_transactions)
    supp_x = count_antecedent / n
    supp_y = count_consequent / n
    supp_xy = count_both / n

    # Confidence = P(Y | X) = Supp(X U Y) / Supp(X)
    confidence = supp_xy / supp_x

    # Lift = Supp(X U Y) / (Supp(X) * Supp(Y))
    # Lift > 1 implies positive correlation; < 1 implies negative correlation; == 1 implies independence.
    if supp_y > 0:
        lift = supp_xy / (supp_x * supp_y)
    else:
        lift = 0.0

    # Leverage = Supp(X U Y) - (Supp(X) * Supp(Y))
    # Measures the difference between observed co-occurrence and expected co-occurrence under independence.
    leverage = supp_xy - (supp_x * supp_y)

    # Conviction = (1 - Supp(Y)) / (1 - Confidence)
    # Measures how much the rule would be incorrect if X and Y were independent.
    # If confidence == 1.0, conviction is infinite.
    if math.isclose(confidence, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        conviction = float('inf')
    else:
        denom = 1.0 - confidence
        if denom > 0:
            conviction = (1.0 - supp_y) / denom
        else:
            conviction = float('inf')

    return {
        "support": supp_xy,
        "confidence": confidence,
        "lift": lift,
        "leverage": leverage,
        "conviction": conviction,
        "count_rule": count_both,
        "count_antecedent": count_antecedent,
        "count_consequent": count_consequent,
        "num_transactions": num_transactions
    }
