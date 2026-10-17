"""
AgriInsight — Visualizations Module
Creates interactive Plotly figures for algorithm benchmarks, dataset distributions,
and association rule exploration.
"""

from typing import List, Dict, Any, Optional, Tuple
import collections

# Color palette inspired by agricultural landscapes
PALETTE = {
    "Apriori": "#E76F51",     # Warm terracotta
    "FP-Growth": "#2A9D8F",    # Deep teal
    "ECLAT": "#E9C46A",        # Golden harvest
    "H-Mine": "#457B9D",       # Forest steel blue
    "Primary": "#1D3557",
    "Accent": "#264653"
}


def plot_execution_time_bar(benchmark_data: List[Dict[str, Any]]):
    """Generates an interactive bar chart of execution time across algorithms."""
    import plotly.graph_objects as go

    algs = [d["Algorithm"] for d in benchmark_data]
    times = [d["Time (s)"] for d in benchmark_data]
    colors = [PALETTE.get(alg, "#888888") for alg in algs]

    fig = go.Figure(data=[
        go.Bar(
            x=algs,
            y=times,
            text=[f"{t:.4f}s" for t in times],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#1D3557", width=1.5)),
            hovertemplate="<b>%{x}</b><br>Execution Time: %{y:.4f} seconds<extra></extra>"
        )
    ])

    fig.update_layout(
        title="Algorithm Execution Time Comparison (Identical Input & Threshold)",
        xaxis_title="Association Rule Mining Algorithm",
        yaxis_title="Execution Time (seconds)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=13)
    )
    return fig


def plot_scalability_curve(scaling_data: List[Dict[str, Any]]):
    """
    Plots execution time vs transaction count across algorithms.
    scaling_data is a list of dicts with keys: 'Algorithm', 'Size', 'Time'.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    # Group by algorithm
    by_alg: Dict[str, List[Tuple]] = {}
    for d in scaling_data:
        alg = d["Algorithm"]
        if alg not in by_alg:
            by_alg[alg] = []
        by_alg[alg].append((d["Size"], d["Time"]))

    for alg, points in by_alg.items():
        points.sort(key=lambda x: x[0])
        sizes = [p[0] for p in points]
        times = [p[1] for p in points]
        fig.add_trace(go.Scatter(
            x=sizes,
            y=times,
            mode="lines+markers",
            name=alg,
            line=dict(color=PALETTE.get(alg, "#888888"), width=3),
            marker=dict(size=8),
            hovertemplate=f"<b>{alg}</b><br>Transactions: %{{x:,}}<br>Time: %{{y:.4f}}s<extra></extra>"
        ))

    fig.update_layout(
        title="Scalability Analysis: Execution Time vs Transaction Count",
        xaxis_title="Number of Agricultural Transactions (N)",
        yaxis_title="Execution Time (seconds)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=13)
    )
    return fig


def plot_min_support_curve(support_sweep_data: List[Dict[str, Any]]):
    """
    Plots execution time vs minimum support threshold.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    by_alg: Dict[str, List[Tuple]] = {}
    for d in support_sweep_data:
        alg = d["Algorithm"]
        if alg not in by_alg:
            by_alg[alg] = []
        by_alg[alg].append((d["Min Support (%)"], d["Time (s)"]))

    for alg, points in by_alg.items():
        points.sort(key=lambda x: x[0])
        supps = [p[0] for p in points]
        times = [p[1] for p in points]
        fig.add_trace(go.Scatter(
            x=supps,
            y=times,
            mode="lines+markers",
            name=alg,
            line=dict(color=PALETTE.get(alg, "#888888"), width=3),
            marker=dict(size=8),
            hovertemplate=f"<b>{alg}</b><br>Min Support: %{{x}}%<br>Time: %{{y:.4f}}s<extra></extra>"
        ))

    fig.update_layout(
        title="Impact of Minimum Support Threshold on Execution Time",
        xaxis_title="Minimum Support Threshold (%)",
        yaxis_title="Execution Time (seconds)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=13)
    )
    return fig


def plot_memory_usage_bar(benchmark_data: List[Dict[str, Any]]):
    """Plots peak memory consumption across algorithms."""
    import plotly.graph_objects as go

    algs = [d["Algorithm"] for d in benchmark_data]
    mems = [d.get("Peak Memory (MB)", 0.0) for d in benchmark_data]
    colors = [PALETTE.get(alg, "#888888") for alg in algs]

    fig = go.Figure(data=[
        go.Bar(
            x=algs,
            y=mems,
            text=[f"{m:.2f} MB" for m in mems],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#1D3557", width=1.5)),
            hovertemplate="<b>%{x}</b><br>Peak Memory: %{y:.2f} MB<extra></extra>"
        )
    ])

    fig.update_layout(
        title="Peak Memory Allocation by Algorithm (tracemalloc)",
        xaxis_title="Association Rule Mining Algorithm",
        yaxis_title="Peak Memory (MB)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=13)
    )
    return fig


def plot_categorical_distribution(counts_dict: Dict[str, int], title: str, chart_type: str = "bar"):
    """Plots category distributions as an interactive bar or donut chart."""
    import plotly.graph_objects as go

    labels = list(counts_dict.keys())
    values = list(counts_dict.values())

    if chart_type == "pie" or chart_type == "donut":
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45 if chart_type == "donut" else 0.0,
                marker=dict(colors=["#2A9D8F", "#E76F51", "#E9C46A", "#457B9D", "#264653", "#F4A261", "#8AB17D"]),
                hovertemplate="<b>%{label}</b>: %{value:,} (%{percent})<extra></extra>"
            )
        ])
    else:
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color="#2A9D8F"),
                hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>"
            )
        ])

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        font=dict(family="Inter, system-ui, sans-serif", size=12)
    )
    return fig


def plot_rules_scatter(rules: List[Dict[str, Any]]):
    """Interactive scatter plot of association rules: Support vs Confidence, colored by Lift."""
    import plotly.graph_objects as go

    if not rules:
        return go.Figure()

    supports = [r["support"] * 100 for r in rules]
    confidences = [r["confidence"] * 100 for r in rules]
    lifts = [min(r["lift"], 5.0) for r in rules]  # Cap extreme lift for visual color scaling
    hover_texts = [
        f"<b>{r['rule_str']}</b><br>Support: {r['support']*100:.2f}%<br>Confidence: {r['confidence']*100:.2f}%<br>Lift: {r['lift']:.2f}"
        for r in rules
    ]

    fig = go.Figure(data=[
        go.Scattergl(
            x=supports,
            y=confidences,
            mode="markers",
            marker=dict(
                size=8,
                color=lifts,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Lift")
            ),
            text=hover_texts,
            hoverinfo="text"
        )
    ])

    fig.update_layout(
        title="Discovered Association Rules (Support vs Confidence, colored by Lift)",
        xaxis_title="Support (%)",
        yaxis_title="Confidence (%)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=13)
    )
    return fig
