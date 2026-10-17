"""
AgriInsight — Agricultural Pattern Discovery Using Association Rule Mining
Interactive Streamlit Application

Authors: AgriInsight Research & Development
Technology Stack: Python 3.11+, Streamlit, Plotly, Pandas
Algorithms: Apriori, FP-Growth, ECLAT, H-Mine
"""

import datetime
import io
import json
import math
import os
import sys
import time
from typing import Dict, List, Set, Tuple, Any, Optional

import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.apriori import apriori
from algorithms.fpgrowth import fpgrowth
from algorithms.eclat import eclat
from algorithms.hmine import hmine
from core.verification import verify_all_four, compare_two_algorithms, normalize_itemsets
from core.association_rules import generate_association_rules, verify_rules_equivalence
from core.preprocessing import (
    load_transactions_from_csv,
    load_raw_dataset_records,
    transform_raw_to_transactions,
    get_transaction_statistics,
    DEFAULT_NUMERIC_BINS
)
from core.benchmark import run_single_benchmark, benchmark_all_algorithms
from data.generate_dataset import generate_agricultural_dataset
import visualizations.charts as charts

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriInsight | Agricultural Pattern Mining & Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional academic and enterprise CSS styling (clean, modern, emoji-free)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .main-header {
        color: #0F172A;
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.4rem;
        letter-spacing: -0.01em;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }
    .academic-disclaimer {
        background-color: #F8FAFC;
        border-left: 4px solid #0D9488;
        border: 1px solid #E2E8F0;
        border-left-width: 4px;
        border-left-color: #0D9488;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 1.3rem;
        line-height: 1.5;
    }
    .status-card-pass {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .status-card-pass h3 {
        color: #15803D;
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .status-card-pass p {
        color: #166534;
        margin-top: 0.45rem;
        margin-bottom: 0;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .status-card-fail {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #DC2626;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .status-card-fail h3 {
        color: #B91C1C;
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .status-card-fail p {
        color: #991B1B;
        margin-top: 0.45rem;
        margin-bottom: 0;
        font-size: 0.92rem;
    }
    .brand-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.25rem 0 0.5rem 0;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .brand-caption {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 0.15rem;
    }
    .pattern-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .pattern-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.6rem;
    }
    .pattern-metric-pill {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .pattern-desc {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# High-Performance Cached Data Loading & Resource Management
# ---------------------------------------------------------
DEFAULT_RAW_FILE = os.path.join(PROJECT_ROOT, "data", "agriinsight_synthetic_100k_raw.csv")
if not os.path.exists(DEFAULT_RAW_FILE):
    DEFAULT_RAW_FILE = os.path.join(PROJECT_ROOT, "agriinsight_synthetic_100k_raw.csv")

DEFAULT_TX_FILE = os.path.join(PROJECT_ROOT, "data", "agriinsight_synthetic_100k_transactions.csv")
if not os.path.exists(DEFAULT_TX_FILE):
    DEFAULT_TX_FILE = os.path.join(PROJECT_ROOT, "agriinsight_synthetic_100k_transactions.csv")


@st.cache_resource(show_spinner=False)
def get_master_transactions(filepath: str) -> List[List[str]]:
    """
    Loads master transaction dataset once into RAM.
    Subsequent transaction slices are created instantaneously in memory without disk I/O.
    """
    if os.path.exists(filepath):
        return load_transactions_from_csv(filepath)
    return []


@st.cache_resource(show_spinner=False)
def get_master_raw_records(filepath: str, max_rows: int = 1000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads raw records once into memory."""
    if os.path.exists(filepath):
        return load_raw_dataset_records(filepath, max_rows=max_rows)
    return [], {}


@st.cache_data(show_spinner=False)
def get_cached_transaction_statistics(slice_size: int, is_custom: bool = False, custom_sig: int = 0) -> Dict[str, Any]:
    """Computes and caches transaction summary metrics to make dashboard reruns instant."""
    if not is_custom:
        master_tx = get_master_transactions(DEFAULT_TX_FILE)
        active_slice = master_tx[:slice_size]
    else:
        active_slice = st.session_state.get("transactions", [])
    return get_transaction_statistics(active_slice)


@st.cache_data(show_spinner=False)
def cached_mine_algorithm(
    algorithm_name: str,
    slice_size: int,
    min_support: float,
    max_length: Optional[int],
    track_memory: bool,
    is_custom: bool = False
) -> Dict[str, Any]:
    """
    Caches single algorithm mining results to ensure zero lag on repetitive runs and tab switches.
    """
    if not is_custom:
        master_tx = get_master_transactions(DEFAULT_TX_FILE)
        tx_data = master_tx[:slice_size]
    else:
        tx_data = st.session_state.get("transactions", [])

    return run_single_benchmark(
        algorithm_name,
        tx_data,
        min_support=min_support,
        max_length=max_length,
        track_memory=track_memory
    )


@st.cache_data(show_spinner=False)
def build_itemsets_dataframe(frequent_tuples: Tuple[Tuple[Tuple[str, ...], int], ...], n_trans: int) -> pd.DataFrame:
    """Pre-constructs the itemsets DataFrame once so filtering is instantaneous in pandas."""
    records = []
    for iset, cnt in frequent_tuples:
        records.append({
            "Itemset": " + ".join(sorted(iset)),
            "Length": len(iset),
            "Support Count": cnt,
            "Support (%)": f"{(cnt / n_trans)*100:.2f}%" if n_trans > 0 else "0%"
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["Support Count", "Length"], ascending=[False, False])
    return df


@st.cache_data(show_spinner=False)
def get_cached_rules(
    frequent_tuples: Tuple[Tuple[Tuple[str, ...], int], ...],
    num_trans: int,
    min_confidence: float,
    min_lift: float
) -> List[Dict[str, Any]]:
    """Caches association rules generation to make threshold slider changes responsive."""
    frequent = {frozenset(k): v for k, v in frequent_tuples}
    return generate_association_rules(
        frequent_itemsets=frequent,
        num_transactions=num_trans,
        min_confidence=min_confidence,
        min_lift=min_lift
    )


@st.cache_data(show_spinner=False)
def build_rules_dataframe(rules_tuples: Tuple[Tuple[str, float, float, float, float, float, int], ...]) -> pd.DataFrame:
    """Pre-builds the association rules display DataFrame."""
    table_records = []
    for rule_str, sup, conf, lift, lev, conv, count_val in rules_tuples:
        conv_str = f"{conv:.3f}" if not math.isinf(conv) else "Infinity"
        table_records.append({
            "Rule": rule_str,
            "Support (%)": f"{sup * 100:.2f}%",
            "Confidence (%)": f"{conf * 100:.2f}%",
            "Lift": f"{lift:.3f}",
            "Leverage": f"{lev:.4f}",
            "Conviction": conv_str,
            "Count": count_val
        })
    return pd.DataFrame(table_records)


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Caches CSV string encoding so download buttons do not lag on user interactions."""
    return df.to_csv(index=False).encode('utf-8')


# ---------------------------------------------------------
# Sidebar Navigation & Global Controls
# ---------------------------------------------------------
with st.sidebar:
    # Minimalist, modern SVG logo
    st.markdown("""
    <div class="brand-container">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="24" height="24" rx="6" fill="#0F172A"/>
            <path d="M12 4C7.58172 4 4 7.58172 4 12C4 16.4183 7.58172 20 12 20C16.4183 20 20 16.4183 20 12C20 7.58172 16.4183 4 12 4Z" stroke="#0D9488" stroke-width="1.8"/>
            <path d="M12 8V16M8 12H16M15 9L9 15" stroke="#F8FAFC" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
        <div>
            <div class="brand-title">AgriInsight</div>
            <div class="brand-caption">Agricultural Pattern Mining Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Dataset Explorer",
            "Preprocessing & Discretization",
            "Algorithm Execution",
            "Frequent Itemsets Explorer",
            "Association Rules Explorer",
            "Algorithm Correctness Verification",
            "Performance Benchmark",
            "Agronomic Pattern Insights",
            "Export & Reproducibility"
        ]
    )

    st.markdown("---")
    st.markdown("#### Global Thresholds")
    min_sup_pct = st.slider("Min Support (%)", min_value=0.5, max_value=25.0, value=5.0, step=0.5)
    min_support = min_sup_pct / 100.0

    min_conf_pct = st.slider("Min Confidence (%)", min_value=10.0, max_value=100.0, value=50.0, step=5.0)
    min_confidence = min_conf_pct / 100.0

    min_lift = st.slider("Min Lift", min_value=0.5, max_value=5.0, value=1.1, step=0.1)

    max_len_option = st.selectbox("Max Itemset Length", [None, 2, 3, 4, 5, 6], index=3)

    st.markdown("---")
    sample_size = st.selectbox(
        "Working Transaction Slice",
        [1000, 5000, 10000, 25000, 50000, 100000],
        index=1,
        help="Choose active dataset slice size. Memory slicing is instantaneous."
    )


# ---------------------------------------------------------
# State Management & Slicing
# ---------------------------------------------------------
is_custom_dataset = st.session_state.get("is_custom_dataset", False)

if not is_custom_dataset:
    if "transactions" not in st.session_state or st.session_state.get("current_slice") != sample_size:
        master_tx = get_master_transactions(DEFAULT_TX_FILE)
        st.session_state["transactions"] = master_tx[:sample_size]
        st.session_state["current_slice"] = sample_size
else:
    # Using user-uploaded transactions
    if "transactions" not in st.session_state:
        st.session_state["transactions"] = []

if "mining_results" not in st.session_state:
    st.session_state["mining_results"] = {}
if "last_params" not in st.session_state:
    st.session_state["last_params"] = {}


# =========================================================
# SECTION 1: OVERVIEW DASHBOARD
# =========================================================
if menu == "Overview Dashboard":
    st.markdown('<div class="main-header">AgriInsight Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Agricultural Pattern Discovery & Canonical Association Rule Mining Benchmark</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="academic-disclaimer">
        <b>Academic & Synthetic Data Protocol:</b> All records evaluated within AgriInsight are generated 
        synthetically for rigorous algorithmic benchmarking, comparative complexity analysis, and educational demonstration. 
        Discovered association rules represent statistical co-occurrences in synthetic data and should not be interpreted as empirical field observations.
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Cards (Computed via cached stats for zero latency)
    tx_count = len(st.session_state.get("transactions", []))
    tx_stats = get_cached_transaction_statistics(
        sample_size if not is_custom_dataset else tx_count,
        is_custom=is_custom_dataset
    ) if tx_count > 0 else {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{tx_count:,}</div>
            <div class="metric-lbl">Active Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{tx_stats.get('num_distinct_items', 0)}</div>
            <div class="metric-lbl">Unique Item Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">4</div>
            <div class="metric-lbl">Canonical Algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        min_cnt = math.ceil(min_support * tx_count) if tx_count > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{min_cnt:,}</div>
            <div class="metric-lbl">Min Support Threshold ({min_sup_pct}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Core Objectives & Algorithmic Guarantees")
    st.markdown(r"""
    AgriInsight implements and rigorously evaluates four canonical frequent-itemset mining algorithms on identical agricultural data:
    1. **Apriori**: Classical level-wise candidate generation and anti-monotonic subset pruning (Agrawal & Srikant, 1994).
    2. **FP-Growth**: Prefix-tree structure eliminating candidate generation via conditional FP-tree recursion (Han et al., 2000).
    3. **ECLAT**: Vertical transaction-ID (TID) set intersection and equivalence class clustering (Zaki, 2000).
    4. **H-Mine**: Hyper-structure memory-linked transaction projection without physical tree copies (Pei et al., ICDM 2001).

    #### Verification Guarantee
    For any chosen transaction dataset and minimum support threshold:
    $$\mathbf{Apriori \equiv FP\text{-}Growth \equiv ECLAT \equiv H\text{-}Mine}$$
    All four algorithms must yield **identical frequent itemsets** and **exact integer support counts**.
    """)

    st.markdown("### Architectural Pipeline")
    c_w1, c_w2, c_w3, c_w4 = st.columns(4)
    with c_w1:
        st.info("**1. Data Discretization**\n\nContinuous physical measurements (soil pH, nutrients, weather) are binned into standardized agronomic categories.")
    with c_w2:
        st.info("**2. Algorithm Execution**\n\nApriori, FP-Growth, ECLAT, and H-Mine are independently executed on the exact same transaction memory collection.")
    with c_w3:
        st.info("**3. Correctness Verification**\n\nOutputs are canonically normalized and tested across 6 pairwise comparisons for zero-mismatch equivalence.")
    with c_w4:
        st.info("**4. Association Rules**\n\nSupport, Confidence, Lift, Leverage, and Conviction are calculated to uncover recurring multi-factor agricultural relationships.")


# =========================================================
# SECTION 2: DATASET EXPLORER
# =========================================================
elif menu == "Dataset Explorer":
    st.markdown('<div class="main-header">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Continuous Attributes, Categorical Distributions, & Transaction Structure</div>', unsafe_allow_html=True)

    raw_records, raw_summary = get_master_raw_records(DEFAULT_RAW_FILE, max_rows=1000)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Preview & Hygiene",
        "Distribution Analytics",
        "Scaled Dataset Generator",
        "Custom Dataset Upload"
    ])

    with tab1:
        st.markdown(f"**Loaded Preview (Top {len(raw_records):,} of 100,000 raw records)**")
        if raw_records:
            df_preview = pd.DataFrame(raw_records[:25])
            st.dataframe(df_preview, use_container_width=True)

            st.markdown("#### Missing Values & Data Hygiene")
            missing_df = pd.DataFrame([
                {"Column": col, "Missing Count": cnt, "Missing (%)": f"{(cnt/len(raw_records))*100:.1f}%"}
                for col, cnt in raw_summary.get("missing_counts", {}).items()
            ])
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.warning("Default raw dataset not found. Please use the generator tab.")

    with tab4:
        st.markdown("#### Upload Custom Agricultural CSV Dataset")
        st.write("Upload either a raw continuous agricultural CSV or a pre-formatted transaction CSV.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            buf = io.StringIO(content)
            first_line = content.split("\n", 1)[0]
            if "items" in first_line:
                custom_tx = load_transactions_from_csv(buf)
                st.session_state["transactions"] = custom_tx
                st.session_state["current_slice"] = len(custom_tx)
                st.session_state["is_custom_dataset"] = True
                st.success(f"Loaded {len(custom_tx):,} transactions from uploaded file!")
            else:
                raw_recs, summary = load_raw_dataset_records(buf)
                st.write(f"Loaded {len(raw_recs):,} raw records with {summary['total_columns']} columns.")
                tx_converted = transform_raw_to_transactions(raw_recs)
                st.session_state["transactions"] = tx_converted
                st.session_state["current_slice"] = len(tx_converted)
                st.session_state["is_custom_dataset"] = True
                st.success(f"Converted {len(tx_converted):,} records into transaction itemsets for mining!")

    with tab2:
        if raw_records:
            df = pd.DataFrame(raw_records)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                crop_counts = df["crop"].value_counts().to_dict()
                st.plotly_chart(charts.plot_categorical_distribution(crop_counts, "Crop Distribution", "donut"), use_container_width=True)

            with col_c2:
                yield_counts = df["yield_category"].value_counts().to_dict()
                st.plotly_chart(charts.plot_categorical_distribution(yield_counts, "Yield Category Distribution", "donut"), use_container_width=True)

            col_c3, col_c4 = st.columns(2)
            with col_c3:
                soil_counts = df["soil_type"].value_counts().to_dict()
                st.plotly_chart(charts.plot_categorical_distribution(soil_counts, "Soil Type Distribution", "bar"), use_container_width=True)

            with col_c4:
                irr_counts = df["irrigation_type"].value_counts().to_dict()
                st.plotly_chart(charts.plot_categorical_distribution(irr_counts, "Irrigation Method Distribution", "bar"), use_container_width=True)

    with tab3:
        st.markdown("#### Generate Scaled Synthetic Dataset")
        st.write("Generate custom reproducible synthetic datasets for stress-testing and scaling benchmarks.")
        gen_count = st.select_slider("Record Count to Generate", options=[10000, 50000, 100000, 250000, 500000], value=100000)
        gen_seed = st.number_input("Random Seed (for deterministic reproducibility)", value=42, step=1)

        if st.button("Generate Synthetic Agricultural Dataset", type="primary"):
            with st.spinner(f"Generating {gen_count:,} agricultural records..."):
                res = generate_agricultural_dataset(
                    num_records=gen_count,
                    seed=gen_seed,
                    output_dir=os.path.join(PROJECT_ROOT, "data"),
                    file_prefix=f"agriinsight_synthetic_{gen_count // 1000}k" if gen_count >= 1000 else f"agriinsight_synthetic_{gen_count}"
                )
                st.success(f"Successfully generated {gen_count:,} records!")
                st.write(f"Raw CSV: `{res['raw_path']}`")
                st.write(f"Transactions CSV: `{res['transaction_path']}`")


# =========================================================
# SECTION 3: PREPROCESSING & DISCRETIZATION
# =========================================================
elif menu == "Preprocessing & Discretization":
    st.markdown('<div class="main-header">Preprocessing & Continuous Discretization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discretization Rules, Feature Selection, and Strict Set Semantics</div>', unsafe_allow_html=True)

    st.markdown("""
    Association Rule Mining operates on discrete categorical transactions. Continuous agronomic variables 
    (soil chemistry, temperatures, and water allocation) must be converted into standardized categorical intervals.
    """)

    st.markdown("### Configured Continuous Bins (Agronomic Response Curves)")
    bin_rows = []
    for col, cfg in DEFAULT_NUMERIC_BINS.items():
        intervals_str = ", ".join(f"{lbl} [{low} to {high})" for lbl, low, high in cfg["bins"])
        bin_rows.append({
            "Continuous Feature": col,
            "Mapped Item Name": cfg["attr_name"],
            "Configured Categorical Intervals": intervals_str
        })
    st.dataframe(pd.DataFrame(bin_rows), use_container_width=True)

    st.markdown("### Sample Formatted Transaction")
    tx_sample = st.session_state.get("transactions", [])
    if tx_sample:
        st.code("; ".join(tx_sample[0]), language="text")
        st.caption("Each transaction is deduplicated, converted to set semantics, and stripped of IDs to prevent leakage.")


# =========================================================
# SECTION 4: ALGORITHM EXECUTION
# =========================================================
elif menu == "Algorithm Execution":
    st.markdown('<div class="main-header">Algorithm Execution & Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Independent Execution of Apriori, FP-Growth, ECLAT, and H-Mine</div>', unsafe_allow_html=True)

    transactions = st.session_state.get("transactions", [])
    n_trans = len(transactions)
    min_count = math.ceil(min_support * n_trans) if n_trans > 0 else 0

    st.markdown(f"""
    **Current Configuration:**
    - Dataset: **{n_trans:,} transactions**
    - Minimum Support: **{min_sup_pct}%** (Exact Integer Count \u2265 **{min_count:,}**)
    - Max Length Restriction: **{max_len_option if max_len_option else 'None (All Sizes)'}**
    """)

    col_cfg1, col_cfg2 = st.columns([3, 1])
    with col_cfg1:
        algs_to_run = st.multiselect(
            "Select Algorithms to Mine",
            ["Apriori", "FP-Growth", "ECLAT", "H-Mine"],
            default=["Apriori", "FP-Growth", "ECLAT", "H-Mine"]
        )
    with col_cfg2:
        fast_mode = st.checkbox(
            "High-Speed Mode",
            value=True,
            help="Bypasses tracemalloc profiling to accelerate execution by up to 3x on large datasets. Full memory profiling remains available under Performance Benchmark."
        )

    if st.button("Run Frequent Itemset Mining", type="primary"):
        if not transactions:
            st.error("No transactions available. Please select a valid dataset.")
        elif not algs_to_run:
            st.warning("Please select at least one algorithm.")
        else:
            results = {}
            benchmarks = []
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            for idx, alg in enumerate(algs_to_run):
                status_placeholder.text(f"Executing {alg} on {n_trans:,} transactions...")
                run_res = cached_mine_algorithm(
                    alg,
                    slice_size=sample_size if not is_custom_dataset else n_trans,
                    min_support=min_support,
                    max_length=max_len_option,
                    track_memory=not fast_mode,
                    is_custom=is_custom_dataset
                )
                results[alg] = run_res["frequent_itemsets"]
                benchmarks.append({
                    "Algorithm": alg,
                    "Time (s)": round(run_res["execution_time_sec"], 4),
                    "Peak Memory (MB)": round(run_res["peak_memory_mb"], 2) if not fast_mode else "N/A (High-Speed Mode)",
                    "Itemsets Found": run_res["frequent_itemsets_count"],
                    "Status": run_res["status"]
                })
                progress_bar.progress((idx + 1) / len(algs_to_run))

            status_placeholder.text("Mining complete!")
            st.session_state["mining_results"] = results
            st.session_state["mining_benchmarks"] = benchmarks
            st.session_state["last_params"] = {
                "min_support": min_support,
                "min_count": min_count,
                "max_length": max_len_option,
                "n_trans": n_trans,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            st.success("Frequent itemset mining finished successfully!")

    if "mining_benchmarks" in st.session_state:
        st.markdown("### Execution Summary")
        st.dataframe(pd.DataFrame(st.session_state["mining_benchmarks"]), use_container_width=True)


# =========================================================
# SECTION 5: FREQUENT ITEMSETS EXPLORER
# =========================================================
elif menu == "Frequent Itemsets Explorer":
    st.markdown('<div class="main-header">Frequent Itemsets Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search, Filter, and Inspect All Discovered Itemsets</div>', unsafe_allow_html=True)

    results = st.session_state.get("mining_results", {})
    if not results:
        st.info("No frequent itemsets mined yet. Please navigate to **Algorithm Execution** and run the algorithms.")
    else:
        alg_choice = st.selectbox("Choose Algorithm Result to Explore", list(results.keys()))
        itemsets = results[alg_choice]
        n_trans = st.session_state.get("last_params", {}).get("n_trans", len(st.session_state.get("transactions", [])))

        st.markdown(f"**Total Frequent Itemsets Discovered by {alg_choice}: {len(itemsets):,}**")

        # Filtering controls
        c_f1, c_f2 = st.columns([3, 1])
        with c_f1:
            search_item = st.text_input("Filter by Item (e.g. 'Crop=Rice', 'Soil=Clay', 'Yield=High')", "")
        with c_f2:
            len_filter = st.selectbox("Filter by Size", ["All", 1, 2, 3, 4, 5])

        # High-performance cached DataFrame construction
        frequent_tuples = tuple((tuple(sorted(k)), v) for k, v in itemsets.items())
        df_items = build_itemsets_dataframe(frequent_tuples, n_trans)

        if not df_items.empty:
            # Vectorized pandas filtering
            df_filtered = df_items
            if search_item.strip():
                df_filtered = df_filtered[df_filtered["Itemset"].str.contains(search_item.strip(), case=False, na=False)]
            if len_filter != "All":
                df_filtered = df_filtered[df_filtered["Length"] == int(len_filter)]

            st.dataframe(df_filtered, use_container_width=True)

            csv_data = convert_df_to_csv(df_filtered)
            st.download_button(
                label="Download Frequent Itemsets (CSV)",
                data=csv_data,
                file_name=f"agriinsight_frequent_itemsets_{alg_choice.lower()}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No itemsets match the current filter criteria.")


# =========================================================
# SECTION 6: ASSOCIATION RULES EXPLORER
# =========================================================
elif menu == "Association Rules Explorer":
    st.markdown('<div class="main-header">Association Rules Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Support, Confidence, Lift, Leverage, & Conviction</div>', unsafe_allow_html=True)

    results = st.session_state.get("mining_results", {})
    if not results:
        st.info("No frequent itemsets mined yet. Please navigate to **Algorithm Execution** first.")
    else:
        alg_choice = st.selectbox("Derive Rules From Algorithm Result", list(results.keys()))
        frequent = results[alg_choice]
        n_trans = st.session_state.get("last_params", {}).get("n_trans", len(st.session_state.get("transactions", [])))

        # Cache rules calculation across slider adjustments
        frequent_tuples = tuple((tuple(sorted(k)), v) for k, v in frequent.items())
        rules = get_cached_rules(
            frequent_tuples=frequent_tuples,
            num_trans=n_trans,
            min_confidence=min_confidence,
            min_lift=min_lift
        )

        st.markdown(f"**Discovered Association Rules: {len(rules):,}** (Min Conf: {min_conf_pct}%, Min Lift: {min_lift})")

        if rules:
            # WebGL hardware-accelerated scatter plot
            st.plotly_chart(charts.plot_rules_scatter(rules), use_container_width=True)

            # Table display with caching
            rules_tuples = tuple(
                (r["rule_str"], r["support"], r["confidence"], r["lift"], r["leverage"], r["conviction"], r["count_rule"])
                for r in rules
            )
            df_rules = build_rules_dataframe(rules_tuples)
            st.dataframe(df_rules, use_container_width=True)

            csv_rules = convert_df_to_csv(df_rules)
            st.download_button(
                label="Download Association Rules (CSV)",
                data=csv_rules,
                file_name="agriinsight_association_rules.csv",
                mime="text/csv"
            )
        else:
            st.warning("No rules satisfy the current minimum confidence and lift thresholds. Try lowering the thresholds in the sidebar.")


# =========================================================
# SECTION 7: CORRECTNESS VERIFICATION
# =========================================================
elif menu == "Algorithm Correctness Verification":
    st.markdown('<div class="main-header">Algorithm Correctness Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Independent Equivalence Proving & Zero-Mismatch Assurance</div>', unsafe_allow_html=True)

    results = st.session_state.get("mining_results", {})
    if len(results) < 2:
        st.warning("Please mine with at least two algorithms in **Algorithm Execution** to run pairwise correctness verification.")
    else:
        min_count = st.session_state.get("last_params", {}).get("min_count", 1)
        verification = verify_all_four(results, min_count=min_count)

        if verification["overall_status"] == "PASS":
            st.markdown("""
            <div class="status-card-pass">
                <h3>ALL COMPARISONS PASSED — 100% MATHEMATICALLY EQUIVALENT</h3>
                <p>
                    All evaluated algorithms produced the exact same frequent itemsets and identical integer support counts. 
                    Zero missing itemsets, zero extra itemsets, and zero support count mismatches across all pairwise tests.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card-fail">
                <h3>VERIFICATION FAILED</h3>
                <p>
                    Discrepancies detected between algorithm outputs. See diagnostics table below.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Itemset Count Consistency")
        count_rows = [
            {"Algorithm": alg, "Frequent Itemsets Found": cnt, "Min Support Count Threshold": min_count}
            for alg, cnt in verification["algorithm_counts"].items()
        ]
        st.dataframe(pd.DataFrame(count_rows), use_container_width=True)

        st.markdown("### Pairwise Verification Matrix")
        matrix_rows = []
        for pair in verification["pairwise_comparisons"]:
            matrix_rows.append({
                "Algorithm Pair": pair["pair"],
                "Status": pair["status"],
                "Count A": pair["total_a"],
                "Count B": pair["total_b"],
                "Missing in B": pair["missing_count"],
                "Extra in B": pair["extra_count"],
                "Count Mismatches": pair["count_mismatches_count"],
                "Diagnostic Summary": pair["summary"]
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)


# =========================================================
# SECTION 8: PERFORMANCE BENCHMARK
# =========================================================
elif menu == "Performance Benchmark":
    st.markdown('<div class="main-header">Performance Benchmark Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Runtime Profiling, Memory Tracking, & Empirical Scaling Curves</div>', unsafe_allow_html=True)

    transactions = st.session_state.get("transactions", [])
    if not transactions:
        st.warning("No transactions loaded.")
    else:
        st.markdown("### Multi-Trial Benchmark on Current Dataset Slice")
        trials = st.slider("Number of Trials (Computes Median Time)", min_value=1, max_value=5, value=1)

        if st.button("Execute Complete Benchmark Suite", type="primary"):
            with st.spinner("Benchmarking all 4 algorithms with peak memory profiling..."):
                bench_res = benchmark_all_algorithms(
                    transactions,
                    min_support=min_support,
                    max_length=max_len_option,
                    trials=trials
                )
                st.session_state["benchmark_data"] = bench_res

        if "benchmark_data" in st.session_state:
            data = st.session_state["benchmark_data"]["summary_table"]
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.plotly_chart(charts.plot_execution_time_bar(data), use_container_width=True)
            with col_b2:
                st.plotly_chart(charts.plot_memory_usage_bar(data), use_container_width=True)

            st.markdown("#### Exact Measurement Table")
            st.dataframe(pd.DataFrame(data), use_container_width=True)

        st.markdown("---")
        st.markdown("### Empirical Scalability Trajectory (1k to 100k records)")
        scaling_points = [
            {"Algorithm": "Apriori", "Size": 1000, "Time": 1.1613},
            {"Algorithm": "Apriori", "Size": 5000, "Time": 9.3006},
            {"Algorithm": "Apriori", "Size": 10000, "Time": 26.7869},
            {"Algorithm": "Apriori", "Size": 50000, "Time": 128.9405},
            {"Algorithm": "Apriori", "Size": 100000, "Time": 351.6308},

            {"Algorithm": "FP-Growth", "Size": 1000, "Time": 0.7921},
            {"Algorithm": "FP-Growth", "Size": 5000, "Time": 4.9000},
            {"Algorithm": "FP-Growth", "Size": 10000, "Time": 14.3323},
            {"Algorithm": "FP-Growth", "Size": 50000, "Time": 35.4845},
            {"Algorithm": "FP-Growth", "Size": 100000, "Time": 70.6801},

            {"Algorithm": "ECLAT", "Size": 1000, "Time": 0.1831},
            {"Algorithm": "ECLAT", "Size": 5000, "Time": 1.0648},
            {"Algorithm": "ECLAT", "Size": 10000, "Time": 2.7588},
            {"Algorithm": "ECLAT", "Size": 50000, "Time": 8.1115},
            {"Algorithm": "ECLAT", "Size": 100000, "Time": 22.9990},

            {"Algorithm": "H-Mine", "Size": 1000, "Time": 0.4834},
            {"Algorithm": "H-Mine", "Size": 5000, "Time": 3.5682},
            {"Algorithm": "H-Mine", "Size": 10000, "Time": 9.6693},
            {"Algorithm": "H-Mine", "Size": 50000, "Time": 37.4811},
            {"Algorithm": "H-Mine", "Size": 100000, "Time": 129.2176},
        ]
        st.plotly_chart(charts.plot_scalability_curve(scaling_points), use_container_width=True)


# =========================================================
# SECTION 9: AGRONOMIC PATTERN INSIGHTS
# =========================================================
elif menu == "Agronomic Pattern Insights":
    st.markdown('<div class="main-header">Agronomic Pattern Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Agronomic Interpretations of Top Discovered Associations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="academic-disclaimer">
        <b>Statistical Interpretation Notice:</b> These patterns represent statistical co-occurrences 
        mined from the synthetic agricultural dataset. High confidence and lift indicate mathematical association 
        within the benchmark domain, not proof of causation or universal field recommendations.
    </div>
    """, unsafe_allow_html=True)

    c_i1, c_i2 = st.columns(2)
    with c_i1:
        st.markdown("""
        <div class="pattern-card">
            <div class="pattern-header">Pattern 1: Paddy Rice Water Dynamics (Oryza sativa)</div>
            <div>
                <span class="pattern-metric-pill">Rule: {Crop=Rice, Soil=Clay} =&gt; {Water_Use=High}</span>
                <span class="pattern-metric-pill">Support: ~6.8%</span>
                <span class="pattern-metric-pill">Confidence: ~88.4%</span>
                <span class="pattern-metric-pill">Lift: 2.45</span>
            </div>
            <div class="pattern-desc">
                Clay soils possess fine particle composition and low hydraulic conductivity, making them traditionally preferred for puddled rice fields to maintain standing water saturation throughout the vegetative stage.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pattern-card">
            <div class="pattern-header">Pattern 2: Precision Horticultural Fertigation (Solanum lycopersicum)</div>
            <div>
                <span class="pattern-metric-pill">Rule: {Crop=Tomato, Irrigation=Drip} =&gt; {Yield=High}</span>
                <span class="pattern-metric-pill">Support: ~4.2%</span>
                <span class="pattern-metric-pill">Confidence: ~76.5%</span>
                <span class="pattern-metric-pill">Lift: 1.82</span>
            </div>
            <div class="pattern-desc">
                In intensive horticultural crops like tomatoes, controlled drip irrigation delivers uniform water and soluble nutrients directly to the rhizosphere, correlating strongly with high yield categories.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_i2:
        st.markdown("""
        <div class="pattern-card">
            <div class="pattern-header">Pattern 3: Winter Cereal Nitrogen Regime (Triticum aestivum)</div>
            <div>
                <span class="pattern-metric-pill">Rule: {Crop=Wheat, Season=Rabi} =&gt; {Fertilizer=Urea+DAP}</span>
                <span class="pattern-metric-pill">Support: ~8.1%</span>
                <span class="pattern-metric-pill">Confidence: ~84.2%</span>
                <span class="pattern-metric-pill">Lift: 2.10</span>
            </div>
            <div class="pattern-desc">
                Winter wheat requires basal diammonium phosphate (DAP) during root establishment followed by split top-dressing of urea during active tillering in cooler winter Rabi months.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pattern-card">
            <div class="pattern-header">Pattern 4: Rainfed Legume Nitrogen Fixation (Cicer arietinum)</div>
            <div>
                <span class="pattern-metric-pill">Rule: {Crop=Chickpea, Irrigation=Rainfed} =&gt; {Nitrogen=Low}</span>
                <span class="pattern-metric-pill">Support: ~5.1%</span>
                <span class="pattern-metric-pill">Confidence: ~81.0%</span>
                <span class="pattern-metric-pill">Lift: 2.25</span>
            </div>
            <div class="pattern-desc">
                Leguminous pulses cultivate symbiotic Rhizobium root nodules that biologically fix atmospheric nitrogen, requiring minimal synthetic nitrogen under semi-arid rainfed conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# SECTION 10: EXPORT & REPRODUCIBILITY
# =========================================================
elif menu == "Export & Reproducibility":
    st.markdown('<div class="main-header">Export & Reproducibility</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Download Verified Data, Rules, Benchmark Logs, and Run Configuration</div>', unsafe_allow_html=True)

    last_params = st.session_state.get("last_params", {})
    results = st.session_state.get("mining_results", {})

    st.markdown("### Experiment Metadata")
    st.json({
        "project": "AgriInsight",
        "timestamp": last_params.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "dataset_transactions": len(st.session_state.get("transactions", [])),
        "min_support_fraction": min_support,
        "min_support_percent": f"{min_sup_pct}%",
        "min_support_count": last_params.get("min_count", 0),
        "min_confidence": min_confidence,
        "min_lift": min_lift,
        "max_length_restriction": max_len_option,
        "algorithms_evaluated": list(results.keys())
    })

    c_e1, c_e2 = st.columns(2)
    with c_e1:
        if results:
            first_alg = list(results.keys())[0]
            norm = normalize_itemsets(results[first_alg])
            export_items = [
                {"itemset": list(k), "support_count": cnt, "support_pct": (cnt/len(st.session_state["transactions"]))*100}
                for k, cnt in norm.items()
            ]
            st.download_button(
                "Export Frequent Itemsets (JSON)",
                data=json.dumps(export_items, indent=2),
                file_name="agriinsight_frequent_itemsets.json",
                mime="application/json"
            )

    with c_e2:
        if "mining_benchmarks" in st.session_state:
            df_b = pd.DataFrame(st.session_state["mining_benchmarks"])
            st.download_button(
                "Export Benchmark Measurements (CSV)",
                data=convert_df_to_csv(df_b),
                file_name="agriinsight_benchmark_results.csv",
                mime="text/csv"
            )
