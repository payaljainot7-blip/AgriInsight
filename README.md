# AgriInsight — Agricultural Pattern Discovery Using Association Rule Mining

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Verification: Equivalence Proven](https://img.shields.io/badge/Verification-100%25%20Equivalence%20PASS-success.svg)](#correctness-verification)

> **Academic Disclaimer**: All datasets provided and generated within AgriInsight are entirely **synthetic** and designed strictly for computer science coursework, algorithmic complexity evaluation, and association-rule-mining benchmarking. The discovered patterns represent statistical co-occurrences within synthetic data and **must never be interpreted as verified real-world agronomic recommendations or guarantees of crop yield**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Objectives & Research Questions](#key-objectives--research-questions)
3. [Technology Stack](#technology-stack)
4. [Dataset Architecture & Planted Patterns](#dataset-architecture--planted-patterns)
5. [Data Preprocessing & Continuous Discretization](#data-preprocessing--continuous-discretization)
6. [The Four Mining Algorithms](#the-four-mining-algorithms)
   - [Apriori](#1-apriori)
   - [FP-Growth](#2-fp-growth)
   - [ECLAT](#3-eclat)
   - [H-Mine](#4-h-mine)
7. [Association Rule Metrics Formalism](#association-rule-metrics-formalism)
8. [Correctness Verification Protocol](#correctness-verification-protocol)
9. [Performance Benchmarking & Empirical Findings](#performance-benchmarking--empirical-findings)
10. [Streamlit Application Walkthrough](#streamlit-application-walkthrough)
11. [Installation & Execution Guide](#installation--execution-guide)
12. [Automated Test Suite](#automated-test-suite)
13. [Limitations & Future Scope](#limitations--future-scope)

---

## Project Overview

**AgriInsight** is an interactive, academic-grade data-mining platform that discovers recurring combinations of crop varieties, soil types, fertilizers, irrigation practices, micro-climates, and yield outcomes.

Rather than predicting yields using opaque black-box machine learning models, AgriInsight employs **Association Rule Mining (ARM)** to extract transparent, explainable, and human-interpretable co-occurrence rules.

A cornerstone contribution of this project is the **independent implementation of four distinct frequent-itemset mining algorithms in pure Python**:
1. **Apriori** (Level-wise candidate generation & subset pruning)
2. **FP-Growth** (Compact prefix-tree representation & pattern growth)
3. **ECLAT** (Vertical transaction-ID set intersections)
4. **H-Mine** (In-memory hyper-structure and dynamically linked projections)

AgriInsight proves that when evaluated on identical preprocessed transactions with identical minimum support thresholds, **all four algorithms yield 100% mathematically identical frequent itemsets and exact integer support counts**.

---

## Key Objectives & Research Questions

AgriInsight addresses five central research questions (RQs):
* **RQ1 (Equivalence)**: Do Apriori, FP-Growth, ECLAT, and H-Mine discover the exact same complete frequent itemsets and exact integer counts?
* **RQ2 (Scalability)**: How does runtime scale across transaction volumes from $1,000$ to $100,000$+ records?
* **RQ3 (Threshold Sensitivity)**: How does execution time and itemset expansion behave as minimum support decreases?
* **RQ4 (Data Structure Trade-offs)**: What are the memory and CPU efficiency trade-offs between vertical inverted indexes (ECLAT), prefix trees (FP-Growth), hyper-structures (H-Mine), and candidate arrays (Apriori)?
* **RQ5 (Agronomic Discoveries)**: Can association rules capture plausible agricultural correlations (e.g., paddy rice affinity for clay soils and high water use)?

---

## Technology Stack

- **Core Programming Language**: Python 3.11+
- **Interactive Web Interface**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Visual Analytics**: Plotly Interactive Charts
- **Profiling & Memory Diagnostics**: Standard Library `time.perf_counter` & `tracemalloc`
- **Testing & Verification**: Pytest & Python Standard `unittest`
- **External Dependency Independence**: All four mining algorithms are implemented from scratch in pure Python without reliance on external black-box libraries (`mlxtend`, `apyori`, etc.).

---

## Dataset Architecture & Planted Patterns

AgriInsight includes a reproducible **100,000-record agricultural dataset**:
- `data/agriinsight_synthetic_100k_raw.csv`: Continuous numeric & categorical records.
- `data/agriinsight_synthetic_100k_transactions.csv`: Pre-binned semicolon-separated item transactions.
- `data/generate_dataset.py`: Reproducible generator supporting scale-up to 250,000 or 500,000 records.

### Attributes (18 Features)
- **Identifiers**: `transaction_id`
- **Spatial & Temporal**: `region`, `season`
- **Crop & Soil**: `crop`, `soil_type`
- **Agricultural Practices**: `irrigation_type`, `fertilizer_type`, `pesticide_intensity`
- **Weather & Climate**: `rainfall_category`, `temperature_c`, `humidity_percent`
- **Soil Nutrients & Chemistry**: `soil_ph`, `nitrogen_kg_ha`, `phosphorus_kg_ha`, `potassium_kg_ha`
- **Inputs & Output**: `fertilizer_kg_ha`, `water_use_m3_ha`, `yield_category`

### Intentionally Planted Patterns
1. **Rice Water Affinity**: `Crop=Rice` heavily correlates with `Soil=Clay`, `Irrigation=Flood`, `Water_Use=High`, and `Season=Kharif`.
2. **Precision Horticulture**: `Crop=Tomato` correlates with `Irrigation=Drip`, `Fertilizer=Water_Soluble_NPK`, `Potassium=High`, and `Yield=High`.
3. **Winter Cereal Dynamics**: `Crop=Wheat` associates with `Season=Rabi`, `Soil=Alluvial`, and `Fertilizer=Urea+DAP`.
4. **Legume Nitrogen-Fixation**: `Crop=Chickpea` associates with `Fertilizer=DAP+Potash`, `Irrigation=Rainfed`, and `Nitrogen=Low`.
5. **Sugarcane Requirements**: `Crop=Sugarcane` correlates with `Water_Use=High`, `Nitrogen=High`, and `Soil=Alluvial`.

---

## Data Preprocessing & Continuous Discretization

To transform raw continuous records into valid transactions:
1. **Identifier Elimination**: `transaction_id` and `farm_id` are removed to prevent trivial self-identifying associations.
2. **Agronomic Discretization**:
   - **Soil pH**: Acidic ($< 6.2$), Neutral ($6.2 - 7.5$), Alkaline ($> 7.5$)
   - **Nitrogen (kg/ha)**: Low ($< 80$), Medium ($80 - 130$), High ($\ge 130$)
   - **Phosphorus (kg/ha)**: Low ($< 35$), Medium ($35 - 65$), High ($\ge 65$)
   - **Potassium (kg/ha)**: Low ($< 35$), Medium ($35 - 75$), High ($\ge 75$)
   - **Temperature**: Low ($< 20^\circ\text{C}$), Moderate ($20 - 30^\circ\text{C}$), High ($\ge 30^\circ\text{C}$)
   - **Humidity**: Low ($< 45\%$), Moderate ($45 - 70\%$), High ($\ge 70\%$)
   - **Fertilizer Quantity**: Low ($< 120\,\text{kg/ha}$), Medium ($120 - 180\,\text{kg/ha}$), High ($\ge 180\,\text{kg/ha}$)
   - **Water Use**: Low ($< 2500\,\text{m}^3/\text{ha}$), Medium ($2500 - 4500\,\text{m}^3/\text{ha}$), High ($\ge 4500\,\text{m}^3/\text{ha}$)
3. **Strict Set Semantics**: Items within each transaction are deduplicated, enforcing set properties:
   $$\text{Transaction } t = \{\text{Crop=Rice}, \text{Soil=Clay}, \text{Irrigation=Flood}, \dots\}$$

---

## The Four Mining Algorithms

### 1. Apriori
- **Mechanism**: Level-wise candidate generation and test.
- **Join**: Generates $C_k$ by joining frequent itemsets in $L_{k-1}$ sharing a $(k-2)$ prefix.
- **Prune**: Verifies that every $(k-1)$ subset of a candidate exists in $L_{k-1}$ before scanning transactions.
- **Complexity**: Time $\mathcal{O}(2^{|I|} \cdot N)$, Space $\mathcal{O}(|C_k|)$.

### 2. FP-Growth
- **Mechanism**: Prefix-tree compression and pattern-growth divide-and-conquer.
- **FP-Tree**: Stores shared transaction prefixes with header table node-links.
- **Conditional Mining**: Traverses paths bottom-up to build conditional pattern bases and conditional FP-trees recursively.
- **Complexity**: Time typically $\mathcal{O}(N)$, Space $\mathcal{O}(\text{FP-Tree Nodes})$.

### 3. ECLAT
- **Mechanism**: Vertical transaction database with Transaction-ID (TID) sets.
- **Intersection**: Support count is computed as the cardinality of intersected TID-sets:
  $$\text{TID}(X \cup Y) = \text{TID}(X) \cap \text{TID}(Y)$$
- **Complexity**: Time $\mathcal{O}(|I|^2 \cdot \text{avg}(|\text{TID}|))$, Space $\mathcal{O}(|I| \cdot N)$.

### 4. H-Mine
- **Mechanism**: Main-memory hyper-structure (H-struct).
- **Hyper-Links**: Chains transactions sharing active items via linked pointers.
- **Projected Databases**: Traverses hyper-links and dynamically adjusts pointers in memory without physical tree replication.
- **Complexity**: Compact memory overhead with linear scalability in database size.

---

## Association Rule Metrics Formalism

For an implication $X \Rightarrow Y$ where $X \subset I, Y = I \setminus X, X \cap Y = \emptyset$:

| Metric | Mathematical Formula | Interpretation |
| :--- | :--- | :--- |
| **Support** | $\text{Supp}(X \Rightarrow Y) = \frac{\text{count}(X \cup Y)}{N}$ | Frequency of joint occurrence in the dataset. |
| **Confidence** | $\text{Conf}(X \Rightarrow Y) = \frac{\text{count}(X \cup Y)}{\text{count}(X)}$ | Conditional probability $P(Y \mid X)$. |
| **Lift** | $\text{Lift}(X \Rightarrow Y) = \frac{\text{Conf}(X \Rightarrow Y)}{\text{Supp}(Y)}$ | Ratio of observed co-occurrence to expectation under independence. |
| **Leverage** | $\text{Lev}(X \Rightarrow Y) = \text{Supp}(X \cup Y) - (\text{Supp}(X) \cdot \text{Supp}(Y))$ | Difference between observed and independent joint probability. |
| **Conviction** | $\text{Conv}(X \Rightarrow Y) = \frac{1 - \text{Supp}(Y)}{1 - \text{Conf}(X \Rightarrow Y)}$ | Degree of implication failure under independence. |

---

## Correctness Verification Protocol

AgriInsight features an automated verification engine (`core/verification.py`):
1. **Canonical Normalization**: Normalizes itemsets into sorted tuples before comparison.
2. **Exact Integer Support Threshold**:
   $$\text{min\_count} = \lceil \text{min\_support} \times N \rceil$$
3. **Six-Way Pairwise Verification Matrix**:
   - Apriori vs FP-Growth
   - Apriori vs ECLAT
   - Apriori vs H-Mine
   - FP-Growth vs ECLAT
   - FP-Growth vs H-Mine
   - ECLAT vs H-Mine
4. **Zero-Tolerance Criteria**: Any missing itemset, extra itemset, or count difference immediately triggers a **FAIL** status with full diagnostic traces.

---

## Performance Benchmarking & Empirical Findings

Empirical evaluation on 10,000 transactions at 5% support reveals:
- **ECLAT** delivers fastest raw throughput due to optimized C-level Python set intersections.
- **FP-Growth** demonstrates optimal scalability on long transactions by compressing prefixes.
- **H-Mine** operates with the lowest peak memory overhead.
- **Apriori** incurs candidate subset matching overhead as transaction lengths increase.

---

## Installation & Execution Guide

### Windows (PowerShell)

```powershell
# 1. Clone repository & navigate to workspace
cd c:\Users\Payal\OneDrive\Desktop\AgriInsight

# 2. Create virtual environment
py -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch AgriInsight Streamlit Application
streamlit run app.py
```

### Linux / macOS (Bash)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit
streamlit run app.py
```

---

## Automated Test Suite

Run the full automated test suite using `pytest` or `run_tests.py`:

```powershell
# Run with Pytest
py -m pytest -v tests

# Or run with the universal test runner
py run_tests.py
```

The test suite covers:
- Verification of manually derived 5-transaction ground truth.
- Empty datasets, empty transactions, duplicate items.
- Support threshold boundary conditions.
- Four-way equivalence on diverse datasets.
- Preprocessing and binning transformations.
- Synthetic generator reproducibility.

---

## Limitations & Future Scope

### Limitations
1. **Synthetic Nature**: Data reflects controlled mathematical distributions rather than empirical field trials.
2. **In-Memory Constraints**: Very low support thresholds ($< 0.5\%$) on 500k+ records can cause combinatorial itemset explosion in main memory.

### Future Scope
- Integration with live weather APIs and real IoT soil moisture sensors.
- Temporal association mining for multi-year crop rotation sequencing.
- Spatial rule mining incorporating GIS elevation and watershed maps.
