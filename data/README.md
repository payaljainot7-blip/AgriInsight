# AgriInsight Agricultural Dataset Documentation

## 1. Academic Disclaimer & Integrity Notice

> [!CAUTION]
> **SYNTHETIC DATA DISCLAIMER**:
> All records in this dataset package are entirely synthetic, generated using controlled statistical distributions and planted agricultural patterns for computer science education, academic evaluation, and data-mining algorithm benchmarking. They **do not represent observed farm surveys** or real-world agricultural field measurements. Discovered association rules describe statistical co-occurrence within this synthetic domain and **must never be treated as verified agronomic recommendations or guarantees of crop yield**.

---

## 2. Dataset Schema & Attribute Descriptions

The synthetic dataset contains 18 comprehensive agricultural features capturing soil, weather, irrigation, input nutrients, and outcome indicators:

| Attribute | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| `transaction_id` | Identifier | Unique transaction key | `AGR000001` - `AGR100000` |
| `region` | Categorical | Agro-climatic zone | Madhya Pradesh, Vidarbha, Punjab, Karnataka, etc. |
| `season` | Categorical | Cropping season | Kharif (monsoon), Rabi (winter), Zaid (summer) |
| `crop` | Categorical | Cultivated crop | Rice, Wheat, Tomato, Maize, Chickpea, Sugarcane, etc. |
| `soil_type` | Categorical | Soil classification | Clay, Alluvial, Black, Loamy, Sandy |
| `irrigation_type` | Categorical | Water application method | Drip, Sprinkler, Flood, Rainfed |
| `fertilizer_type` | Categorical | Applied fertilizer blend | Urea+DAP, Water_Soluble_NPK, DAP+Potash, NPK_Complex |
| `pesticide_intensity` | Categorical | Intensity of chemical pest management | Low, Medium, High |
| `rainfall_category` | Categorical | Seasonal precipitation | Low (<400mm), Medium (400-800mm), High (>800mm) |
| `nitrogen_kg_ha` | Continuous | Soil/Applied Nitrogen content | Continuous kg/ha (binned: Low < 80, Med 80-130, High >= 130) |
| `phosphorus_kg_ha` | Continuous | Soil/Applied Phosphorus content | Continuous kg/ha (binned: Low < 35, Med 35-65, High >= 65) |
| `potassium_kg_ha` | Continuous | Soil/Applied Potassium content | Continuous kg/ha (binned: Low < 35, Med 35-75, High >= 75) |
| `soil_ph` | Continuous | Soil acidity / alkalinity | Continuous scale (binned: Acidic < 6.2, Neutral 6.2-7.5, Alkaline > 7.5) |
| `temperature_c` | Continuous | Mean ambient temperature | °C (binned: Low < 20°C, Moderate 20-30°C, High >= 30°C) |
| `humidity_percent` | Continuous | Mean relative humidity | % (binned: Low < 45%, Moderate 45-70%, High >= 70%) |
| `fertilizer_kg_ha` | Continuous | Total fertilizer quantity | kg/ha (binned: Low < 120, Medium 120-180, High >= 180) |
| `water_use_m3_ha` | Continuous | Volume of irrigation water | $m^3/ha$ (binned: Low < 2500, Medium 2500-4500, High >= 4500) |
| `yield_category` | Categorical | Harvested outcome classification | Low, Medium, High |

---

## 3. Planted Pattern Dependencies

To test the association-rule-mining capabilities of Apriori, FP-Growth, ECLAT, and H-Mine, controlled probabilistic patterns were introduced:

1. **Rice Water Affinity**: `Crop=Rice` heavily co-occurs with `Soil=Clay`, `Irrigation=Flood`, `Water_Use=High`, and `Season=Kharif`.
2. **Precision Horticulture**: `Crop=Tomato` co-occurs with `Irrigation=Drip`, `Fertilizer=Water_Soluble_NPK`, `Potassium=High`, and `Yield=High`.
3. **Winter Cereal Dynamics**: `Crop=Wheat` is predominantly planted in `Season=Rabi`, paired with `Fertilizer=Urea+DAP`, and `Temperature=Moderate`.
4. **Legume Nitrogen-Fixation**: `Crop=Chickpea` is correlated with `Fertilizer=DAP+Potash`, `Irrigation=Rainfed`, and `Nitrogen=Low`.
5. **Sugarcane Demands**: `Crop=Sugarcane` exhibits high co-occurrence with `Water_Use=High`, `Nitrogen=High`, and `Soil=Alluvial`.

---

## 4. File Structure & Included Datasets

The repository provides the complete 100,000-record dataset out of the box:
- `data/agriinsight_synthetic_100k_raw.csv`: Continuous numeric & categorical records for exploratory analysis and custom binning.
- `data/agriinsight_synthetic_100k_transactions.csv`: Pre-binned, semicolon-separated item transactions for direct mining.
- `data/agriinsight_synthetic_100k.csv`: Clean categorical analysis-ready dataset.

---

## 5. Dataset Generation & Scalability Instructions

You can generate any custom dataset size (e.g. 100,000, 250,000, or 500,000 rows) using `data/generate_dataset.py`:

```bash
# Generate default 100,000 records (reproducible seed=42)
py data/generate_dataset.py --num-records 100000 --seed 42 --output-dir data

# Generate 250,000 records for scalability benchmarking
py data/generate_dataset.py --num-records 250000 --seed 42 --output-dir data

# Generate 500,000 records for stress-testing main-memory algorithms
py data/generate_dataset.py --num-records 500000 --seed 42 --output-dir data
```
