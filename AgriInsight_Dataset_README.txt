AgriInsight synthetic dataset package
=====================================

Files
-----
1. agriinsight_synthetic_100k_raw.csv
   100,000 synthetic agricultural records with numeric and categorical fields.
2. agriinsight_synthetic_100k_transactions.csv
   The same 100,000 records converted to semicolon-separated transaction items for association rule mining.

Important
---------
These records are synthetic and generated for coursework/testing. They are NOT observed farm data and must not be presented as real-world findings.
Correlations were intentionally introduced (for example, crop type affects likely fertilizer/irrigation choices) so association mining can demonstrate useful patterns.
Use the raw CSV for preprocessing experiments; use the transaction CSV directly for Apriori, FP-Growth, ECLAT, or other frequent-itemset algorithms.

Suggested starting thresholds
-----------------------------
Minimum support: 0.03 (3%)
Minimum confidence: 0.50 (50%)
Minimum lift: 1.10
Maximum itemset size: 4 or 5

Note on transaction parsing
---------------------------
Each transaction's `items` column contains semicolon-separated strings. Split on `;`.
Do not include `transaction_id` as an item.
Numeric columns in the raw CSV should be binned into categories before association rule mining.
