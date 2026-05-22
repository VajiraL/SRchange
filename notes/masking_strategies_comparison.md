# Masking Strategies and Global SR Statistics

This document records the evolution of masking strategies used for global root zone water storage capacity (SR) change attribution and compares their impact on final results.

## Overview of Strategies

### 1. Dynamic Masking (Initial Approach)
*   **Definition:** Masking based on the available non-NaN pixels in each individual scenario prediction file.
*   **Logic:** For every model member and every scenario, the global mean is calculated over valid vegetated pixels in that specific file. This is done by masking raw predictions based on the non-vegetated class of the LU data for corresponsing period.
*   **Pros:** Uses all available information for each scenario.
*   **Cons:** The underlying domain changes between scenarios. Specifically, the `1991_2020` period (used for present-day baseline) has ~83,000 more NaNs than other scenarios because it consists of merged cross-validation test folds, leading to small gaps where data was withheld.
*   **Key Scripts:** 
    *   `scripts/DEPRECATE/calculate_per_model_stats_dynamic_mask.py`
    *   `scripts/DEPRECATE/analyze_change_uncertainty_incremental.py` (older versions)
*   **Key Data:** 
    *   `output_SR/DELELE/sr_scn_per_model_stats_k6_masked.csv`

### 2. Static Masking (Current Standard - v2)
*   **Definition:** Using a single, binary mask derived from the training data availability.
*   **Logic:** A static mask is created from `data/SR-target/training_domain_mask.nc` (where 1 = keep). Only the 5,169,061 pixels present in the training set are used for all mean calculations across all 168 scenarios.
*   **Pros:** Ensures consistent area-weighted comparisons. Eliminates artifacts caused by shifting domains.
*   **Cons:** Excludes a small number of valid predictions in historical/future scenarios that aren't in the training mask.
*   **Key Scripts:** 
    *   `scripts/modeling/calculate_per_model_stats.py`
    *   `scripts/analysis/analyze_change_uncertainty.py`
    *   `scripts/analysis/aggregate_attribution_results.py`
*   **Key Data:** 
    *   `data/SR-target/training_domain_mask.nc`
    *   `output_SR/sr_scn_per_model_stats_k6.csv`
    *   `output_SR/sr_change_uncertainty_all.csv`

## Comparison of Global Statistics

The choice of masking strategy significantly impacts the global attribution of SR change, particularly for the transition from the **Pre-Industrial (PI)** era to the **Present Day (1991-2020)**.

### SR Change Attribution (PI -> 1991-2020)

| Component | Dynamic Mask (mm) | **Static Mask (v2) (mm)** | Delta (mm) |
| :--- | :---: | :---: | :---: |
| **Net Change** | -1.33 | **+1.89** | +3.22 |
| Climate Change | +7.62 | **+9.06** | +1.44 |
| Land Use Change | -11.11 | **-8.34** | +2.77 |
| Interaction | +2.16 | **+1.18** | -0.98 |

### Key Takeaway
Using a **Static Mask** flips the sign of the net global SR change from negative to positive. This is because the Dynamic Mask was inadvertently penalizing the `1991_2020` mean by including more NaNs in specific high-SR or low-SR regions that weren't missing in the PI scenario. The Static Mask provides a more robust and scientifically defensible baseline for climate and land-use change attribution.
