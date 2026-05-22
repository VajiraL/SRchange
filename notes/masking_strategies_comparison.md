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

### 2. Static Masking (v2)
*   **Definition:** Using a single, binary mask derived from the training data availability.
*   **Logic:** A static mask is created from `data/SR-target/training_domain_mask.nc` (where 1 = keep). Initially, 5,169,061 pixels were used.
*   **Cons:** Led to a ~121,000 pixel discrepancy between historical/future scenarios and the training domain, and did not account for the 1991-2020 baseline being a merged product of 5 test folds.

### 3. Super-Static Masking (Current Standard - v3)
*   **Definition:** The intersection of valid pixels across all 168 scenarios and the training domain.
*   **Logic:** Generated as `data/SR-target/super_static_mask.nc` (4,787,114 pixels).
*   **Baseline Logic:** The 1991-2020 baseline is reconstructed using 5 mutually exclusive test-set ensemble members (reconstructed from 25 folds), ensuring that training pixels are never included in the baseline mean.
*   **Pros:** Ensures 100% pixel-perfect consistency across every single transition and scenario. Aligns statistics perfectly with the aggregated Zarr outputs.
*   **Key Scripts:** 
    *   `scripts/modeling/apply_super_static_mask.py`
    *   `scripts/modeling/calculate_per_model_stats.py`
    *   `scripts/analysis/analyze_change_uncertainty.py`
*   **Key Data:** 
    *   `data/SR-target/super_static_mask.nc`
    *   `output_SR/sr_scn_per_model_stats_k6.csv`
    *   `output_SR/sr_change_attribution/sr_change_uncertainty_all.csv`

## Comparison of Global Statistics

The choice of masking strategy significantly impacts the global attribution of SR change, particularly for the transition from the **Pre-Industrial (PI)** era to the **Present Day (1991-2020)**.

### SR Change Attribution (PI -> 1991-2020)

| Component | Dynamic Mask (mm) | Static Mask (v2) (mm) | **Super-Static (v3) (mm)** |
| :--- | :---: | :---: | :---: |
| **Net Change** | -1.33 | +1.89 | **+0.42** |
| Climate Change | +7.62 | +9.06 | **+9.31** |
| Land Use Change | -11.11 | -8.34 | **-10.03** |
| Interaction | +2.16 | +1.18 | **+1.14** |

### Key Takeaway
The **Super-Static Mask (v3)** provides the most rigorous assessment by ensuring that the exact same 4.78 million pixels are compared across all scenarios. By specifically handling the 1991-2020 baseline as a test-set ensemble reconstruction, we eliminate the bias introduced by training pixels (which are not available in other scenario predictions). This results in a slightly smaller but still positive net SR change (+0.42 mm) compared to the initial v2 static mask.
