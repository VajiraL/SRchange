# Plan: LAI and Albedo Modeling Demonstration

This plan outlines the steps to model and predict Leaf Area Index (LAI) and Albedo using the established climate-clustered XGBoost framework. This is a simplified demonstration using a single ensemble, global scale, and 6 climate clusters.

## 1. Data Preparation (Averaging & Harmonization)
*   **Averaging:**
    *   **LAI (95th P):** Compute mean of annual 95th percentile maps for 1982-1990 and 2011-2015.
    *   **Albedo (10th P):** Compute mean of annual 10th percentile maps for 1981-1990 and 2011-2020.
*   **Harmonization:**
    *   Resample GIMMS LAI (1/12°) to the framework's standard 0.05° resolution using `rioxarray` or `xarray.interp`.
    *   Align Albedo (0.05°) with the reference grid.

## 2. Cross-Validation Setup
*   Generate random 5-fold cross-validation masks at 0.05° resolution, excluding oceans and permanent ice (consistent with $S_R$ masks).
*   Script: `scripts/data-preparation/calc_randomcv_masks_demo.py`

## 3. Modeling Framework
*   **Model:** XGBoost (CPU-based for simplicity, or GPU if available).
*   **Clusters:** Use the existing `data/climate_clusters.zarr` (k=6).
*   **Features:**
    *   Climate: Multi-period averages (1901-1930, 1961-1990, 1991-2020).
    *   Land Use (LU): Scenario-based (Pre-industrial, 1961-1990, 1991-2020).
*   **Targets:**
    *   LAI (2011-2015 mean)
    *   Albedo (2011-2020 mean)

## 4. Execution (Training & Prediction)
Create a unified script `scripts/modeling/run_lai_albedo_demo.py` to:
1.  Train 6 cluster-specific models for each target.
2.  Predict for the following 5 scenarios:
    *   **Scenario A:** 1991-2020 Clim x 1991-2020 LU (Present day)
    *   **Scenario B:** 1961-1990 Clim x 1961-1990 LU (Historical Validation)
    *   **Scenario C:** 1901-1930 Clim x Pre-industrial LU
    *   **Scenario D:** 1991-2020 Clim x Pre-industrial LU
    *   **Scenario E:** 1901-1930 Clim x 1991-2020 LU

## 5. Output Management
*   **LAI Outputs:** `output_lai/` (Predictions, model files, and validation metrics).
*   **Albedo Outputs:** `output_albedo/` (Predictions, model files, and validation metrics).

## 6. Verification
*   Evaluate model performance using RMSE and $R^2$ against the 2011-2020/2011-2015 observation targets.
*   Compare Historical (Scenario B) predictions with the 1981-1990 observation averages.
