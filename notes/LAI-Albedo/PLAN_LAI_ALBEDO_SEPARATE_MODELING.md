# Plan: Separate LAI and Albedo Modeling

This plan details the independent modeling of Leaf Area Index (LAI) and Albedo using the climate-clustered XGBoost framework.

## 1. Target Variables
*   **LAI Target:** `data/LAI_mean_95th_2011_2015.nc` (Mean of annual 95th percentiles, QC=0 filtered).
*   **Albedo Target:** `data/albedo_mean_10th_2011_2020.nc` (Mean of annual 10th percentiles).

## 2. Spatial Masking & CV
*   **Independent Masks:** Instead of a common mask, each modeling task will define its training area based on the non-NaN pixels of its respective target map.
*   **LAI Mask:** Not expected to be too restrictive. Only a few QC fails.
*   **Albedo Mask:** Near-global coverage (excluding oceans).
*   **Cross-Validation:** 5-fold random spatial CV will be performed independently for each target.
*   **Ensemble:** Only run ONE ensemble (Ens 1).

## 3. Modeling Framework (XGBoost)
*   **Core Architecture:** 1 Global model + 6 Cluster-specific models based on k=6 climate cluster map (using `data/climate_clusters.zarr`).
*   **Feature Set (47 Predictors):**
    *   **Static (3):** `elev`, `slope`, `roughness`.
    *   **Climate (40):** Standard suite (Precip, Temp, PET, Aridity, etc.) from `data/climate_features.zarr`.
    *   **Land Use (4):** `gHM`, `irr_frac`, `rain_frac`, `forest_age` from `data/LU_features/`.
*   **Hyperparameters:** Consistent with the $S_R$ framework (Poisson objective for LAI/Albedo to ensure positivity, or standard Regressor if preferred).

## 4. Scenario Predictions & Aggregation
Each model will generate global 0.05° predictions for 5 scenarios. Aggregation logic is as follows:

*   **Scenario 1 (Present: 1991-2020 Climate × 1991-2020 LU):** 
    *   Mosaic a full global map using only the 'test' fold predictions from each of the 5 folds (Out-of-Sample prediction).
*   **Scenarios 2-5 (Historical & PI Mixes):**
    *   Average the predictions from all 5 folds to create a more robust ensemble mean.

### Scenario List:
1.  **Present:** 1991-2020 Climate × 1991-2020 LU.
2.  **Historical:** 1961-1990 Climate × 1961-1990 LU (for 1980s validation).
3.  **PI Mix 1:** 1901-1930 Climate × Pre-industrial LU.
4.  **PI Mix 2:** 1991-2020 Climate × Pre-industrial LU.
5.  **PI Mix 3:** 1901-1930 Climate × 1991-2020 LU.

## 5. Verification & Outputs
*   **Metrics:** $R^2$ and RMSE for the 2011-2020 baseline period.
*   **Historical Validation:** Compare "Historical" scenario predictions (Scenario 2) against the observed averages for 1982-1990 (LAI) and 1981-1990 (Albedo).
*   **Structure:**
    *   `output_lai/`: `.nc` maps, model artifacts, metrics.
    *   `output_albedo/`: `.nc` maps, model artifacts, metrics.
    *   `scripts/albedo/: all albedo scripts including modelling and aggregation
    *   `scripts/lai/: all LAI scripts including modelling and aggregation

## 6. Implementation Steps
1.  Develop `scripts/lai/train_predict_lai.py`.
2.  Develop `scripts/albedo/train_predict_albedo.py`.
3.  Execute training and scenario prediction.
4.  Generate validation reports and difference maps.
