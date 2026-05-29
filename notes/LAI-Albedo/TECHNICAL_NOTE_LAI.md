# Technical Note: Modeling Global Leaf Area Index (LAI) Sensitivities

## 1. Methodology
- **Target Variable:** Annual 95th percentile LAI derived from GIMMS 3g.v1 (1982–2015) at 0.05° resolution. Quality control (QC=0) applied to filter snow-contaminated and low-quality pixels.
- **Modeling Framework:** Independent Climate-Clustered XGBoost.
    - **Clustering:** 6 global climate clusters based on PCA-reduced bioclimatic variables.
    - **Ensemble:** 5-fold spatial cross-validation.
    - **Features:** 40 climate features (CRU-JRA), 4 land-use features (LUH2/gHM), and 3 terrain features.
- **Scenarios Modeling:** 
    - `Present`: 1991–2020 Climate & Land Use.
    - `Historical`: 1961–1990 Climate & Land Use.
    - `Sensitivity`: 3 factorial mixtures of Pre-industrial (PI) and Present-day drivers.

## 2. Model Performance (Spatial Validation)
The model exhibits high fidelity in capturing global LAI gradients.
- **Valid Training Pixels:** 4,598,543
- **Spatial R² (Present OOS):** 0.9473
- **Spatial RMSE:** 0.3769 LAI units
- **Spatial Bias:** < 0.0001

## 3. Temporal Change Validation (1980s to 2010s)
Validation of the predicted shift from 1982–1990 to 2011–2015 against GIMMS observations:
- **Change Correlation (Pearson):** 0.1865
- **Sign Agreement:** 57.65%
- **Zonal Correlation:** 0.4429

![Zonal Profile](../output_lai/plots/lai_zonal_profile.png)
*Figure 1: Latitudinal profile of observed vs. predicted LAI change. The model captures the magnitude of Arctic greening accurately.*

![Sign Agreement](../output_lai/plots/lai_sign_agreement_map.png)
*Figure 2: Global map of sign agreement between observed and predicted LAI changes.*

## 4. Sensitivity Results
Modelled changes reveal a dominant climate signal over land-use change for 95th percentile LAI at the global scale:
- **Climate Effect:** Captured substantial expansion of high-LAI zones in high latitudes and parts of the tropics.
- **Total Change:** Difference map `output_lai/lai_total_change.nc` provides the delta from Pre-industrial to Present.

![Total Change Map](../output_lai/plots/lai_total_change_map.png)
*Figure 3: Total predicted LAI change from Pre-industrial to Present-day.*

## 5. Key Assets
- **Primary Script:** `scripts/lai/train_predict_lai.py`
- **Validation Script:** `scripts/validate_changes.py`
- **Final Output:** `output_lai/lai_present_ensemble_mean.nc`
- **Sensitivity Map:** `output_lai/lai_sensitivity_climate.nc`
- **Sign Agreement Map:** `output_lai/lai_change_sign_agreement.nc`
