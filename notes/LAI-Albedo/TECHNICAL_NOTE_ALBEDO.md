# Technical Note: Modeling Global Albedo Sensitivities

## 1. Methodology
- **Target Variable:** Annual 10th percentile Surface Albedo derived from GLASS (1981–2020) at 0.05° resolution. This captures the base state of the canopy reflectance, minimizing ephemeral snow signals.
- **Modeling Framework:** Independent Climate-Clustered XGBoost.
    - **Clustering:** 6 global climate clusters.
    - **Ensemble:** 5-fold spatial cross-validation.
    - **Features:** 40 climate features, 4 land-use features, and 3 terrain features.
- **Scenarios Modeling:** 
    - `Present`: 1991–2020 Climate & Land Use.
    - `Historical`: 1961–1990 Climate & Land Use.
    - `Sensitivity`: Factorial mixtures (PI/Present) to isolate climate and land-use effects.

## 2. Model Performance (Spatial Validation)
The model captures global albedo variations with extremely high accuracy.
- **Valid Training Pixels:** 5,658,632
- **Spatial R² (Present OOS):** 0.9632
- **Spatial RMSE:** 0.0139
- **Spatial Bias:** < 0.0001

## 3. Temporal Change Validation (1980s to 2010s)
Validation of the predicted shift from 1981–1990 to 2011–2020 against GLASS observations:
- **Change Correlation (Pearson):** 0.2531
- **Sign Agreement:** 60.52%
- **Zonal Correlation:** 0.6682

![Zonal Profile](../output_albedo/plots/albedo_zonal_profile.png)
*Figure 1: Latitudinal profile of observed vs. predicted Albedo change.*

![Sign Agreement](../output_albedo/plots/albedo_sign_agreement_map.png)
*Figure 2: Global map of sign agreement for Albedo temporal changes.*

## 4. Sensitivity Results
Albedo sensitivity analysis:
- **Climate Change Effect:** Significant darkening predicted in high latitudes and arid regions transitioning to semi-arid states.
- **Land Use Effect:** Captures local albedo shifts associated with deforestation and agricultural expansion.
- **Total Change:** Difference map `output_albedo/albedo_total_change.nc`.

![Total Change Map](../output_albedo/plots/albedo_total_change_map.png)
*Figure 3: Total predicted Albedo change from Pre-industrial to Present-day.*

## 5. Key Assets
- **Primary Script:** `scripts/albedo/train_predict_albedo.py`
- **Validation Script:** `scripts/validate_changes.py`
- **Final Output:** `output_albedo/albedo_present_ensemble_mean.nc`
- **Sensitivity Map:** `output_albedo/albedo_sensitivity_climate.nc`
- **Sign Agreement Map:** `output_albedo/albedo_change_sign_agreement.nc`
