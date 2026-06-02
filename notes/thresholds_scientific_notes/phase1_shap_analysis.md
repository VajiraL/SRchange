# Phase 1: SHAP Dependence and Interaction Analysis

## Overview
In this phase, we analyzed the SHAP dependence of $S_R$ on key climate and land-use predictors. We focused on identifying nonlinearities and interactions that suggest ecohydrological thresholds.

## Key Interactions Analyzed
1.  **Aridity Index vs. Forest Age:** Exploring how the sensitivity of $S_R$ to aridity changes as forests mature.
2.  **gHM (Human Modification) vs. Aridity:** Investigating if human impact on $S_R$ is amplified in arid vs. humid regions.
3.  **PET vs. Precipitation Seasonality:** Looking for thresholds in evaporative demand sensitivity under different rainfall regimes.

## Methodology
- **Data Source:** Pre-computed SHAP values from XGBoost cluster models.
- **Spatial Alignment:** Sampled points were mapped back to their geographic locations (lat/lon) and assigned a Koppen-Geiger (KG) class from the 1991-2020 Beck et al. (2023) map.
- **Grouping:** Dependence plots were generated for each cluster and colored by KG class to provide ecological context.

## Initial Observations
- **Non-linearities:** Several clusters show clear "elbows" or saturation points in the SHAP dependence curves, particularly for Aridity Index and gHM.
- **KG Clustering:** Different KG classes within the same model cluster often exhibit distinct dependence profiles, suggesting that further sub-grouping by KG is necessary for precise threshold detection.
- **Forest Age Interaction:** There is a visible interaction where the slope of SHAP(Aridity) changes depending on whether the forest is young or old, indicating a possible adaptation threshold.

## Next Steps
- Apply segmented regression to the extracted curves to quantify the "elbow" points.
- Map the global distribution of these detected thresholds.
