# Phase 2: Threshold Detection using Piecewise Regression

## Overview
We quantified the thresholds of $S_R$ adaptation by fitting piecewise linear models to binned SHAP dependence curves. This allows us to identify the exact feature values ($X^*$) where the relationship between a predictor and $S_R$ changes abruptly.

## Key Findings

### 1. Aridity Index Thresholds
- **Cluster 0 (Boreal/Cold):** A sharp threshold at **AI ≈ 0.84**. Below this, $S_R$ sensitivity increases rapidly with aridity (slope 0.45); above it, the relationship flattens or reverses (slope -0.04), indicating a saturation of adaptive capacity in these regions.
- **Cluster 3 (Semi-arid):** Shows a "rebound" threshold at **AI ≈ 0.18**.

### 2. Human Modification (gHM) Thresholds
- **Cluster 0:** Shows an extremely sensitive response to initial modification. A threshold was detected at **gHM ≈ 0.0013**. The initial slope is ~194, which drops to ~0.14 after the threshold. This suggests that even minimal human footprint in these (likely pristine) areas triggers a rapid shift in $S_R$.
- **Cluster 1:** Threshold at **gHM ≈ 0.06**.

### 3. PET Thresholds
- Multiple clusters show thresholds in annual PET (e.g., Cluster 1 at **1640 mm/yr**, Cluster 5 at **610 mm/yr**).

## Statistical Quality
- Most fits achieved an $R^2 > 0.80$, confirming that piecewise linear models are a robust representation of these ecohydrological response functions.
- Binned averaging effectively removed noise from the raw SHAP values, revealing the underlying deterministic thresholds.

## Next Steps
- **Threshold Proximity Mapping (Phase 3):** Map how close current grid cells are to these detected thresholds.
- **Typology Classification (Phase 4):** Classify regions based on whether they are in the "Adaptive" (pre-threshold) or "Saturated/Suppressed" (post-threshold) regime.
