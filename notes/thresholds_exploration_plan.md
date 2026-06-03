# Planning Document: Nonlinear Ecohydrological Thresholds in $S_R$ (Version 3)

## 1. Objective
Model and quantify ecohydrological thresholds of $S_R$ adaptation to climate and land-use stress using global cluster-based XGBoost models and SHAP analysis.

## 2. Implementation Status

### Phase 1-6: Detection and Modeling (Completed)
- [x] SHAP dependence mapping to geographic/KG coordinates.
- [x] GAM-based threshold taxonomy (Hard, Soft, Collapse).
- [x] Spatial Block Bootstrap (N=100) for uncertainty quantification.
- [x] Multivariate Interaction Surfaces (Tensor-product GAMs).
- [x] Bootstrap integration for 2D interaction surfaces.

### Phase 8: Dynamic Scenario & Probabilistic Mapping (Completed)
- [x] Probabilistic mapping of "Saturated" regimes ($P(X > X^*)$) across historical and future scenarios.
- [x] Inclusion of conditional thresholds (e.g., Aridity thresholds shifted by gHM).

### Phase 7: Ecological Validation (Completed)
- [x] Ingest LHSIF (Long-Term Harmonized SIF) at 0.05° resolution.
- [x] Calculate SIF stability metrics (CV) as proxies for ecosystem stress.
- [x] Validate model-inferred \$P_{sat}\$ against observed SIF variability.
- [ ] Cross-validate with VODCA (VOD) once available.

## 3. Paper 2 Strategy: The Adaptation Gap (Separate Section)
*Note: This will be implemented in a separate phase after the core threshold paper is finalized.*

### Phase 9: Adaptation Gap Framework
- **$S_{R,demand}$:** Calculate cumulative climatic water deficit (CWD) from climate data (P - PET).
- **$S_{R,capacity}$:** Define maximum expressible $S_R$ using multivariate frontiers and saturation surfaces.
- **Gap Quantification:** $Gap = S_{R,demand} - S_{R,capacity}$.

## 4. Key Attributes for Validation Datasets (Phase 7)
For Phase 7, we require datasets that serve as proxies for ecosystem stress or function.

| Attribute | Vegetation Optical Depth (VOD) | Gross Primary Productivity (GPP) | Solar-Induced Fluorescence (SIF) |
| :--- | :--- | :--- | :--- |
| **Proxy for** | Water content / Biomass | Photosynthetic activity | Actual electron transport |
| **Spatial Res** | 0.05° - 0.25° | 0.05° | 0.05° - 0.5° |
| **Temporal Res** | Daily/Monthly (Long-term) | Monthly | Daily/Monthly |
| **Method** | Passive Microwave (VODCA) | Fluxcom / MODIS / VPM | TROPOSIF / GOME-2 |
| **Threshold Logic** | Lower resilience (slower recovery) beyond $X^*$ | Higher sensitivity to VPD/SM beyond $X^*$ | Abrupt drop in SIF/GPP ratio beyond $X^*$ |

**Ideal Dataset Selection:**
1. **VODCA (VOD):** Excellent for long-term trends and resilience analysis.
2. **FLUXCOM (GPP):** Good for mean state, but may be too "smooth" as it's upscaled.
3. **MODIS GPP (MOD17):** Better for capturing anomalies and high-resolution stress.
4. **TROPOSIF (SIF):** High sensitivity to instantaneous drought stress.
