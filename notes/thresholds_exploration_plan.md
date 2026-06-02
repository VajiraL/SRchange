# Planning Document: Nonlinear Ecohydrological Thresholds in $S_R$ (Version 2)

## 1. Objective
Study thresholds of $S_R$ adaptation to climate and land-use (LU) stress. Identify identifiable hydroclimatic or LU thresholds beyond which $S_R$ changes abruptly, saturates, or declines. Transition from simple curve fitting to a robust, manuscript-quality threshold analysis.

## 2. Research Questions
- Are there identifiable hydroclimatic or land-use thresholds for $S_R$?
- Does $S_R$ respond nonlinearly (tipping-like responses) to stress?
- How do these thresholds vary across biomes, regions, or ecosystems?
- How do climate and land-use interactions shift these thresholds?

## 3. Implementation Strategy (Version 2)

### Phase 5: Advanced Threshold Detection (High Priority)
- **GAM-based Smooth Thresholds:** Implement Generalized Additive Models (GAMs) to identify "soft" thresholds. Use the first derivative of the $SHAP(S_R)$ vs. Predictor curve to detect saturation points (e.g., where slope falls below 10% of max).
- **Threshold Taxonomy:** Classify every cluster-feature relationship into: **Hard** (abrupt breakpoint), **Soft** (smooth saturation), **Collapse** (derivative sign reversal), or **No Threshold**.
- **Variable Selection:** Perform advanced detection across all top predictors identified in the modeling phase. Final selection of variables for multivariate surfaces (Phase 6) will be based on the robustness and ecological relevance of detected thresholds.
- **Uncertainty Quantification:** Implement a spatial block bootstrap (n=100+) to estimate the probability distribution of $X^*$.

### Phase 6: Multivariate Threshold Surfaces
- **Climate-LU Interactions:** Use tensor-product GAMs to model $SHAP(S_R) = f(Aridity, gHM)$.
- **Threshold Contours:** Extract the threshold for Aridity as a function of Human Modification. Identify if human pressure lowers the threshold of hydroclimatic adaptation.

### Phase 7: Independent Ecological Validation
- **VOD & SIF Sensitivity:** Test if grid cells in "Saturated" regimes (Phase 3) show higher sensitivity of Productivity (GPP/SIF) or Vegetation Water Content (VOD) to drought (SPEI).
- **Mortality Risk:** Cross-reference saturated zones with global tree mortality databases.

### Phase 8: Dynamic Scenario Analysis
- **Dynamic Threshold Migration:** Instead of static thresholds, use the multivariate surfaces (Phase 6) to model how thresholds shift under future land-use and climate scenarios.

---

## Appendix: Version 1 Plan (Completed/Initial)

### 3.1 Data and Grouping Strategy (V1)
- **SHAP values:** Pre-computed SHAP values for each model cluster.
- **Climate Clusters:** K=6 map used for modeling.
- **Köppen-Geiger Classification:** Standard climate-based classification for grouping.

### 3.3 Analysis Steps (V1)
1. **SHAP Dependence Analysis:** Generate SHAP dependence plots for key predictors. [COMPLETED]
2. **Threshold Detection:** Apply piecewise linear regression to identify breakpoints ($X^*$). [COMPLETED for R2 > 0.8]
3. **Threshold Proximity Mapping:** Compute distance to thresholds and classify "Adaptive" vs "Saturated" regimes. [COMPLETED for 1991-2020]
4. **Scenario-Based Threshold Shifts:** Apply static thresholds to 1901 and 2099 scenarios. [COMPLETED]

---

## 4. Expected Figures (Refined)
- **Figure 1:** Conceptual response curves (Hard, Soft, Collapse, Interaction).
- **Figure 2:** Threshold Surface Map: Aridity threshold as a function of gHM/Forest Age.
- **Figure 3:** Global Map of Threshold Proximity (Probabilistic, from bootstrap).
- **Figure 4:** Validation Figure: Drought sensitivity of VOD/SIF inside vs. outside saturated regimes.

## 5. Implementation Status
- [x] SHAP dependence extraction with KG mapping.
- [x] Piecewise linear threshold detection.
- [x] Global proximity/regime mapping.
- [x] Initial scenario analysis.
- [ ] Implement GAM-based derivative detection (Phase 5).
- [ ] Implement Spatial Block Bootstrap (Phase 5).
- [ ] Create Tensor-product interaction surfaces (Phase 6).
- [ ] Ecological validation with external datasets (Phase 7).
