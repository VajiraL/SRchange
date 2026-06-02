# Planning Document: Nonlinear Ecohydrological Thresholds in $S_R$

## 1. Objective
Study thresholds of $S_R$ adaptation to climate and land-use (LU) stress. Identify identifiable hydroclimatic or LU thresholds beyond which $S_R$ changes abruptly, saturates, or declines.

## 2. Research Questions
- Are there identifiable hydroclimatic or land-use thresholds for $S_R$?
- Does $S_R$ respond nonlinearly (tipping-like responses) to stress?
- How do these thresholds vary across biomes, regions, or ecosystems?

## 3. Methodology

### 3.1 Data and Grouping Strategy

#### Existing Datasets
- **SHAP values:** Pre-computed SHAP values for each model cluster (`output_SR/explanation/data/shap_cluster_*.npz`).
- **Climate Clusters:** The $k=6$ map used for modeling, plus a sweep of $k=2$ to $25$ (`data/climate_clusters.zarr`, `data/k_sweep_clusters.zarr`). These provide a model-intrinsic grouping.
- **Anthromes:** Useful for identifying human modification thresholds (`data/anthromes/`).

#### Eco-Regional Grouping Candidates
- **WWF Ecoregions/Biomes (TEOW):** Shapefiles located in `data-raw/TEOW`.
    - *Note:* TEOW biomes represent static boundaries based on recent past climate/compilation data. They do not capture biome shifts over time (past/future) and may contain compilation artifacts.
- **Dynamic Biome Classifications:** To account for shifts, we could explore:
    - **Holdridge Life Zones:** Quantifiable zones based on PET, precipitation, and biotemperature (can be calculated from our climate data).
- **Köppen-Geiger Classification:** Standard climate-based classification. Historical (1901-2020) and future (SSPs) maps are available in `data-raw/BeckKG/`. 
    - *Note:* These are period-specific, allowing for analysis of threshold shifts across climate scenarios, making them more robust than static biomes for temporal studies.

### 3.2 Implementation Strategy
- **Isolated Workspace:** All new scripts will be placed in `scripts/thresholds_analysis/`. Existing scripts will remain untouched.
- **Output Management:** All intermediate data and figures will be saved in `output_SR/thresholds_analysis/` and `plots/thresholds_analysis/` respectively.
- **Documentation & Logging:**
    - **Analysis Log:** A running log file `notes/thresholds_analysis_log.md` will record every script execution, including inputs, outputs, and a brief description of the task.
    - **Scientific Notes:** Detailed scientific summaries for each major analysis phase (e.g., SHAP interactions, threshold detection results) will be created as separate files in `notes/thresholds_scientific_notes/`.

### 3.3 Analysis Steps
1. **SHAP Dependence Analysis:**
   - Generate SHAP dependence plots for key predictors.
   - Analyze interactions (e.g., Aridity vs. Forest Age).
2. **Threshold Detection:**
   - Apply segmented regression or GAMs to SHAP dependence curves.
   - Use derivative analysis or breakpoint detection to identify threshold values ($X^*$).
   - **Refinement Note (Future Work):** While piecewise linear models (Phase 2) effectively capture many transitions (especially for high $R^2 > 0.8$), some clusters exhibit more complex non-linearities (e.g., multi-modal or smooth transitions) that may be better captured by GAMs or higher-order segmented models. Future iterations should explore automated breakpoint selection and local regression techniques to handle low-$R^2$ regions.
3. **Threshold Proximity Mapping:**
   - Compute distance to thresholds for each grid cell ($D_i = X_i - X_i^*$).
   - Classify cells: below, approaching, beyond threshold.
4. **Typology Classification:**
   - Adaptive, Saturation, Suppressed, or Reversal zones.
5. **Scenario-Based Threshold Shifts (Phase 4):**
   - Apply static thresholds to past and future climate/LU data to quantify the expansion of "Saturation" regimes.
   - Study the velocity of threshold proximity change.

## 4. Expected Figures
- **Figure 1:** Conceptual response curves.
- **Figure 2:** SHAP dependence plots by cluster (Aridity, PET, Seasonality, Forest Age, gHM).
- **Figure 3:** Detected thresholds (values and response changes).
- **Figure 4:** Global map of threshold proximity.
- **Figure 5:** Tipping-risk typology map.

## 5. Implementation Plan
- [ ] Create a dedicated notebook/script for SHAP dependence extraction.
- [ ] Implement threshold detection algorithms (Segmented Regression, GAMs).
- [ ] Group results by biomes/clusters.
- [ ] Generate maps and typology classifications.
