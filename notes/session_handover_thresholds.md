# Session Handover: SR Thresholds Analysis

## 1. Context Summary
We are starting a new research branch `sr-thresholds` to study nonlinear ecohydrological thresholds in Root Zone Water Storage Capacity ($S_R$). This follows a major modeling effort using XGBoost where $S_R$ was predicted globally using climate and land-use (LU) variables.

The goal is to move from attribution into ecological response theory: identifying tipping-like behaviors where $S_R$ adaptation saturates or collapses under stress.

## 2. Current State
- **Branch:** `sr-thresholds` (active).
- **Plan:** Finalized in `notes/thresholds_exploration_plan.md`.
- **Infrastructure:** Directories created:
  - `scripts/thresholds_analysis/`
  - `output_SR/thresholds_analysis/`
  - `plots/thresholds_analysis/`
  - `notes/thresholds_scientific_notes/`
- **Logging:** Initialized in `notes/thresholds_analysis_log.md`.

## 3. Key Data Assets
- **SHAP Data:** `output_SR/explanation/data/shap_cluster_*.npz` (Pre-computed).
- **Climate Clusters:** `data/climate_clusters.zarr` (K=6 map used in modelling).
- **Koppen-Geiger:** `data-raw/BeckKG/` (Period-specific maps for 1901-2099).
- **Ecoregions:** `data-raw/TEOW` (WWF shapefiles).

## 4. Next Session Start (Phase 1)
The immediate next step is to begin **Phase 1: SHAP Dependence Analysis**.

### Tasks for the next agent:
1.  **Read the SHAP data:** Load the `.npz` files for each cluster.
2.  **Extract Interactions:** Focus on key predictors:
    - Aridity vs. Forest Age
    - gHM (Human Modification) vs. Aridity
    - PET/Precipitation Seasonality interactions.
3.  **Group by KG Classes:** Use the 1991-2020 Koppen-Geiger map to group the SHAP dependence curves for more "objective" scientific interpretation.
4.  **Visualize:** Generate SHAP dependence plots (SHAP value vs. Predictor, colored by interacting variable).
5.  **Log Work:** Update `notes/thresholds_analysis_log.md` after every script run.

## 5. Implementation Guardrails
- **No changes to existing scripts.** All new code must be in `scripts/thresholds_analysis/`.
- **Absolute paths only.**
- **Scientific focus:** Look for abrupt changes in slope, saturation points, or sign reversals in the SHAP values.
