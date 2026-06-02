# Thresholds Analysis Log

This file tracks the execution of scripts and major milestones for the nonlinear $S_R$ thresholds analysis.

| Date | Script | Inputs | Outputs | Description |
|------|--------|--------|---------|-------------|
| 2026-06-02 | N/A | N/A | N/A | Project structure initialized and planning document finalized. |
| 2026-06-02 | scripts/thresholds_analysis/extract_shap_dependence_with_kg.py | output_SR/explanation/data/*.npz, data-raw/BeckKG/1991_2020/ | plots/thresholds_analysis/shap_dependence/, output_SR/thresholds_analysis/extracted_shap_interactions.parquet | Extracted SHAP dependence for key interactions (Aridity vs Forest Age, gHM vs Aridity, PET vs Precip Seasonality). Grouped samples by Koppen-Geiger classes using spatial matching. |
| 2026-06-02 | scripts/thresholds_analysis/detect_thresholds.py | output_SR/thresholds_analysis/extracted_shap_interactions.parquet | output_SR/thresholds_analysis/detected_thresholds.csv, plots/thresholds_analysis/thresholds/*.png | Applied piecewise linear regression to binned SHAP dependence curves for each cluster. Identified 18 significant thresholds for Aridity Index, gHM, and PET. |
| 2026-06-02 | scripts/thresholds_analysis/map_threshold_proximity.py | data/climate_features.zarr, data/climate_clusters.zarr, data/LU_features/1991_2020.nc, output_SR/thresholds_analysis/detected_thresholds.csv | output_SR/thresholds_analysis/threshold_mapping_*.nc, plots/thresholds_analysis/maps/*.png | Mapped global proximity to detected adaptation thresholds and classified grid cells into "Adaptive" and "Saturated/Post-Threshold" regimes for Aridity Index and gHM. |
| 2026-06-02 | scripts/thresholds_analysis/analyze_scenario_threshold_shifts.py | data/climate_features.zarr, output_SR/thresholds_analysis/detected_thresholds.csv | output_SR/thresholds_analysis/scenarios/*.nc, plots/thresholds_analysis/scenarios/*.png | Applied high-confidence (R2 >= 0.8) thresholds to past (1901-1930) and future (SSP126, SSP370) scenarios. Quantified the global expansion/contraction of the "Saturated" adaptation regime. |
