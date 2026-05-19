# SRchange
Modelling and analysis of SR change

In this project we use ML for predicting SR (Root zone water storage capacity) change under past/future/counterfactual climate x LU scenarios. We decompose the climate and LU effects and study the underlying drivers of change, spatial patterns and trends.

## Scripts Overview

### Data Preparation
| Script | Description |
| :--- | :--- |
| `harmonize_lu_classes.py` | Harmonizes land-use datasets (HILDA+, LUH2) into a consistent classification. |
| `estimate_lu_scenario_features.py` | Generates synthetic land-use features (e.g., forest age, irrigation) for scenarios. |
| `compute_climate_clusters.py` | Defines modelling units by clustering grid cells based on climate variables. |
| `calc_randomcv_masks.py` | Generates spatial masks for cross-validation ensemble splits. |
| `train_lu_distributions.py` | Trains statistical distributions for land-use features to support scenario synthesis. |
| `estimate_hilda_forest_age.py` | Estimates forest stand age based on historical land-use transitions. |
| `process_anthromes.py` | Prepares anthromes (anthropogenic biomes) data as model predictors. |
| `process_population.py` | Processes and scales global population density datasets. |
| `generate_k_maps.py` | Creates spatial visualizations of climate-based cluster assignments. |
| `validate_synthetic_features.py` | Validates synthesized scenario features against reference data. |
| `prepare_lu_classes.py` | Initial processing and preparation of land-use class layers. |
| `prepare_forest_lgrip.py` | Processes LGRIP forest cover data for feature engineering. |
| `prepare_ghm.py` | Processes Global Hydrological Model outputs for inclusion in the modelling pipeline. |

### Modeling
| Script | Description |
| :--- | :--- |
| `predict_sr_xgb_cpu.py` | Executes SR predictions using XGBoost models for all climate and land-use scenarios. |
| `calculate_per_model_stats.py` | Computes area-weighted global statistics for each ensemble member with on-the-fly vegetation masking. |
| `explain_models.py` | Performs SHAP-based feature importance and attribution analysis for the clustered models. |
| `accuracy_assessment.py` | Calculates performance metrics (R², RMSE, KGE, etc.) for model validation. |
| `aggregate_predictions_parallel.py` | Aggregates ensemble predictions into scenario-level means and standard deviations. |
| `cluster_sensitivity_analysis.py` | Tests model response to perturbations in individual features at the cluster level. |
| `evaluate_k_sweep.py` | Analyzes the impact of the number of clusters (k) on model performance. |
| `sampling_strategy_assessment.py` | Evaluates the effect of different training data sampling methods on prediction accuracy. |
| `mask_sr_predictions.py` | Post-processing script to apply land-cover masks to aggregated spatial outputs. |
| `aggregate_sensitivity.py` | Aggregates sensitivity analysis results across multiple model runs. |
| `sensitivity_all_models.py` | Conducts comprehensive sensitivity tests across all clustered models in the ensemble. |

### Analysis & Uncertainty
| Script | Description |
| :--- | :--- |
| `analyze_change_uncertainty.py` | Decomposes SR changes into climate, LU, and interaction drivers with uncertainty propagation. |
| `calculate_scenario_uncertainty.py` | Aggregates ensemble results to determine scenario-level means and 95% confidence intervals. |
| `check_weighting_logic.py` | Verification utility to ensure area-weighting calculations are robust and consistent. |

### Plotting & Visualization
| Script | Description |
| :--- | :--- |
| `plot_sr_step_attribution.py` | Generates stepwise bar plots attributing global SR change to climate and LU components. |
| `plot_change_maps.py` | Produces global maps showing the spatial distribution of SR changes between periods. |
| `compare_feature_dependence.py` | Compares SHAP dependence plots across different models or environmental clusters. |
| `plot_sr_change_attribution.py` | Visualizes the relative contribution of different drivers to total SR change. |
| `plot_shap_dependence.py` | Generates detailed feature dependence plots for key model drivers. |
| `plot_sensitivity_analysis.py` | Visualizes how SR predictions vary across feature ranges in sensitivity tests. |
| `plot_k_sweep.py` | Visualizes the trade-off between model complexity (k clusters) and predictive accuracy. |
| `plot_diagnostics.py` | Generates standard diagnostic plots for model performance and residual analysis. |

### Optimization
| Script | Description |
| :--- | :--- |
| `optimize_xgb.py` | Automates hyperparameter tuning for XGBoost models using Optuna. |
| `baseline_benchmark.py` | Compares the performance of the clustered XGBoost approach against simple baseline models. |
