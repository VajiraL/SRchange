# SRchange
Modelling and analysis of $S_R$ change

Root zone storage capacity ($S_R$) is the maximum limit of vegetation-accessible water storage in soil. This capacity supports survival of vegetation during dry-spells. Vegetation (ecosystems) adapt the root zone (and $S_R$) responding the changing hydroclimatic conditions. $S_R$ is also modified by land use change.

In this project we use ML for predicting $S_R$ (Root zone water storage capacity) change under past/future/counterfactual climate x LU scenarios. We decompose the climate and LU effects and study the underlying drivers of change, spatial patterns and trends.

## Scripts Overview

### Data Preparation
| Script | Description |
| :--- | :--- |
| `prepare_forest_lgrip.py` | Align LGRIP data|
| `prepare_ghm.py` | Align GHM (global human modification index) data|
| `process_anthromes.py` | Align anthrome data to common grid|
| `process_population.py` | Align population data to common grid. |
| `prepare_lu_classes.py` | Align past anf future land-use class to common grid and create a unified zarr store|
| `harmonize_lu_classes.py` | Harmonizes the unified past anf future land-use dataset from `prepare_lu_classes.py` into a consistent classification. |
| `estimate_hilda_forest_age.py` | Estimates forest stand age based on historical land-use transitions. (Deprecated)|
| `compute_climate_clusters.py` | Defines modelling units by clustering grid cells based on climate variables. Produces equivalent cluster maps for the different climates|
| `train_lu_distributions.py` | Trains statistical distributions within present period combined LU class x climate cluster.|
| `estimate_lu_scenario_features.py` | Generates synthetic land-use features for scenarios by sampling from fitted distributions from `train_lu_distributions.py`.|
| `validate_synthetic_features.py` | Validates synthesized scenario features against reference data. |
| `calc_randomcv_masks.py` | Generates spatial masks for cross-validation ensemble splits. |
| `generate_k_maps.py` | Generate a cluster maps for a range of k for studying sensitivity to k (k_sweep analysis)|
| `merge-target-sr` | Create the training target by merging $S_{R-Stocker}$ with $S_{R-WE}$ |
### Modeling
| Script | Description |
| :--- | :--- |
| `predict_sr_xgb_cpu.py` | XGBoost modelling with CPU parellelization for SR prediction for all scenarios.|
| `aggregate_predictions_parallel.py` | Aggregates ensemble predictions into scenario-level means and standard deviations. |
| `accuracy_assessment.py` | Calculates performance metrics (R², RMSE, KGE, etc.) for model accuracy.|
| `apply_all-scenario_mask.py` | Apply an all-scenario super-mask (mask if a cell in all predictions if it's missing in any of the original 168 scenarios)|
| `calculate_per_model_stats.py` | Computes area-weighted global statistics for each ensemble member while applying the super-mask|
| `cluster_sensitivity_analysis.py` | Testing sensitivity of predictions of climate cluster shift between training and prediction|
| `evaluate_k_sweep.py` | Analyzing the sensitivity to the number of clusters (k) |
| `sampling_strategy_assessment.py` | Evaluates the effect of different training data sampling methods on prediction. |
| `sensitivity_all_models.py` | Conducts comprehensive sensitivity tests across all clustered models in the ensemble. |
| `aggregate_sensitivity.py` | Aggregates sensitivity analysis outputs from `sensitivity_all_models.py`|
| `explain_models.py` | SHAP analysis and feature importance |

### Analysis
| Script | Description |
| :--- | :--- |
| `analyze_change_uncertainty.py` | Decomposes SR changes into climate, LU, and interaction drivers with uncertainty propagation. |
| `calculate_scenario_uncertainty.py` | Aggregates ensemble results to determine scenario-level means and 95% confidence intervals. |
| `check_weighting_logic.py` | Verification utility to ensure area-weighting calculations are robust and consistent.(Deprecate) |

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