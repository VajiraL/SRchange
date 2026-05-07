# Activity Log - SRchange Project

## Summary of Recent Activities (May 2026)

### 1. Core Modeling Pipeline
- **Ensemble Modeling:** Implemented a 3-ensemble, 5-fold spatial block cross-validation pipeline using XGBoost.
- **Clustering:** Developed a climate-cluster guided modeling approach ($k=6$) to capture regional environmental relationships.
- **Scenario Processing:** Automated prediction for 168 scenarios (Climate x Land Use) across historical and future periods.
- **Data Aggregation:** Consolidated all scenario outputs into a unified, optimized Zarr store (`output_SR/aggregated_sr_scenarios.zarr`).

### 2. Validation & Sensitivity Analysis
- **Accuracy Assessment:** Verified that the clustered model ($R^2 \approx 0.75$) significantly outperforms the global model ($R^2 \approx 0.71$).
- **Cluster Specialization:** Performed a cross-cluster sensitivity analysis, proving that regional models are specialized to their respective climate zones ("Diagonal Dominance").
- **k-Sweep Analysis:** Systematically evaluated the impact of cluster count ($k=1$ to $k=20$) on accuracy, justifying the choice of $k=6$ as the optimal parsimonious configuration.

### 3. Key Scripts Developed
| Script | Purpose |
| :--- | :--- |
| `scripts/modeling/train_prediction_ensemble.py` | Main training and global prediction engine. |
| `scripts/modeling/accuracy_assessment.py` | Calculates test-fold metrics and validates Zarr outputs. |
| `scripts/modeling/sensitivity_all_models.py` | Cross-predicts all models on all cells for sensitivity analysis. |
| `scripts/modeling/evaluate_k_sweep.py` | Systematically evaluates accuracy across different $k$ values. |
| `scripts/data-preparation/generate_k_maps.py` | Generates global cluster maps for $k=2..20$. |
| `scripts/plotting/plot_k_sweep.py` | Visualizes the accuracy vs. cluster count curve. |

### 4. Project Structure & Reproducibility
- **Standardized Inputs:** Transitioned to using synthetic/standardized land-use predictors for all periods to ensure change signals are not artifacts of data sources.
- **Environment Management:** Project is managed using `uv` with Python 3.12+.
- **HPC Optimization:** Leveraged NVIDIA L40S GPU for high-speed XGBoost training and global predictions.
