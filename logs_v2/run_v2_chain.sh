#!/usr/bin/env bash
# v2 production chain (Oct 2026): corrected LU features -> retrain -> cube -> stats -> SHAP
set -euo pipefail
cd /home/vajira/SRchange
PY=.venv/bin/python
step() { echo "=== [$(date '+%F %T')] $1"; }
step "1/6 train + predict (3 ensembles x 5 folds, 168 scenarios)"
$PY scripts/modeling/predict_sr_xgb_cpu.py
step "2/6 aggregate -> output_SR/aggregated_sr_scenarios_v2.zarr"
$PY scripts/modeling/aggregate_predictions_parallel.py
step "3/6 all-scenario mask -> output_SR/aggregated_sr_scenarios_masked_v2.zarr"
$PY scripts/modeling/apply_all-scenario_mask.py
step "4/6 per-member statistics"
$PY scripts/modeling/calculate_per_model_stats.py
step "5/6 SHAP, 1901-1930 baseline"
SHAP_BASE_CLIM=1901_1930 SHAP_BASE_LU=1901_1930 $PY scripts/modeling/explain_change_full_global.py
step "6/6 SHAP, 1961-1990 baseline"
SHAP_BASE_CLIM=1961_1990 SHAP_BASE_LU=1961_1990 $PY scripts/modeling/explain_change_full_global.py
step "CHAIN COMPLETE"
