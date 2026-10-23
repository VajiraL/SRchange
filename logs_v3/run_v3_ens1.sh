#!/usr/bin/env bash
# v3 run (HYDE-driven land use; gHM, irrigated and rainfed fraction; no forest age),
# ensemble 1 only. Historical scenarios only (4 climates x 6 land-use states).
# Ensembles 2-3 later with SR_TASKS=2:1,...,3:5 and the same environment.
set -uo pipefail
cd /home/vajira/SRchange
PY=.venv/bin/python
export SR_RUN=v3 SR_LU_DIR=data/LU_features_v3 SR_LU_FEATURES=gHM,irr_frac,rain_frac
export SR_OUTPUT_BASE=output_SR/scenario_predictions_v3
export SR_SCENARIOS=1901_1930:intact,1901_1930:pre-industrial,1901_1930:1901_1930,1901_1930:1931_1960,1901_1930:1961_1990,1901_1930:1991_2020,1931_1960:intact,1931_1960:pre-industrial,1931_1960:1901_1930,1931_1960:1931_1960,1931_1960:1961_1990,1931_1960:1991_2020,1961_1990:intact,1961_1990:pre-industrial,1961_1990:1901_1930,1961_1990:1931_1960,1961_1990:1961_1990,1961_1990:1991_2020,1991_2020:intact,1991_2020:pre-industrial,1991_2020:1901_1930,1991_2020:1931_1960,1991_2020:1961_1990,1991_2020:1991_2020
step() { echo "=== [$(date '+%F %T')] $1"; }
step "1/5 train + predict ensemble 1 (24 historical scenarios, 47 predictors)"
SR_TASKS=1:1,1:2,1:3,1:4,1:5 $PY scripts/modeling/predict_sr_xgb_cpu.py || exit 1
n=$(find $SR_OUTPUT_BASE -name sr_k6.nc | wc -l); step "prediction files: $n / 120"; [ "$n" -eq 120 ] || exit 1
step "2/5 aggregate";             AGG_WORKERS=24 $PY scripts/modeling/aggregate_predictions_parallel.py || exit 1
step "3/5 all-scenario mask";     $PY scripts/modeling/apply_all-scenario_mask.py || exit 1
step "4/5 per-member statistics"; STATS_WORKERS=24 $PY scripts/modeling/calculate_per_model_stats.py || exit 1
step "5/5 SHAP (64 cores)"
for b in 1901_1930 1961_1990; do
  SHAP_CORES=64 SHAP_OUTPUT_BASE=output_SR/explain_change_v3 SHAP_BASE_CLIM=$b SHAP_BASE_LU=$b $PY scripts/modeling/explain_change_full_global.py || exit 1
done
step "V3 ENS1 MODEL CHAIN DONE"
