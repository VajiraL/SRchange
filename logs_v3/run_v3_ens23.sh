#!/usr/bin/env bash
# v3 run, ensembles 2-3 (10 folds), then the 15-member cube, mask, stats and analyses.
# Ensemble-1 results are kept under *_ens1 (cube, mask, stats, outputs/v3_ens1, figures/v3_ens1).
set -uo pipefail
cd /home/vajira/SRchange
PY=.venv/bin/python
export SR_RUN=v3 SR_LU_DIR=data/LU_features_v3 SR_LU_FEATURES=gHM,irr_frac,rain_frac
export SR_OUTPUT_BASE=output_SR/scenario_predictions_v3
export SR_SCENARIOS=1901_1930:intact,1901_1930:pre-industrial,1901_1930:1901_1930,1901_1930:1931_1960,1901_1930:1961_1990,1901_1930:1991_2020,1931_1960:intact,1931_1960:pre-industrial,1931_1960:1901_1930,1931_1960:1931_1960,1931_1960:1961_1990,1931_1960:1991_2020,1961_1990:intact,1961_1990:pre-industrial,1961_1990:1901_1930,1961_1990:1931_1960,1961_1990:1961_1990,1961_1990:1991_2020,1991_2020:intact,1991_2020:pre-industrial,1991_2020:1901_1930,1991_2020:1931_1960,1991_2020:1961_1990,1991_2020:1991_2020
step() { echo "=== [$(date '+%F %T')] $1"; }
step "1/5 train + predict ensembles 2-3"
SR_TASKS=2:1,2:2,2:3,2:4,2:5,3:1,3:2,3:3,3:4,3:5 $PY scripts/modeling/predict_sr_xgb_cpu.py || exit 1
n=$(find $SR_OUTPUT_BASE -name sr_k6.nc | wc -l); step "prediction files: $n / 360"; [ "$n" -eq 360 ] || exit 1
step "2/5 aggregate (15 members)"; AGG_WORKERS=24 $PY scripts/modeling/aggregate_predictions_parallel.py || exit 1
step "3/5 all-scenario mask";      $PY scripts/modeling/apply_all-scenario_mask.py || exit 1
step "4/5 per-member statistics";  STATS_WORKERS=24 $PY scripts/modeling/calculate_per_model_stats.py || exit 1
step "5/5 analyses (RootZoneChange, core)"
cd /home/vajira/RootZoneChange
SR_VERSION=v3 bash scripts/run_v2_analyses.sh core 2>&1 | tee logs/v3_full_core.log
SR_VERSION=v3 .venv/bin/python -m scripts.v1_v2_compare > outputs/v3/v1_v3_compare.md 2>&1
step "V3 FULL PIPELINE DONE"
