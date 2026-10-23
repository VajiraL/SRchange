#!/usr/bin/env bash
# Complete the v2 run: ensembles 2-3 (10 folds), then rebuild the 15-member cube,
# mask and per-member stats, and rerun the cube-dependent analyses.
# Ensemble-1 results are kept under *_ens1 (cube, mask, stats, outputs/v2_ens1, figures/v2_ens1).
# SHAP is not rerun: it uses a single reference model independent of the ensemble.
set -uo pipefail
cd /home/vajira/SRchange
PY=.venv/bin/python
step() { echo "=== [$(date '+%F %T')] $1"; }
step "1/5 train + predict ensembles 2-3"
SR_TASKS=2:1,2:2,2:3,2:4,2:5,3:1,3:2,3:3,3:4,3:5 $PY scripts/modeling/predict_sr_xgb_cpu.py || exit 1
n=$(find output_SR/scenario_predictions_v2 -name sr_k6.nc | wc -l)
step "prediction files: $n / 2520"; [ "$n" -eq 2520 ] || { echo "incomplete -- stopping"; exit 1; }
step "2/5 aggregate (15 members)";  AGG_WORKERS=32 $PY scripts/modeling/aggregate_predictions_parallel.py || exit 1
step "3/5 all-scenario mask";       $PY scripts/modeling/apply_all-scenario_mask.py || exit 1
step "4/5 per-member statistics";   STATS_WORKERS=32 $PY scripts/modeling/calculate_per_model_stats.py || exit 1
step "5/5 analyses (RootZoneChange, core)"
cd /home/vajira/RootZoneChange
bash scripts/run_v2_analyses.sh core 2>&1 | tee logs/v2_full_core.log
SR_VERSION=v2 .venv/bin/python -m scripts.v1_v2_compare > outputs/v2/v1_v2_compare.md 2>&1
step "FULL V2 PIPELINE DONE"
