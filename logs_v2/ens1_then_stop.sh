#!/usr/bin/env bash
# Stop the v2 production run once ensemble 1 (folds 1-5) has finished, discard
# partial ensemble 2/3 output, then build the ensemble-1 cube and run analyses
# with reduced CPU use (shared server). Resume ensembles 2-3 later with:
#   SR_TASKS=2:1,2:2,2:3,2:4,2:5,3:1,3:2,3:3,3:4,3:5 .venv/bin/python scripts/modeling/predict_sr_xgb_cpu.py
set -uo pipefail
cd /home/vajira/SRchange
LOG=logs_v2/chain.log
PRED=output_SR/scenario_predictions_v2
step() { echo "=== [$(date '+%F %T')] $1"; }

step "waiting for Ens1 folds 1-5 to finish"
until [ "$(grep -cE '\[Ens1_Fold[1-5]\] Finished\.' $LOG)" -ge 5 ]; do sleep 30; done

step "stopping prediction run (own process tree only)"
ROOT=$(pgrep -u "$USER" -f '^.venv/bin/python scripts/modeling/predict_sr_xgb_cpu.py' | head -1)
if [ -n "$ROOT" ]; then
  # collect all descendants of the predict process (joblib/loky workers)
  PIDS="$ROOT"; FRONT="$ROOT"
  while [ -n "$FRONT" ]; do
    NEXT=$(for p in $FRONT; do pgrep -P "$p"; done | tr '\n' ' ')
    PIDS="$PIDS $NEXT"; FRONT="$NEXT"
  done
  echo "killing: $PIDS"
  kill $PIDS 2>/dev/null; sleep 10; kill -9 $PIDS 2>/dev/null
fi
step "removing partial ensemble 2/3 output"
find $PRED -mindepth 2 -maxdepth 2 -type d \( -name ensemble_2 -o -name ensemble_3 \) -exec rm -rf {} +
n1=$(find $PRED -path '*ensemble_1*' -name sr_k6.nc | wc -l)
step "ensemble 1 files: $n1 / 840"
[ "$n1" -eq 840 ] || { echo "ensemble 1 incomplete -- stopping"; exit 1; }

PY=.venv/bin/python
step "aggregate (ensemble 1 only)";   AGG_WORKERS=32 $PY scripts/modeling/aggregate_predictions_parallel.py || exit 1
step "all-scenario mask";             $PY scripts/modeling/apply_all-scenario_mask.py || exit 1
step "per-member statistics";         STATS_WORKERS=32 $PY scripts/modeling/calculate_per_model_stats.py || exit 1
step "SHAP 1901-1930 (64 cores)";     SHAP_CORES=64 SHAP_BASE_CLIM=1901_1930 SHAP_BASE_LU=1901_1930 $PY scripts/modeling/explain_change_full_global.py || exit 1
step "SHAP 1961-1990 (64 cores)";     SHAP_CORES=64 SHAP_BASE_CLIM=1961_1990 SHAP_BASE_LU=1961_1990 $PY scripts/modeling/explain_change_full_global.py || exit 1
step "analyses (RootZoneChange)"
cd /home/vajira/RootZoneChange
bash scripts/run_v2_analyses.sh all 2>&1 | tee logs/v2_ens1_analyses.log
SR_VERSION=v2 .venv/bin/python -m scripts.v1_v2_compare > outputs/v2/v1_v2_compare.md 2>&1
step "ENS1 PIPELINE DONE"
