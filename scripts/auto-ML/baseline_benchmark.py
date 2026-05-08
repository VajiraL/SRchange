import os
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import json

# ---------------------------
# Configuration
# ---------------------------
ROOT = "/home/vajira/SRchange"
SUB_SAMPLE_SIZE = 200000 # Using a smaller subset for faster iterations
N_FOLDS = 5
K = 6

XGB_PARAMS_BASELINE = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'tree_method': 'hist',
    'device': 'cuda',
}

STATIC_FEATURES = ['elev', 'slope', 'roughness']
CLIMATE_FEATURES = [
    'annual_precip', 'aridity_index', 'frost_months', 'gdm_0', 'gdm_10', 'gdm_5', 
    'longest_dry_spell_months', 'max_monthly_precip', 'max_monthly_tmean', 
    'mean_annual_tmp', 'min_monthly_precip', 'min_monthly_tmean', 
    'monthly_precip_std', 'monthly_tmean_range', 'monthly_tmean_std', 
    'months_gt_25C', 'months_gt_30C', 'pet_annual', 'precip_coldest_quarter', 
    'precip_cv', 'precip_deficit_annual', 'precip_iqr', 'precip_p10', 'precip_p25', 
    'precip_p5', 'precip_p50', 'precip_p75', 'precip_p90', 'precip_p95', 
    'precip_warmest_quarter', 'temp_driest_quarter', 'temp_wettest_quarter', 
    'tmean_iqr', 'tmean_p10', 'tmean_p25', 'tmean_p5', 'tmean_p50', 'tmean_p75', 
    'tmean_p90', 'tmean_p95'
]
LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + LU_FEATURES

# Paths
TARGET_FILE = f"{ROOT}/data/SR-target/target-mergedSR_masked.nc"
CLIMATE_ZARR = f"{ROOT}/data/climate_features.zarr"
TERRAIN_FILE = f"{ROOT}/data/terrain_0p05.nc"
LU_FILE = f"{ROOT}/data/scenario_features/standard_1991_2020_none.nc"
CLUSTER_ZARR = f"{ROOT}/data/climate_clusters.zarr"
OUT_DIR = f"{ROOT}/output_SR/auto-ML"

def calculate_pbias(y_true, y_pred):
    return 100 * (np.sum(y_pred - y_true) / np.sum(y_true))

def run_experiment(name, params, df, use_clusters=True):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    test_metrics = []
    train_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
        fold_preds_test = np.full(len(df_test), np.nan)
        fold_preds_train = np.full(len(df_train), np.nan)
        
        if use_clusters:
            for cid in range(K):
                sub_tr = df_train[df_train['cluster'] == cid]
                sub_te = df_test[df_test['cluster'] == cid]
                
                if len(sub_tr) < 100:
                    continue
                
                dtr = xgb.DMatrix(sub_tr[ALL_FEATURES], label=sub_tr['sr'])
                model = xgb.train(params, dtr, num_boost_round=1000)
                
                if not sub_te.empty:
                    fold_preds_test[df_test['cluster'] == cid] = model.predict(xgb.DMatrix(sub_te[ALL_FEATURES]))
                fold_preds_train[df_train['cluster'] == cid] = model.predict(xgb.DMatrix(sub_tr[ALL_FEATURES]))
        else:
            dtr = xgb.DMatrix(df_train[ALL_FEATURES], label=df_train['sr'])
            model = xgb.train(params, dtr, num_boost_round=1000)
            fold_preds_test = model.predict(xgb.DMatrix(df_test[ALL_FEATURES]))
            fold_preds_train = model.predict(xgb.DMatrix(df_train[ALL_FEATURES]))
            
        # Metrics for fold
        valid_te = np.isfinite(fold_preds_test)
        y_true_te = df_test['sr'].values[valid_te]
        y_pred_te = fold_preds_test[valid_te]
        
        valid_tr = np.isfinite(fold_preds_train)
        y_true_tr = df_train['sr'].values[valid_tr]
        y_pred_tr = fold_preds_train[valid_tr]
        
        test_metrics.append({
            'r2': r2_score(y_true_te, y_pred_te),
            'mae': mean_absolute_error(y_true_te, y_pred_te),
            'pbias': calculate_pbias(y_true_te, y_pred_te)
        })
        train_metrics.append({
            'r2': r2_score(y_true_tr, y_pred_tr)
        })

    avg_test = pd.DataFrame(test_metrics).mean().to_dict()
    avg_train_r2 = pd.DataFrame(train_metrics)['r2'].mean()
    
    result = {
        'name': name,
        'test_r2': avg_test['r2'],
        'test_mae': avg_test['mae'],
        'test_pbias': avg_test['pbias'],
        'train_r2': avg_train_r2,
        'overtraining': avg_train_r2 - avg_test['r2']
    }
    return result

def main():
    print("Loading data for auto-ML...")
    sr_target = xr.open_dataarray(TARGET_FILE).where(lambda x: x.lat < 75.0, drop=True).load()
    terrain = xr.open_dataset(TERRAIN_FILE).interp_like(sr_target, method='nearest')[STATIC_FEATURES].load()
    climate = xr.open_zarr(CLIMATE_ZARR, group='1991_2020').interp_like(sr_target, method='nearest')[CLIMATE_FEATURES].load()
    lu = xr.open_dataset(LU_FILE).interp_like(sr_target, method='nearest').load()
    cluster_da = xr.open_zarr(CLUSTER_ZARR, group='1991_2020')['cluster'].interp_like(sr_target, method='nearest').load()
    
    ds = xr.merge([terrain, climate, lu, sr_target.rename('sr'), cluster_da], join='inner')
    df_all = ds.to_dataframe().reset_index().dropna(subset=['sr', 'cluster'] + ALL_FEATURES)
    print(f"Columns in df_all: {df_all.columns.tolist()}")
    
    # Stratified-like sampling by cluster to keep cluster representation
    df = df_all.groupby('cluster', group_keys=True).apply(lambda x: x.sample(min(len(x), SUB_SAMPLE_SIZE // K), random_state=42))
    df = df.reset_index(level='cluster')
    df = df.reset_index(drop=True)
    print(f"Columns in final df: {df.columns.tolist()}")
    print(f"Sampled {len(df)} points for training.")
    
    results_file = f"{OUT_DIR}/experiment_log.json"
    experiments = []
    
    # 1. Benchmark
    print("Running Benchmark (Baseline XGB)...")
    bench = run_experiment("Baseline_XGB", XGB_PARAMS_BASELINE, df)
    experiments.append(bench)
    print(f"Benchmark: {bench}")
    
    with open(results_file, 'w') as f:
        json.dump(experiments, f, indent=4)

if __name__ == "__main__":
    main()
