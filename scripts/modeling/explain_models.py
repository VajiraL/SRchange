import os
import gc
import json
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_hand_0p05.nc')
LU_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/explanation_hand')
PLOT_DIR = os.path.join(OUTPUT_BASE, 'plots')
DATA_DIR = os.path.join(OUTPUT_BASE, 'data')

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

STATIC_FEATURES = ['elev', 'slope', 'roughness', 'hand']
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

XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': 256,
    'device': 'cpu',
    'eval_metric': 'rmse',
}

GLOBAL_SHAP_CAP = 1000000 # 500k
CLUSTER_SHAP_CAP = 500000 # 200k

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Data Loading & Preparation
# ---------------------------
def load_training_data():
    log("Loading target SR data...")
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    
    log("Loading terrain features...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    log("Loading climate features (1991-2020)...")
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    
    log("Loading land use features (1991-2020)...")
    lu_ds = xr.open_dataset(os.path.join(LU_FEATURES_DIR, '1991_2020.nc'))[LU_FEATURES]
    lu = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]
    
    log("Loading climate clusters (1991-2020)...")
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters = cl_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').values
    
    features_3d = np.concatenate([terrain, clim, lu], axis=0)
    X_full = features_3d.reshape(len(ALL_FEATURES), -1).T
    y_full = sr_target.values.flatten()
    c_full = clusters.flatten()
    
    df = pd.DataFrame(X_full, columns=ALL_FEATURES)
    df['sr'] = y_full
    df['cluster'] = c_full
    
    log(f"Initial row count: {len(df)}")
    df = df.dropna(subset=['sr'] + ALL_FEATURES)
    log(f"Valid row count: {len(df)}")
    
    return df

# ---------------------------
# 3. Explanation Logic
# ---------------------------
def explain_model(model, X_sample, name, cap):
    log(f"Explaining model: {name}")
    
    # Built-in Importance
    importance_types = ['weight', 'gain', 'cover']
    imp_df = pd.DataFrame(index=ALL_FEATURES)
    for it in importance_types:
        scores = model.get_score(importance_type=it)
        imp_df[it] = imp_df.index.map(lambda x: scores.get(x, 0))
    
    imp_df.to_csv(os.path.join(OUTPUT_BASE, f"importance_{name}.csv"))
    
    # Sampling for SHAP
    s_size = min(len(X_sample), cap)
    # If 20% of cluster is larger than 200k, we cap at 250k.
    X_shap = X_sample.sample(n=s_size, random_state=42)
    
    log(f"Computing SHAP values for {len(X_shap)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    expected_value = explainer.expected_value
    
    # Save raw SHAP data
    np.savez_compressed(
        os.path.join(DATA_DIR, f"shap_{name}.npz"),
        shap_values=shap_values,
        X=X_shap.values,
        expected_value=expected_value,
        feature_names=ALL_FEATURES
    )
    
    # Basic Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.title(f"SHAP Summary: {name} (N={len(X_shap)})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"shap_summary_{name}.png"), dpi=300)
    plt.close()
    
    # Mean SHAP Bar Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
    plt.title(f"Mean |SHAP|: {name} (N={len(X_shap)})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"shap_bar_{name}.png"), dpi=300)
    plt.close()

# ---------------------------
# 4. Main
# ---------------------------
def main():
    df = load_training_data()
    
    # 1. Global Model
    log("--- GLOBAL MODEL ---")
    X_glob = df[ALL_FEATURES]
    y_glob = df['sr']
    
    dtrain_glob = xgb.DMatrix(X_glob, label=y_glob)
    model_glob = xgb.train(XGB_PARAMS, dtrain_glob, num_boost_round=2000) 
    
    explain_model(model_glob, df[ALL_FEATURES], "global", GLOBAL_SHAP_CAP)
    
    del dtrain_glob, model_glob, X_glob, y_glob
    gc.collect()
    
    # 2. Cluster Models
    log("--- CLUSTER MODELS ---")
    cluster_ids = sorted([int(c) for c in df['cluster'].unique() if c >= 0])
    
    all_imp = []
    
    for cid in cluster_ids:
        log(f"Processing Cluster {cid}...")
        df_c = df[df['cluster'] == cid]
        if len(df_c) < 1000:
            log(f"Cluster {cid} too small, skipping.")
            continue
            
        X_c = df_c[ALL_FEATURES]
        y_c = df_c['sr']
        
        dtrain_c = xgb.DMatrix(X_c, label=y_c)
        model_c = xgb.train(XGB_PARAMS, dtrain_c, num_boost_round=1600)
        
        # 20% or 500k
        c_cap = max(int(len(df_c) * 0.2), CLUSTER_SHAP_CAP)
        # Apply an absolute upper bound to prevent massive memory usage if 20% is huge
        c_cap = min(c_cap, 500000) 
        
        explain_model(model_c, df_c[ALL_FEATURES], f"cluster_{cid}", c_cap)
        
        # Collect importance for comparison
        scores = model_c.get_score(importance_type='gain')
        for feat, score in scores.items():
            all_imp.append({'cluster': cid, 'feature': feat, 'gain': score})
            
        del dtrain_c, model_c, X_c, y_c, df_c
        gc.collect()
        
    # Save combined importance for easy cross-cluster comparison
    pd.DataFrame(all_imp).to_csv(os.path.join(OUTPUT_BASE, "cluster_importance_comparison.csv"), index=False)
    
    log("Explanation workflow complete.")

if __name__ == "__main__":
    main()