import os
import gc
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
LU_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features_2')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/explanation_hand')
LOCAL_PLOT_DIR = os.path.join(OUTPUT_BASE, 'local_plots')

os.makedirs(LOCAL_PLOT_DIR, exist_ok=True)

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

# Define your target locations here: {"Location_Name": (lat, lon)}
TARGET_LOCATIONS = {
    "Amazon_Basin": (-1.18, -62.71),
    "South_Brazil": (-21.38, -47.78),
    "Alabama_deforestation": (32.87, -85.72),
    "Sudd_wetlands": (6.91, 31.17),
    "South_Sudan": (8.01, 32.18),
    "South_Madagascar": (-22.88, 45.81),
    "Wilpattu": (8.325, 80.03),
    "South_Sweden": (56.55, 15.62),
    "Loess_Plateau": (36.57, 109.33),
    "Vic_Agri": (-36.37, 145.12),
    "WA_bush": (-32.42, 116.23),
    "WA_agri": (-31.98, 111.12),
    "Orinoco_COL_sav": (6.48, -67.67),
    "Orinoco_VEN_forest": (6.37, -66.82),
}

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Extract Data for Specific Coordinates
# ---------------------------
def extract_location_data(lat_lon_dict):
    log("Extracting raw feature grids...")
    sr_target = xr.open_dataarray(SR_TARGET_PATH)
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    lu_ds = xr.open_dataset(os.path.join(LU_FEATURES_DIR, '1991_2020.nc'))[LU_FEATURES]
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    
    rows = []
    for name, (lat, lon) in lat_lon_dict.items():
        log(f"Extracting data for {name} (Lat: {lat}, Lon: {lon})...")
        
        # Use nearest neighbor matching to pull point values out of the NetCDF/Zarr arrays
        target_val = float(sr_target.sel(lat=lat, lon=lon, method='nearest').values)
        cluster_val = int(cl_ds.sel(lat=lat, lon=lon, method='nearest').values)
        
        t_vals = terrain_ds.sel(lat=lat, lon=lon, method='nearest').to_array().values.flatten()
        c_vals = clim_ds.sel(lat=lat, lon=lon, method='nearest').to_array().values.flatten()
        l_vals = lu_ds.sel(lat=lat, lon=lon, method='nearest').to_array().values.flatten()
        
        # Handle cases where some datasets retain extra dimensions (like your original script's 4D check)
        if len(t_vals) > len(STATIC_FEATURES): t_vals = t_vals[:len(STATIC_FEATURES)]
        if len(c_vals) > len(CLIMATE_FEATURES): c_vals = c_vals[:len(CLIMATE_FEATURES)]
        if len(l_vals) > len(LU_FEATURES): l_vals = l_vals[:len(LU_FEATURES)]
        
        feat_all = np.concatenate([t_vals, c_vals, l_vals])
        
        data_dict = dict(zip(ALL_FEATURES, feat_all))
        data_dict['location_name'] = name
        data_dict['lat'] = lat
        data_dict['lon'] = lon
        data_dict['sr'] = target_val
        data_dict['cluster'] = cluster_val
        
        rows.append(data_dict)
        
    return pd.DataFrame(rows)

# ---------------------------
# 3. Generate Local SHAP Explanations
# ---------------------------
def explain_single_estimate(model, X_row, name, model_type):
    """
    Generates waterfall and force plots for a single observation row.
    """
    log(f"Generating local SHAP plots for {name} using [{model_type}] model...")
    
    # Initialize tree explainer
    explainer = shap.TreeExplainer(model)
    
    # To get proper visualization objects for waterfall/force plots, 
    # we use the modern Explainer interface __call__ structure
    shap_values_obj = explainer(X_row)
    
    # 1. Waterfall Plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_obj[0], max_display=15, show=False)
    plt.title(f"SHAP Local Waterfall: {name} ({model_type} Model)", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(LOCAL_PLOT_DIR, f"{name}_{model_type}_waterfall.png"), dpi=300)
    plt.close()
    
    # 2. Force Plot
    plt.figure(figsize=(12, 4))
    # Matplotlib output format requires explicit matplotlib=True toggle
    shap.plots.force(
        explainer.expected_value, 
        shap_values_obj.values[0], 
        X_row.iloc[0], 
        matplotlib=True, 
        show=False,
        contribution_threshold=0.05
    )
    plt.title(f"SHAP Force Plot: {name} ({model_type} Model)", fontsize=12, pad=30)
    plt.savefig(os.path.join(LOCAL_PLOT_DIR, f"{name}_{model_type}_force.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# 4. Main Workflow
# ---------------------------
def main():
    # Load all training data to train baseline background models 
    # (or you can modify this step to load a pre-saved XGBoost json model file)
    from explain_models import load_training_data
    df_train = load_training_data()
    
    # Extract target points
    df_locs = extract_location_data(TARGET_LOCATIONS)
    
    # Loop over each target location
    for idx, row in df_locs.iterrows():
        loc_name = row['location_name']
        cluster_id = int(row['cluster'])
        
        # Check if row data is invalid
        if pd.isna(row['sr']) or np.isnan(cluster_id) or cluster_id < 0:
            log(f"Skipping {loc_name} due to missing target or cluster indices.")
            continue
            
        X_single = pd.DataFrame([row[ALL_FEATURES]])
        y_single = pd.Series([row['sr']])
        
        # --- Run Explanations on Global Model ---
        log(f"Building Global Model context for {loc_name}...")
        X_glob = df_train[ALL_FEATURES]
        y_glob = df_train['sr']
        dtrain_glob = xgb.DMatrix(X_glob, label=y_glob)
        model_glob = xgb.train(XGB_PARAMS, dtrain_glob, num_boost_round=2000) 
        
        explain_single_estimate(model_glob, X_single, loc_name, "global")
        
        del dtrain_glob, model_glob, X_glob, y_glob
        gc.collect()
        
        # --- Run Explanations on specific Cluster Model ---
        log(f"Building Cluster {cluster_id} Model context for {loc_name}...")
        df_c = df_train[df_train['cluster'] == cluster_id]
        
        if len(df_c) >= 1000:
            X_c = df_c[ALL_FEATURES]
            y_c = df_c['sr']
            dtrain_c = xgb.DMatrix(X_c, label=y_c)
            model_c = xgb.train(XGB_PARAMS, dtrain_c, num_boost_round=1600)
            
            explain_single_estimate(model_c, X_single, loc_name, f"cluster_{cluster_id}")
            
            del dtrain_c, model_c, X_c, y_c
        else:
            log(f"Cluster {cluster_id} training data too small; skipped cluster local explanation.")
            
        del df_c
        gc.collect()

    log("Local location analysis complete.")

if __name__ == "__main__":
    main()