import os
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

# Configuration (mirrored from Albedo script)
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

DIR_ROOT = "/home/vajira/SRchange"
ALBEDO_TARGET_PATH = os.path.join(DIR_ROOT, 'data/albedo_mean_10th_2011_2020.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_0p05.nc')
LU_FILE = os.path.join(DIR_ROOT, 'data/LU_features/1991_2020.nc')

def run_diagnostic():
    print("Loading data for Albedo Cluster 3 diagnostic...")
    target_da = xr.open_dataset(ALBEDO_TARGET_PATH)['albedo_10th']
    lat_vals, lon_vals = target_da.lat.values, target_da.lon.values
    
    # Load assets
    terrain = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES].sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    clim = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES].sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    
    lu = xr.open_dataset(LU_FILE).sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]
    
    clusters = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster'].sel(lat=lat_vals, lon=lon_vals, method='nearest').values
    
    # Flatten
    features_3d = np.concatenate([terrain, clim, lu], axis=0)
    X = features_3d.reshape(len(ALL_FEATURES), -1).T
    y = target_da.values.flatten()
    c = clusters.flatten()
    
    # Filter for Cluster 3 and valid pixels
    mask = (c == 3) & (~np.isnan(y)) & (~np.isnan(X).any(axis=1))
    X_sub = X[mask]
    y_sub = y[mask]
    
    print(f"Cluster 3 pixels: {len(y_sub)}")
    
    # Train quick XGBoost
    dtrain = xgb.DMatrix(X_sub, label=y_sub, feature_names=ALL_FEATURES)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.1,
        'tree_method': 'hist',
        'nthread': 32
    }
    model = xgb.train(params, dtrain, 200)
    
    # Feature Importance
    importance = model.get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Features for Albedo Cluster 3 (by Gain):")
    for f, v in sorted_imp[:20]:
        print(f"{f:25}: {v:.4f}")

if __name__ == "__main__":
    run_diagnostic()
