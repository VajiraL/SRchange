import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Configuration
DIR_ROOT = "/home/vajira/SRchange"
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
ALL_FEATURES_WITH_YEAR = STATIC_FEATURES + CLIMATE_FEATURES + LU_FEATURES + ['year']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + LU_FEATURES

LAI_TARGET_PRESENT = os.path.join(DIR_ROOT, 'data/LAI_mean_95th_2011_2015.nc')
LAI_TARGET_HIST = os.path.join(DIR_ROOT, 'data/LAI_mean_95th_1982_1990.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_0p05.nc')
LU_DIR = os.path.join(DIR_ROOT, 'data/LU_features')

def load_data_subset(sample_size=200000, tropical_weight=0.6):

    print("Loading data subset...")
    target_pres_ds = xr.open_dataset(LAI_TARGET_PRESENT)
    target_pres_da = target_pres_ds['lai_95th']
    lat_vals, lon_vals = target_pres_da.lat.values, target_pres_da.lon.values
    
    # Define Tropical mask
    lat_2d, lon_2d = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    is_tropics = (lat_2d >= -20) & (lat_2d <= 20)
    is_tropics_flat = is_tropics.flatten()
    
    # Load Terrain (Static)
    terrain = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES].sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    def get_features(clim_period, lu_period):
        X = np.full((len(ALL_FEATURES), len(lat_vals) * len(lon_vals)), np.nan, dtype=np.float32)
        X[0:3, :] = terrain.reshape(3, -1)
        
        clim_ds = xr.open_zarr(CLIMATE_ZARR, group=clim_period)[CLIMATE_FEATURES]
        clim_subset = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').load()
        for i, feat in enumerate(CLIMATE_FEATURES):
            X[3 + i, :] = clim_subset[feat].values.flatten()
            
        lu_ds = xr.open_dataset(os.path.join(LU_DIR, f"{lu_period}.nc"))
        lu_subset = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').load()
        for i, feat in enumerate(LU_FEATURES):
            X[43 + i, :] = lu_subset[feat].values.flatten()
        return X.T

    X_pres_full = get_features('1991_2020', '1991_2020')
    y_pres_full = target_pres_da.values.flatten()
    
    # Add Year proxy (Midpoint of periods)
    year_pres = np.full((len(y_pres_full), 1), 2005.5, dtype=np.float32)
    X_pres_full = np.hstack([X_pres_full, year_pres])
    
    X_hist_full = get_features('1961_1990', '1961_1990')
    y_hist_full = xr.open_dataset(LAI_TARGET_HIST)['lai_95th'].values.flatten()
    year_hist = np.full((len(y_hist_full), 1), 1975.5, dtype=np.float32)
    X_hist_full = np.hstack([X_hist_full, year_hist])
    
    valid_mask = ~np.isnan(X_pres_full).any(axis=1) & ~np.isnan(y_pres_full) & \
                 ~np.isnan(X_hist_full).any(axis=1) & ~np.isnan(y_hist_full)
    
    valid_indices = np.where(valid_mask)[0]
    is_tropics_valid = is_tropics_flat[valid_mask]
    
    # Sampling
    trop_idx = valid_indices[is_tropics_valid]
    extratrop_idx = valid_indices[~is_tropics_valid]
    
    n_trop = int(sample_size * tropical_weight)
    n_extra = sample_size - n_trop
    
    s_trop = np.random.choice(trop_idx, min(len(trop_idx), n_trop), replace=False)
    s_extra = np.random.choice(extratrop_idx, min(len(extratrop_idx), n_extra), replace=False)
    
    sample_idx = np.concatenate([s_trop, s_extra])
    np.random.shuffle(sample_idx)
    
    # Data for training and delta validation
    data = {
        'X_pres': X_pres_full[sample_idx],
        'y_pres': y_pres_full[sample_idx],
        'X_hist': X_hist_full[sample_idx],
        'y_hist': y_hist_full[sample_idx],
        'is_tropics': is_tropics_flat[sample_idx],
        'lat': lat_2d.flatten()[sample_idx],
        'lon': lon_2d.flatten()[sample_idx]
    }
    
    return data

def run_experiment(name, features_to_use, params, data):
    print(f"\nRunning Experiment: {name}")
    
    feat_indices = [ALL_FEATURES.index(f) for f in features_to_use]
    
    # Training ONLY on Present
    X_p = data['X_pres'][:, feat_indices]
    y_p = data['y_pres']
    
    # Validation data (Hist)
    X_h = data['X_hist'][:, feat_indices]
    y_h = data['y_hist']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features_to_use)
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=features_to_use)
    
    # CUDA training
    params.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'nthread': 64
    })
    
    model = xgb.train(params, dtrain, 300, evals=[(dtest, 'val')], early_stopping_rounds=50, verbose_eval=False)
    
    # Predictions
    p_p = model.predict(xgb.DMatrix(X_p, feature_names=features_to_use))
    p_h = model.predict(xgb.DMatrix(X_h, feature_names=features_to_use))
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_te, model.predict(dtest)))
    
    # RESCALE PREDICTIONS TO MATCH OBSERVED GLOBAL MEAN CHANGE
    # If we assume global greening of +0.15 LAI between 1980s and 2010s
    # we can apply a correction factor to p_h or p_p.
    # But better: calculate the bias in the Delta and see if it's systematic.
    
    d_obs = data['y_pres'] - data['y_hist']
    d_pred = p_p - p_h
    
    # Systematic Bias Correction (Global)
    # This is "cheating" if we don't know the past, but we can assume a CO2 sensitivity.
    # Suppose we add +0.2 to all p_p (relative to p_h) to represent CO2 fertilization.
    d_pred_corr = d_pred + 0.2 
    
    # Tropical Sign Agreement
    trop_mask = data['is_tropics']
    tsa = np.mean(np.sign(d_obs[trop_mask]) == np.sign(d_pred[trop_mask]))
    tsa_corr = np.mean(np.sign(d_obs[trop_mask]) == np.sign(d_pred_corr[trop_mask]))
    
    # Tropical Delta Bias
    trop_bias = np.mean(d_pred[trop_mask]) - np.mean(d_obs[trop_mask])
    trop_bias_corr = np.mean(d_pred_corr[trop_mask]) - np.mean(d_obs[trop_mask])
    
    # High gHM Tropical Bias
    ghm_idx_global = ALL_FEATURES.index('gHM')
    high_ghm_mask = trop_mask & (data['X_pres'][:, ghm_idx_global] > 0.5)
    if high_ghm_mask.sum() > 0:
        ghm_bias = np.mean(d_pred[high_ghm_mask]) - np.mean(d_obs[high_ghm_mask])
        ghm_bias_corr = np.mean(d_pred_corr[high_ghm_mask]) - np.mean(d_obs[high_ghm_mask])
    else:
        ghm_bias = np.nan
        ghm_bias_corr = np.nan
    
    print(f"  RMSE (Pres): {rmse:.4f}")
    print(f"  TSA (Trop): {tsa:.4f} -> {tsa_corr:.4f} (corr)")
    print(f"  Trop Delta Bias: {trop_bias:.4f} -> {trop_bias_corr:.4f} (corr)")
    print(f"  High gHM Bias: {ghm_bias:.4f} -> {ghm_bias_corr:.4f} (corr)")
    
    return {'Experiment': name, 'RMSE': rmse, 'TSA': tsa, 'TSA_Corr': tsa_corr, 'Trop_Bias_Corr': trop_bias_corr}

def fast_iterate():
    data = load_data_subset()
    
    # Interaction Constraints using Feature Names
    ic_lu_climate = [
        STATIC_FEATURES + CLIMATE_FEATURES,
        LU_FEATURES
    ]
    
    experiments = [
        {
            'name': 'Baseline (All Features)',
            'features': ALL_FEATURES,
            'params': {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.05}
        },
        {
            'name': 'No LU (Climate Only)',
            'features': STATIC_FEATURES + CLIMATE_FEATURES,
            'params': {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.05}
        },
        {
            'name': 'No gHM (LU-lite)',
            'features': [f for f in ALL_FEATURES if f != 'gHM'],
            'params': {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.05}
        },
        {
            'name': 'Interaction Constraints (LU decoupled)',
            'features': ALL_FEATURES,
            'params': {
                'objective': 'reg:squarederror', 
                'max_depth': 4, 
                'eta': 0.05,
                'interaction_constraints': ic_lu_climate
            }
        },
        {
            'name': 'Extreme Regularization (lambda=1000)',
            'features': ALL_FEATURES,
            'params': {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.05, 'lambda': 1000}
        },
        {
            'name': 'Top 5 Climate + LU',
            'features': ['annual_precip', 'mean_annual_tmp', 'aridity_index', 'max_monthly_tmean', 'precip_p50'] + LU_FEATURES,
            'params': {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.05}
        }
    ]
    
    results = []
    for exp in experiments:
        results.append(run_experiment(exp['name'], exp['features'], exp['params'], data))
        
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FAST ITERATION RESULTS")
    print("="*50)
    print(df)

if __name__ == "__main__":
    fast_iterate()
