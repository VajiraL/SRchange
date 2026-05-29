import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

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
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + LU_FEATURES

# Paths
LAI_TARGET_PRESENT = os.path.join(DIR_ROOT, 'data/LAI_mean_95th_2011_2015.nc')
LAI_TARGET_HIST = os.path.join(DIR_ROOT, 'data/LAI_mean_95th_1982_1990.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_0p05.nc')
LU_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
MASK_PATH = os.path.join(DIR_ROOT, 'data/random_splits_mergedSR/ensemble_1/mask_fold_1.nc')

def load_data(clim_period, lu_period, target_path):
    print(f"Loading data for {clim_period} / {lu_period}...")
    target_ds = xr.open_dataset(target_path)
    target_da = target_ds['lai_95th']
    lat_vals, lon_vals = target_da.lat.values, target_da.lon.values
    
    terrain = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES].sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
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
        
    X = X.T
    y = target_da.values.flatten()
    
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[valid_mask], y[valid_mask], valid_mask, lat_vals, lon_vals

def benchmark():
    # Load Present (Training/Validation)
    X_pres, y_pres, mask_pres, lat_vals, lon_vals = load_data('1991_2020', '1991_2020', LAI_TARGET_PRESENT)
    
    # Load Historical (For Delta validation)
    X_hist, y_hist_obs, mask_hist, _, _ = load_data('1961_1990', '1961_1990', LAI_TARGET_HIST)
    
    # Use spatial fold for testing
    mask_da = xr.open_dataarray(MASK_PATH)
    print(f"Lat range: {lat_vals.min()} to {lat_vals.max()}")
    # We need to map this mask back to our flattened X_pres
    # But for a benchmark, a 80/20 random split on the present data is sufficient for model comparison
    # while the temporal delta check is the real "truth" test.
    
    # We need to load cluster data for X_tr to do clustered importances
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters_full = cl_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').values.flatten()
    clusters_pres = clusters_full[mask_pres]
    
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X_pres, y_pres, clusters_pres, test_size=0.2, random_state=42)
    
    # Subset for speed
    n_train = 200000
    n_test = 50000
    idx_tr = np.random.choice(len(X_train), n_train, replace=False)
    idx_te = np.random.choice(len(X_test), n_test, replace=False)
    X_tr, y_tr, c_tr = X_train[idx_tr], y_train[idx_tr], c_train[idx_tr]
    X_te, y_te, c_te = X_test[idx_te], y_test[idx_te], c_test[idx_te]

    models = {
        'XGB_MSE': {'params': {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.05, 'lambda': 10, 'alpha': 1}},
        'XGB_Poisson': {'params': {'objective': 'count:poisson', 'max_depth': 4, 'eta': 0.05, 'lambda': 10, 'alpha': 1}},
        'XGB_Gamma': {'params': {'objective': 'reg:gamma', 'max_depth': 4, 'eta': 0.05, 'lambda': 10, 'alpha': 1}},
        'RandomForest': {'type': 'RF', 'params': {'n_estimators': 100, 'max_depth': 12, 'n_jobs': 32}}
    }
    
    results = []
    
    for name, config in models.items():
        print(f"\nEvaluating {name}...")
        if config.get('type') == 'RF':
            model = RandomForestRegressor(**config['params'])
            model.fit(X_tr, y_tr)
            p_te = model.predict(X_te)
            p_hist = model.predict(X_hist)
            
            # Feature Importance
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
            print("\nTop 10 Features (RF Global):")
            print(feat_imp.head(10))
            
            # Clustered Importances
            print("\nTop 5 Features per Cluster (RF):")
            for cid in range(6):
                c_mask = (c_tr == cid)
                if c_mask.sum() < 1000: continue
                m_c = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=32)
                m_c.fit(X_tr[c_mask], y_tr[c_mask])
                c_imp = pd.Series(m_c.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
                print(f"Cluster {cid}: {', '.join(c_imp.head(5).index.tolist())}")
        else:
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dte = xgb.DMatrix(X_te, label=y_te)
            dhist = xgb.DMatrix(X_hist)
            model = xgb.train(config['params'], dtr, 300, evals=[(dte, 'val')], early_stopping_rounds=50, verbose_eval=False)
            p_te = model.predict(dte)
            p_hist = model.predict(dhist)
            
        # Stats on Present test set
        r2 = r2_score(y_te, p_te)
        rmse = np.sqrt(mean_squared_error(y_te, p_te))
        
        # Stats on Delta (Historical to Present)
        # Delta Predicted = Present_Pred (on Hist features) - Present_Pred (on Pres features)? 
        # No: Delta Predicted = Model(Pres_Features) - Model(Hist_Features)
        # We need model predictions on ALL present features
        if config.get('type') == 'RF':
            p_pres_all = model.predict(X_hist) # Warning: this is predicting on Hist features to compare with y_hist_obs
            # Actually, to validate delta:
            # delta_obs = y_pres - y_hist  (where both valid)
            # delta_pred = model(X_pres) - model(X_hist)
            pass # We'll do a simpler check: Correlation of Predicted Delta vs Observed Delta
        
        # To avoid massive memory use, let's just use a sample of 100k points for delta check
        n_delta = 100000
        common_mask = mask_pres & mask_hist
        common_idx = np.where(common_mask)[0]
        sample_idx = np.random.choice(common_idx, n_delta, replace=False)
        
        # Get data for sample
        # X_pres_s = X_temp_pres[sample_idx] ... this requires keeping full X
        # Let's just do it on the subset we loaded
        
        # Actually, let's re-align X_pres and X_hist
        # Find common indices
        indices_pres = np.where(mask_pres)[0]
        indices_hist = np.where(mask_hist)[0]
        common = np.intersect1d(indices_pres, indices_hist)
        sample_common = np.random.choice(common, 50000, replace=False)
        
        # Extract features for sample_common
        # This is getting complex. Let's just evaluate the model's bias in predicting y_hist_obs
        # If the model is oversensitive, it will under-predict y_hist_obs (predicting too much browning back in time? 
        # or predicting too much browning forward in time?)
        # User said: "LAI change from noticeable lai decrease (than observed) in places with human LU"
        # This means Model(Present) - Model(Historical) << Observed(Present) - Observed(Historical)
        
        # Bias check: Mean(Predicted Delta) - Mean(Observed Delta)
        # We need model predictions on the same points for both periods.
        
        # Re-loading a small aligned sample
        sample_lats = np.random.choice(len(xr.open_dataset(LAI_TARGET_PRESENT).lat), 100)
        # ... Forget it, I'll just use the full mask and extract
        
        # EVALUATE DELTA
        # We need p_pres and p_hist for the same pixels
        # Let's use 100k valid pixels from the common set, shuffled
        indices_pres_map = {idx: i for i, idx in enumerate(np.where(mask_pres)[0])}
        indices_hist_map = {idx: i for i, idx in enumerate(np.where(mask_hist)[0])}
        common_pixels = [idx for idx in common if idx in indices_pres_map and idx in indices_hist_map]
        np.random.shuffle(common_pixels)
        cp = common_pixels[:100000]
        
        X_p_s = X_pres[[indices_pres_map[idx] for idx in cp]]
        X_h_s = X_hist[[indices_hist_map[idx] for idx in cp]]
        y_p_s = y_pres[[indices_pres_map[idx] for idx in cp]]
        y_h_s = y_hist_obs[[indices_hist_map[idx] for idx in cp]]
        
        if config.get('type') == 'RF':
            pred_p = model.predict(X_p_s)
            pred_h = model.predict(X_h_s)
        else:
            pred_p = model.predict(xgb.DMatrix(X_p_s))
            pred_h = model.predict(xgb.DMatrix(X_h_s))
            
        d_obs = y_p_s - y_h_s
        d_pred = pred_p - pred_h
        
        delta_corr, _ = pearsonr(d_obs, d_pred)
        delta_bias = np.mean(d_pred) - np.mean(d_obs)
        
        # Zonal Bias (Delta Predicted - Delta Observed by Latitude)
        # We need the latitudes for X_p_s
        cp_arr = np.array(cp)
        lats_s = lat_vals[cp_arr // len(lon_vals)]
        print(f"  Sample Lat range: {lats_s.min():.2f} to {lats_s.max():.2f}")
        df_zonal = pd.DataFrame({'lat': lats_s, 'd_obs': d_obs, 'd_pred': d_pred})
        zonal_bias = df_zonal.groupby('lat').mean()
        # Equator is approx lat 0
        eq_mask = np.abs(zonal_bias.index) < 5
        print(f"  Points in Equator mask: {eq_mask.sum()}")
        eq_bias = zonal_bias.loc[eq_mask, 'd_pred'].mean() - zonal_bias.loc[eq_mask, 'd_obs'].mean()
        
        print(f"  R2: {r2:.4f} | RMSE: {rmse:.4f}")
        print(f"  Delta Corr: {delta_corr:.4f} | Delta Bias: {delta_bias:.4f} | Eq Delta Bias: {eq_bias:.4f}")
        
        results.append({
            'Model': name,
            'R2': r2,
            'RMSE': rmse,
            'Delta_Corr': delta_corr,
            'Delta_Bias': delta_bias,
            'Eq_Delta_Bias': eq_bias
        })

    df_res = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(df_res)
    df_res.to_csv(os.path.join(DIR_ROOT, 'output_lai/benchmark_results.csv'), index=False)

if __name__ == "__main__":
    benchmark()
