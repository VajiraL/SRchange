import os
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.spatial import KDTree

# ---------------------------
# Configuration
# ---------------------------
ROOT = "/home/vajira/SRchange"
N_FOLDS = 5
MORAN_SAMPLES = 20000  # Subsample size for Moran's I calculation
XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'tree_method': 'hist',
    'device': 'cuda',
}

# Feature definitions (from evaluate_k_sweep.py logic)
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
OUT_CSV = f"{ROOT}/output_SR/sampling_assessment_results.csv"

def calculate_morans_i(coords, residuals, n_samples=MORAN_SAMPLES):
    """Calculates Global Moran's I on a subsample of residuals."""
    if len(residuals) > n_samples:
        idx = np.random.choice(len(residuals), n_samples, replace=False)
        coords = coords[idx]
        residuals = residuals[idx]
    
    # Normalize residuals
    z = residuals - np.mean(residuals)
    
    # Use KDTree for efficient nearest neighbor weights (within ~1 degree)
    tree = KDTree(coords)
    # We define weights as 1/distance for neighbors within 1.0 units
    pairs = tree.query_pairs(r=1.0)
    
    w_sum = 0
    numerator = 0
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist > 0:
            w = 1.0 / dist
            numerator += w * (z[i] * z[j])
            w_sum += w
            
    denominator = np.sum(z**2)
    if w_sum == 0 or denominator == 0:
        return 0.0
        
    # Moran's I = (N / W) * (sum(w_ij * z_i * z_j) / sum(z_i^2))
    n = len(residuals)
    return (n / (2 * w_sum)) * (numerator / denominator)

def get_block_ids(df, size=1.0):
    lat_bins = np.floor(df['lat'] / size)
    lon_bins = np.floor(df['lon'] / size)
    return (lat_bins * 1000 + lon_bins).astype(int)

def main():
    print("Loading data...")
    sr_target = xr.open_dataarray(TARGET_FILE).where(lambda x: x.lat < 75.0, drop=True).load()
    terrain = xr.open_dataset(TERRAIN_FILE).interp_like(sr_target, method='nearest')[STATIC_FEATURES].load()
    climate = xr.open_zarr(CLIMATE_ZARR, group='1991_2020').interp_like(sr_target, method='nearest')[CLIMATE_FEATURES].load()
    lu = xr.open_dataset(LU_FILE).interp_like(sr_target, method='nearest').load()
    
    ds = xr.merge([terrain, climate, lu, sr_target.rename('sr')], join='inner')
    df = ds.to_dataframe().reset_index().dropna(subset=['sr'] + ALL_FEATURES)
    
    # Subsample for faster assessment if needed, but let's try full first 
    # as we have huge RAM. If it's too slow, we can slice.
    if len(df) > 1000000:
        print(f"Sampling 1M pixels from {len(df)} for assessment efficiency...")
        df = df.sample(1000000, random_state=42).reset_index(drop=True)

    strategies = [
        ('Random', None),
        ('Block_1deg', 1.0),
        ('Block_5deg', 5.0)
    ]
    
    all_results = []

    # 2. Loop over strategies
    for name, block_size in strategies:
        print(f"Evaluating Strategy: {name}")
        
        # We use k=6 as established for the study
        k = 6
        cluster_da = xr.open_zarr(f"{ROOT}/data/climate_clusters.zarr", group='1991_2020')['cluster'].interp_like(sr_target, method='nearest').load()
        df_k = df.copy()
        df_k['cluster'] = cluster_da.sel(lat=xr.DataArray(df_k['lat'], dims='z'), 
                                       lon=xr.DataArray(df_k['lon'], dims='z'), 
                                       method='nearest').values

        if block_size is None:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_indices = list(kf.split(df_k))
        else:
            block_ids = get_block_ids(df_k, size=block_size)
            unique_blocks = np.unique(block_ids)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_indices = []
            for train_b_idx, test_b_idx in kf.split(unique_blocks):
                tr_blocks = unique_blocks[train_b_idx]
                te_blocks = unique_blocks[test_b_idx]
                fold_indices.append((
                    np.where(np.isin(block_ids, tr_blocks))[0],
                    np.where(np.isin(block_ids, te_blocks))[0]
                ))

        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            print(f"  Fold {fold+1}/{N_FOLDS}")
            df_train = df_k.iloc[train_idx]
            df_test = df_k.iloc[test_idx]
            
            fold_preds = np.full(len(df_test), np.nan)
            
            for cid in range(k):
                sub_tr = df_train[df_train['cluster'] == cid]
                sub_te = df_test[df_test['cluster'] == cid]
                
                if len(sub_tr) < 100 or len(sub_te) == 0:
                    continue
                
                dtr = xgb.DMatrix(sub_tr[ALL_FEATURES], label=sub_tr['sr'])
                model = xgb.train(XGB_PARAMS, dtr, num_boost_round=500)
                
                dte = xgb.DMatrix(sub_te[ALL_FEATURES])
                fold_preds[df_test['cluster'] == cid] = model.predict(dte)
            
            # Filter valid predictions
            valid = np.isfinite(fold_preds)
            y_true = df_test['sr'].values[valid]
            y_pred = fold_preds[valid]
            residuals = y_true - y_pred
            
            # Metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Spatial Autocorrelation (Moran's I)
            coords = df_test[['lon', 'lat']].values[valid]
            moran_i = calculate_morans_i(coords, residuals)
            
            all_results.append({
                'strategy': name,
                'fold': fold + 1,
                'r2': r2,
                'mae': mae,
                'moran_i': moran_i,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Assessment complete. Results saved to {OUT_CSV}")
    print("\nSummary Statistics:")
    print(results_df.groupby('strategy')[['r2', 'mae', 'moran_i']].mean())

if __name__ == "__main__":
    main()
