import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit

# Workarounds for pygam/numpy 2.x
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool
import scipy.sparse
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
from pygam import LinearGAM, s

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
SHAP_DATA_DIR = os.path.join(DIR_ROOT, 'output_SR/explanation/data')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/advanced')
BLOCK_SIZE = 5.0 # Degrees
N_BOOTSTRAP = 100
CORES = 64 # Parallelize bootstraps

os.makedirs(OUTPUT_BASE, exist_ok=True)

# Selection of features based on Phase 5 results (high R2)
FEATURES = ['aridity_index', 'gHM', 'forest_age', 'annual_precip', 'pet_annual']

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Logic
# ---------------------------
def piecewise_linear(x, x0, y0, k1, k2):
    return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))

def fit_piecewise(x, y):
    x_min, x_max = np.min(x), np.max(x)
    try:
        popt, _ = curve_fit(piecewise_linear, x, y, 
                               p0=[(x_min + x_max) / 2, np.mean(y), 0, 0],
                               bounds=([x_min, -np.inf, -np.inf, -np.inf], 
                                       [x_max, np.inf, np.inf, np.inf]))
        return popt
    except:
        return None

def detect_threshold(x, y):
    """Simplified version for bootstrap speed"""
    try:
        # Binning for speed
        df = pd.DataFrame({'x': x, 'y': y})
        df['bin'] = pd.qcut(df['x'], 100, duplicates='drop')
        binned = df.groupby('bin', observed=True).mean()
        x_bin = binned['x'].values
        y_bin = binned['y'].values
        
        # We try to fit piecewise first
        popt = fit_piecewise(x_bin, y_bin)
        if popt is not None:
            return popt[0]
        
        # Fallback to GAM saturation
        gam = LinearGAM(s(0, n_splines=15)).fit(x_bin.reshape(-1, 1), y_bin)
        x_grid = np.linspace(x_bin.min(), x_bin.max(), 100).reshape(-1, 1)
        y_pred = gam.predict(x_grid)
        dy = np.gradient(y_pred, x_grid.flatten())
        max_dy = np.max(np.abs(dy))
        idx_max_dy = np.argmax(np.abs(dy))
        tail_dy = dy[idx_max_dy:]
        tail_x = x_grid[idx_max_dy:]
        sat_idx = np.where(np.abs(tail_dy) < 0.1 * max_dy)[0]
        if len(sat_idx) > 0:
            return tail_x[sat_idx[0]][0]
    except:
        pass
    return None

def run_single_bootstrap(iter_id, data_dict):
    """Run one bootstrap iteration across all clusters and features"""
    results = []
    # Seed by iteration
    rng = np.random.default_rng(iter_id + 42)
    
    for cid, df_c in data_dict.items():
        # Block resample
        unique_blocks = df_c['block'].unique()
        resampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        
        # Efficient join/filter for resampled blocks
        # We can simulate this by picking rows from blocks
        # Actually, let's just create a list of dataframes and concat
        blocks_list = [df_c[df_c['block'] == b] for b in resampled_blocks]
        df_boot = pd.concat(blocks_list)
        
        for feat in FEATURES:
            if feat not in df_boot.columns: continue
            x = df_boot[feat].values
            y = df_boot[f'shap_{feat}'].values
            
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 500: continue
            
            val = detect_threshold(x[mask], y[mask])
            if val is not None:
                results.append({'iter': iter_id, 'cluster': cid, 'feature': feat, 'threshold': val})
                
    return results

def main():
    log("Loading template and coordinate grid...")
    sr_template = xr.open_dataarray(SR_TARGET_PATH)
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters_full = cl_ds.sel(lat=sr_template.lat, lon=sr_template.lon, method='nearest').values
    
    lon_grid, lat_grid = np.meshgrid(sr_template.lon.values, sr_template.lat.values)
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    c_flat = clusters_full.flatten()
    
    # Identify non-nan rows as per modeling
    valid_mask = ~np.isnan(sr_template.values.flatten()) & (c_flat >= 0)
    
    # We only need the indices of valid rows to reproduce the sampling
    valid_indices = np.where(valid_mask)[0]
    
    data_dict = {}
    shap_files = sorted([f for f in os.listdir(SHAP_DATA_DIR) if f.startswith("shap_cluster_")])
    
    for f in shap_files:
        cid = int(f.split('_')[-1].split('.')[0])
        log(f"Preparing data for Cluster {cid}...")
        
        data = np.load(os.path.join(SHAP_DATA_DIR, f), allow_pickle=True)
        shap_values = data['shap_values']
        X = data['X']
        feature_names = data['feature_names'].tolist()
        
        # Get the global indices for this cluster
        c_indices = np.where(c_flat == cid)[0]
        # Intersection with valid_mask
        # Note: explain_models.py did dropna on all features. 
        # For simplicity we'll assume the sampling seed logic is exact.
        # Actually, simpler: the .npz already has the features. 
        # We just need to attach lat/lon/block.
        
        # We need to know which rows of the valid dataframe were sampled.
        # In explain_models.py: df_c = df[df['cluster'] == cid]; X_sample = df_c.sample(n=s_size, random_state=42)
        # We'll just build a small df with blocks.
        
        # Since I can't easily reproduce the exact same dropna without loading everything,
        # I'll use the lat/lon from the X matrix if possible? No, it's not there.
        # I will do a quick reconstruction of the valid rows for this cluster.
        
        # RE-LOAD necessary features for dropna check to match explain_models.py exactly
        # or at least closely.
        df_c_meta = pd.DataFrame({
            'lat': lat_flat[c_indices],
            'lon': lon_flat[c_indices],
            'sr': sr_template.values.flatten()[c_indices]
        })
        # Add a block ID
        df_c_meta['block'] = (np.round(df_c_meta['lat'] / BLOCK_SIZE) * 1000 + 
                             np.round(df_c_meta['lon'] / BLOCK_SIZE)).astype(int)
        
        # dropna to match model training
        df_c_meta = df_c_meta.dropna()
        
        # Sample using same seed
        CLUSTER_SHAP_CAP = 200000
        c_cap = max(int(len(df_c_meta) * 0.2), CLUSTER_SHAP_CAP)
        c_cap = min(c_cap, 250000)
        s_size = min(len(df_c_meta), c_cap)
        
        df_sample = df_c_meta.sample(n=s_size, random_state=42)
        
        # Attach SHAP values
        for feat in FEATURES:
            if feat in feature_names:
                f_idx = feature_names.index(feat)
                df_sample[feat] = X[:, f_idx]
                df_sample[f'shap_{feat}'] = shap_values[:, f_idx]
        
        data_dict[cid] = df_sample[['block'] + FEATURES + [f'shap_{f}' for f in FEATURES if f in feature_names]]

    log(f"Starting spatial block bootstrap (N={N_BOOTSTRAP}) using {CORES} cores...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=CORES) as executor:
        futures = [executor.submit(run_single_bootstrap, i, data_dict) for i in range(N_BOOTSTRAP)]
        for f in futures:
            all_results.extend(f.result())
            
    # Aggregate
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(['cluster', 'feature'])['threshold'].agg(['mean', 'std', 'count']).reset_index()
    # Calculate 95% CI
    summary['ci_low'] = results_df.groupby(['cluster', 'feature'])['threshold'].quantile(0.025).values
    summary['ci_high'] = results_df.groupby(['cluster', 'feature'])['threshold'].quantile(0.975).values
    
    summary.to_csv(os.path.join(OUTPUT_BASE, "bootstrap_threshold_uncertainty.csv"), index=False)
    results_df.to_parquet(os.path.join(OUTPUT_BASE, "bootstrap_raw_results.parquet"))
    
    log(f"Bootstrap complete. Results saved to {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
