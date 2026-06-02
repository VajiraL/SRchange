import os
import numpy as np

# Workaround for pygam/numpy 2.x compatibility
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

# Workaround for scipy 1.15+ compatibility with pygam
import scipy.sparse
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
if not hasattr(scipy.sparse.csc_matrix, 'A'):
    scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())

import pandas as pd

import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from datetime import datetime
from scipy.optimize import curve_fit

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/advanced')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/advanced_thresholds')
SHAP_DATA_DIR = os.path.join(DIR_ROOT, 'output_SR/explanation/data')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Selection of features to analyze (can be expanded)
FEATURES_TO_ANALYZE = [
    'aridity_index', 'gHM', 'forest_age', 'pet_annual', 
    'annual_precip', 'monthly_precip_std', 'mean_annual_tmp'
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Threshold Detection Logic
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
        residuals = y - piecewise_linear(x, *popt)
        r2 = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        return popt, r2
    except:
        return None, 0

def detect_advanced_thresholds(x, y, feature_name, cluster_id):
    # 1. Fit GAM
    gam = LinearGAM(s(0, n_splines=20)).fit(x.reshape(-1, 1), y)
    
    # 2. Get derivatives
    x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_pred = gam.predict(x_grid)
    # Numerical derivative
    dy = np.gradient(y_pred, x_grid.flatten())
    
    # 3. Fit Piecewise (for Hard Threshold check)
    popt_pw, r2_pw = fit_piecewise(x, y)
    
    # 4. Taxonomy and Threshold extraction
    max_dy = np.max(np.abs(dy))
    # Saturation: where derivative drops below 10% of max
    # We look from the point of max absolute derivative
    idx_max_dy = np.argmax(np.abs(dy))
    
    threshold_val = None
    threshold_type = "None"
    
    # Simple logic for taxonomy
    if r2_pw > 0.85:
        threshold_type = "Hard"
        threshold_val = popt_pw[0]
    elif max_dy > 0:
        # Check for sign reversal (Collapse)
        if (np.min(dy) < -0.1 * max_dy) and (np.max(dy) > 0.1 * max_dy):
            threshold_type = "Collapse"
            # Point of zero crossing
            zero_crossings = np.where(np.diff(np.sign(dy)))[0]
            if len(zero_crossings) > 0:
                threshold_val = x_grid[zero_crossings[0]][0]
        else:
            threshold_type = "Soft"
            # Find saturation point
            # Look after the point of maximum change
            tail_dy = dy[idx_max_dy:]
            tail_x = x_grid[idx_max_dy:]
            sat_idx = np.where(np.abs(tail_dy) < 0.1 * max_dy)[0]
            if len(sat_idx) > 0:
                threshold_val = tail_x[sat_idx[0]][0]
    
    return {
        'cluster': cluster_id,
        'feature': feature_name,
        'type': threshold_type,
        'val': threshold_val,
        'r2_pw': r2_pw,
        'max_dy': max_dy,
        'x_grid': x_grid.flatten(),
        'y_pred': y_pred,
        'dy': dy
    }

# ---------------------------
# 3. Main Execution
# ---------------------------
def main():
    results = []
    
    # Identify clusters
    shap_files = sorted([f for f in os.listdir(SHAP_DATA_DIR) if f.startswith("shap_cluster_")])
    
    for f in shap_files:
        cid = int(f.split('_')[-1].split('.')[0])
        log(f"Processing Cluster {cid}...")
        
        data = np.load(os.path.join(SHAP_DATA_DIR, f), allow_pickle=True)
        shap_values = data['shap_values']
        X = data['X']
        feature_names = data['feature_names'].tolist()
        
        for feat in FEATURES_TO_ANALYZE:
            if feat not in feature_names: continue
            
            f_idx = feature_names.index(feat)
            x_raw = X[:, f_idx]
            y_raw = shap_values[:, f_idx]
            
            # Remove NaNs
            mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
            x_raw, y_raw = x_raw[mask], y_raw[mask]
            
            if len(x_raw) < 1000: continue
            
            # Binning for speed and noise reduction
            df_tmp = pd.DataFrame({'x': x_raw, 'y': y_raw})
            df_tmp['bin'] = pd.qcut(df_tmp['x'], 100, duplicates='drop')
            binned = df_tmp.groupby('bin', observed=True).mean()
            x_bin = binned['x'].values
            y_bin = binned['y'].values
            
            res = detect_advanced_thresholds(x_bin, y_bin, feat, cid)
            results.append(res)
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(x_bin, y_bin, color='gray', alpha=0.3, label='Binned Data')
            plt.plot(res['x_grid'], res['y_pred'], 'r-', label='GAM fit')
            if res['val'] is not None:
                plt.axvline(res['val'], color='blue', linestyle='--', 
                           label=f"Threshold ({res['type']}): {res['val']:.3f}")
            
            plt.title(f"Cluster {cid} - {feat} ({res['type']})")
            plt.xlabel(feat)
            plt.ylabel("SHAP Value")
            plt.legend()
            
            plt.savefig(os.path.join(PLOT_DIR, f"c{cid}_{feat}.png"), dpi=300)
            plt.close()

    # Save summary
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['x_grid', 'y_pred', 'dy']} for r in results])
    summary_df.to_csv(os.path.join(OUTPUT_BASE, "advanced_thresholds_summary.csv"), index=False)
    log(f"Phase 5 complete. Summary saved to {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
