import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# Workarounds for pygam/numpy 2.x
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool
import scipy.sparse
# More robust fix for 'A' attribute in all sparse matrix types
if not hasattr(scipy.sparse.spmatrix, 'A'):
    scipy.sparse.spmatrix.A = property(lambda self: self.toarray())

from pygam import LinearGAM, te, s

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SHAP_DATA_DIR = os.path.join(DIR_ROOT, 'output_SR/explanation/data')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/advanced/interactions')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/interaction_surfaces')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

INTERACTIONS = [
    (0, 'aridity_index', 'gHM'),
    (0, 'aridity_index', 'forest_age'),
    (3, 'aridity_index', 'gHM'),
    (5, 'aridity_index', 'gHM'),
    (2, 'aridity_index', 'forest_age'),
    (4, 'aridity_index', 'forest_age')
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Interaction Modeling
# ---------------------------
def fit_interaction_surface(cid, f1, f2):
    log(f"Modeling Interaction: Cluster {cid}, {f1} x {f2}")
    
    file_path = os.path.join(SHAP_DATA_DIR, f"shap_cluster_{cid}.npz")
    if not os.path.exists(file_path):
        log(f"  Data file not found: {file_path}")
        return None
    
    data = np.load(file_path, allow_pickle=True)
    shap_values = data['shap_values']
    X = data['X']
    feature_names = data['feature_names'].tolist()
    
    if f1 not in feature_names or f2 not in feature_names:
        log(f"  Features not found in Cluster {cid}")
        return None
        
    idx1 = feature_names.index(f1)
    idx2 = feature_names.index(f2)
    
    x1 = X[:, idx1]
    x2 = X[:, idx2]
    # We want to model the SHAP value of f1 as a function of f1 and f2
    y = shap_values[:, idx1]
    
    mask = ~np.isnan(x1) & ~np.isnan(x2) & ~np.isnan(y)
    x1, x2, y = x1[mask], x2[mask], y[mask]
    
    if len(x1) < 5000:
        log(f"  Insufficient data points ({len(x1)})")
        return None
        
    # Sampling for speed
    if len(x1) > 50000:
        idx = np.random.choice(len(x1), 50000, replace=False)
        x1, x2, y = x1[idx], x2[idx], y[idx]
    
    # Fit Tensor-Product GAM
    # te(0, 1) means interaction between feature 0 and feature 1
    gam = LinearGAM(te(0, 1, n_splines=[10, 10])).fit(np.column_stack([x1, x2]), y)
    
    # Create grid for prediction
    x1_grid = np.linspace(x1.min(), x1.max(), 50)
    x2_grid = np.linspace(x2.min(), x2.max(), 50)
    XX1, XX2 = np.meshgrid(x1_grid, x2_grid)
    YY = gam.predict(np.column_stack([XX1.flatten(), XX2.flatten()])).reshape(XX1.shape)
    
    # Calculate threshold contour (where derivative with respect to f1 is small)
    # Numerical derivative along axis 1 (f1)
    dy_df1 = np.gradient(YY, x1_grid, axis=1)
    
    return {
        'cid': cid, 'f1': f1, 'f2': f2,
        'x1_grid': x1_grid, 'x2_grid': x2_grid,
        'YY': YY, 'dy_df1': dy_df1
    }

def plot_surface(res):
    cid, f1, f2 = res['cid'], res['f1'], res['f2']
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: SHAP Surface
    plt.subplot(1, 2, 1)
    cp = plt.contourf(res['x1_grid'], res['x2_grid'], res['YY'], cmap='viridis', levels=20)
    plt.colorbar(cp, label=f'SHAP({f1})')
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f"Cluster {cid}: SHAP Surface")
    
    # Plot 2: Derivative (Threshold detection)
    plt.subplot(1, 2, 2)
    # We find where the derivative is low (saturation)
    max_dy = np.max(np.abs(res['dy_df1']))
    # Draw a contour where derivative is 10% of max
    plt.contourf(res['x1_grid'], res['x2_grid'], np.abs(res['dy_df1']), cmap='magma', levels=20)
    plt.colorbar(label=f'|dSHAP/d{f1}|')
    
    # Attempt to draw the threshold contour line
    plt.contour(res['x1_grid'], res['x2_grid'], np.abs(res['dy_df1']), 
                levels=[0.1 * max_dy], colors='white', linestyles='--')
    
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f"Threshold Contour (10% of max slope)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"surface_c{cid}_{f1}_{f2}.png"), dpi=300)
    plt.close()

def main():
    all_results = []
    for cid, f1, f2 in INTERACTIONS:
        res = fit_interaction_surface(cid, f1, f2)
        if res:
            plot_surface(res)
            # Save raw grid for Phase 8
            save_name = f"surface_c{cid}_{f1}_{f2}.npz"
            np.savez(os.path.join(OUTPUT_BASE, save_name), **res)
            
    log("Phase 6 complete. Surfaces saved.")

if __name__ == "__main__":
    main()
