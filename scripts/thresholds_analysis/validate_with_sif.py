import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SIF_PATH = os.path.join(DIR_ROOT, 'data/LHSIF_0p05_monthly.zarr')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLUSTER_MAP = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
PROB_DIR = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/scenarios_probabilistic')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/validation')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/validation')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Validation Logic
# ---------------------------
def main():
    log("Loading datasets...")
    # 1. Load SIF (Observed proxy for ecosystem function)
    sif_ds = xr.open_zarr(SIF_PATH)['sif']
    sif_mean = sif_ds.mean('time').load()
    sif_std = sif_ds.std('time').load()
    sif_cv = (sif_std / sif_mean).load() # Coefficient of Variation as proxy for stability/stress
    
    # 2. Load Probability Maps for 1991-2020
    prob_files = glob.glob(os.path.join(PROB_DIR, "*1991_2020.nc"))
    
    results = []
    
    for pf in prob_files:
        fname = os.path.basename(pf)
        # Extract metadata from filename
        # prob_sat_c{cid}_{f1}_{f2}_1991_2020.nc
        parts = fname.replace(".nc", "").split("_")
        cid = int(parts[2][1:]) if parts[2].startswith('c') else int(parts[2])
        f1 = parts[3] + "_" + parts[4]
        f2 = parts[5] + "_" + parts[6] if len(parts) > 8 else parts[5]
        
        log(f"Validating Cluster {cid}: {f1} x {f2}")
        
        prob_ds = xr.open_dataset(pf)
        var_name = list(prob_ds.data_vars)[0]
        prob = prob_ds[var_name]
        
        # Align SIF metrics with probability map
        # reindex_like works if coords match, otherwise use interpn or nearest
        s_cv_aligned = sif_cv.reindex_like(prob, method='nearest')
        
        mask = ~np.isnan(prob.values) & ~np.isnan(s_cv_aligned.values)
        
        p_val = prob.values[mask]
        s_val = s_cv_aligned.values[mask]
        
        if len(p_val) < 1000: continue
        
        # Binned analysis
        df_bin = pd.DataFrame({'prob_sat': p_val, 'sif_cv': s_val})
        bins = np.linspace(0, 1, 11)
        df_bin['bin'] = pd.cut(df_bin['prob_sat'], bins=bins)
        summary = df_bin.groupby('bin', observed=True)['sif_cv'].agg(['mean', 'std', 'count']).reset_index()
        summary['cluster'] = cid
        summary['interaction'] = f"{f1}_{f2}"
        
        # Get bin centers for plotting
        summary['bin_center'] = summary['bin'].apply(lambda x: x.mid).astype(float)
        
        results.append(summary)
        
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.errorbar(summary['bin_center'], summary['mean'], yerr=summary['std'], fmt='o-', capsize=5)
        plt.xlabel("Probability of Saturation (Model Inferred)")
        plt.ylabel("SIF CV (Observed Stress Proxy)")
        plt.title(f"Validation: Cluster {cid} ({f1} x {f2})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, f"val_c{cid}_{f1}_{f2}.png"), dpi=300)
        plt.close()

    if results:
        final_df = pd.concat(results)
        final_df.to_csv(os.path.join(OUTPUT_BASE, "sif_validation_summary.csv"), index=False)
        log("Validation complete.")
    else:
        log("No validation results generated.")

if __name__ == "__main__":
    main()
