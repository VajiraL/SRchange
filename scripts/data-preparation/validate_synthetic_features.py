import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
DATA_PRESENT = f"{ROOT_DIR}/data/presentLU"
DATA_SYNTHETIC = f"{ROOT_DIR}/data/scenario_features/standard_1991_2020_none.nc"
OUTPUT_VAL = f"{ROOT_DIR}/notes/validation_1991_2020"

FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def run_validation():
    log("Starting validation of 1991-2020 synthetic features...")
    os.makedirs(OUTPUT_VAL, exist_ok=True)
    
    ds_syn = xr.open_dataset(DATA_SYNTHETIC)
    
    stats_summary = {}

    for feat in FEATURES:
        log(f"Comparing {feat}...")
        
        # Load Observation
        if feat == 'gHM':
            ds_obs = xr.open_dataset(f"{DATA_PRESENT}/gHM.nc")
            obs = ds_obs['ghm'].values
        elif feat == 'forest_age':
            ds_obs = xr.open_dataset(f"{DATA_PRESENT}/forest_age.nc")
            obs = ds_obs['forest_age'].values
        else:
            ds_obs = xr.open_dataset(f"{DATA_PRESENT}/{feat}.nc")
            obs = ds_obs[feat].values
            
        syn = ds_syn[feat].values
        
        # Valid mask (exclude NaNs from both)
        mask = ~np.isnan(obs) & ~np.isnan(syn)
        o_v = obs[mask]
        s_v = syn[mask]
        
        # Metrics
        bias = np.mean(s_v - o_v)
        rmse = np.sqrt(np.mean((s_v - o_v)**2))
        corr = np.corrcoef(o_v, s_v)[0, 1]
        
        stats_summary[feat] = {
            "Mean Obs": float(np.mean(o_v)),
            "Mean Syn": float(np.mean(s_v)),
            "Bias": float(bias),
            "RMSE": float(rmse),
            "Correlation": float(corr)
        }
        
        log(f"  {feat} -> Corr: {corr:.4f}, Bias: {bias:.4f}, RMSE: {rmse:.4f}")

    # Save summary
    with open(f"{OUTPUT_VAL}/summary_stats.json", 'w') as f:
        import json
        json.dump(stats_summary, f, indent=2)
    
    log(f"Validation complete. Stats saved to {OUTPUT_VAL}/summary_stats.json")

if __name__ == "__main__":
    run_validation()
