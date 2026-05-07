import xarray as xr
import numpy as np
import os
import glob
from datetime import datetime
from scipy import stats
import json

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
DATA_PRESENT = f"{ROOT_DIR}/data/presentLU"
DATA_LU = f"{ROOT_DIR}/data/harmonized_LU_classes.zarr"
DATA_CLUSTER = f"{ROOT_DIR}/data/climate_clusters.zarr"
DATA_POP = f"{ROOT_DIR}/data/population"
OUTPUT_MODELS = f"{ROOT_DIR}/data/statistical_models"

FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
POP_BINS = 10 # 0-9

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def fit_beta_point_mass(data, feat=None):
    """Fit a mixture of Zero-Point-Mass, One-Point-Mass (for saturated age), and Beta distribution."""
    # Remove NaNs
    data = data[~np.isnan(data)]
    if len(data) < 20: # Minimum sample size
        return None
    
    n_total = len(data)
    
    # Zero Point Mass
    zero_mask = (data <= 1e-6)
    n_zero = np.sum(zero_mask)
    weight_zero = n_zero / n_total
    
    # One Point Mass (Saturated age)
    # Only relevant for forest_age which is normalized to [0, 1] with 300 as max
    weight_one = 0.0
    if feat == 'forest_age':
        one_mask = (data >= (1.0 - 1e-6))
        n_one = np.sum(one_mask)
        weight_one = n_one / n_total
        nonzero_mask = ~(zero_mask | one_mask)
    else:
        nonzero_mask = ~zero_mask
        
    nonzero_data = data[nonzero_mask]
    
    # Beta distribution is defined on (0, 1)
    if len(nonzero_data) < 10:
        return {
            "weight_zero": float(weight_zero), 
            "weight_one": float(weight_one),
            "beta_params": None,
            "mean": float(np.mean(data)),
            "count": int(n_total)
        }
    
    # Clip to (1e-6, 1-1e-6) for stability
    nonzero_data = np.clip(nonzero_data, 1e-6, 1-1e-6)
    
    try:
        # beta.fit returns (a, b, loc, scale). We fix loc=0, scale=1.
        a, b, loc, scale = stats.beta.fit(nonzero_data, floc=0, fscale=1)
        return {
            "weight_zero": float(weight_zero),
            "weight_one": float(weight_one),
            "beta_params": [float(a), float(b)],
            "mean": float(np.mean(data)),
            "count": int(n_total)
        }
    except:
        return {
            "weight_zero": float(weight_zero), 
            "weight_one": float(weight_one),
            "beta_params": None,
            "mean": float(np.mean(data)),
            "count": int(n_total)
        }

def run_learner():
    log("Starting Learner script...")
    os.makedirs(OUTPUT_MODELS, exist_ok=True)
    
    # 1. Load Present-day Baselines (Conditioning variables)
    log("Loading conditioning variables (2015 baseline)...")
    ds_lu = xr.open_zarr(DATA_LU, group='1991_2020')
    ds_cluster = xr.open_zarr(DATA_CLUSTER, group='1991_2020')
    # Using 2015 as representative year for present-day population
    ds_pop = xr.open_zarr(f"{DATA_POP}/population_histsoc_30arcmin_annual_1901_2014_binned.zarr")
    pop_2015 = ds_pop['pop_bin'].sel(time='2014-01-01', method='nearest').values # Latest available in series
    
    lu_classes = ds_lu['LU_class'].values
    clusters = ds_cluster['cluster'].values
    
    # 2. Load Target Features (Present-day observations)
    obs = {}
    for feat in FEATURES:
        log(f"Loading observation: {feat}")
        if feat == 'gHM':
            ds = xr.open_dataset(f"{DATA_PRESENT}/gHM.nc")
            obs[feat] = ds['ghm'].values
        elif feat == 'forest_age':
            ds = xr.open_dataset(f"{DATA_PRESENT}/forest_age.nc")
            obs[feat] = ds['forest_age'].values
            # Normalize forest age to [0, 1] using actual max (300 in current dataset)
            obs[feat] = np.clip(obs[feat] / 300.0, 0, 1)
        else:
            ds = xr.open_dataset(f"{DATA_PRESENT}/{feat}.nc")
            obs[feat] = ds[feat].values

    # 3. Bin and Fit
    # Combined Bin ID: LU (2 digits) + Cluster (1 digit) + Pop (1 digit)
    # Total bins ~ 7 * 6 * 10 = 420
    
    model_database = {feat: {} for feat in FEATURES}
    
    log("Iterating through bins and fitting distributions...")
    unique_lus = np.unique(lu_classes[lu_classes > 0])
    
    for lu in unique_lus:
        for cluster in range(6):
            for pop in range(10):
                bin_mask = (lu_classes == lu) & (clusters == cluster) & (pop_2015 == pop)
                if not np.any(bin_mask):
                    continue
                
                bin_key = f"{int(lu)}_{int(cluster)}_{int(pop)}"
                
                for feat in FEATURES:
                    sample = obs[feat][bin_mask]
                    model = fit_beta_point_mass(sample, feat=feat)
                    if model:
                        model_database[feat][bin_key] = model
                        
    # 2.5 Load Anthromes for special scenarios
    log("Loading Anthromes for Pre-industrial/Intact baseline...")
    ds_anthrome = xr.open_dataset(f"{ROOT_DIR}/data/anthromes/anthrome_2015.nc")
    anthromes = ds_anthrome['anthrome'].values
    
    unique_anthromes = np.unique(anthromes[anthromes > 0])
    
    # Combined Bin ID for Special: LU + Cluster + Anthrome
    # We will fit these in the same loop
    special_models = {feat: {} for feat in FEATURES}

    log("Iterating through bins and fitting distributions...")
    unique_lus = np.unique(lu_classes[lu_classes > 0])
    
    for lu in unique_lus:
        for cluster in range(6):
            # Standard: Pop-based
            for pop in range(10):
                bin_mask = (lu_classes == lu) & (clusters == cluster) & (pop_2015 == pop)
                if not np.any(bin_mask):
                    continue
                
                bin_key = f"{int(lu)}_{int(cluster)}_{int(pop)}"
                
                for feat in FEATURES:
                    sample = obs[feat][bin_mask]
                    model = fit_beta_point_mass(sample, feat=feat)
                    if model:
                        model_database[feat][bin_key] = model
            
            # Special: Anthrome-based
            for ant in unique_anthromes:
                bin_mask = (lu_classes == lu) & (clusters == cluster) & (anthromes == ant)
                if not np.any(bin_mask):
                    continue
                
                bin_key = f"{int(lu)}_{int(cluster)}_{int(ant)}"
                
                for feat in FEATURES:
                    sample = obs[feat][bin_mask]
                    model = fit_beta_point_mass(sample)
                    if model:
                        special_models[feat][bin_key] = model
                        
    # 4. Save model database
    log(f"Saving model database to {OUTPUT_MODELS}...")
    with open(f"{OUTPUT_MODELS}/beta_mixture_models.json", 'w') as f:
        json.dump(model_database, f, indent=2)
    
    with open(f"{OUTPUT_MODELS}/anthrome_mixture_models.json", 'w') as f:
        json.dump(special_models, f, indent=2)

if __name__ == "__main__":
    run_learner()
