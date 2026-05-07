import xarray as xr
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats
import multiprocessing as mp
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
DATA_LU = f"{ROOT_DIR}/data/harmonized_LU_classes.zarr"
DATA_CLUSTER = f"{ROOT_DIR}/data/climate_clusters.zarr"
DATA_POP = f"{ROOT_DIR}/data/population"
DATA_ANTHROME = f"{ROOT_DIR}/data/anthromes"
DATA_MODELS = f"{ROOT_DIR}/data/statistical_models"
OUTPUT_DIR = f"{ROOT_DIR}/data/LU_features"

FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def sample_from_beta_mixture(model, ranks):
    """Sample values given Beta parameters and percentile ranks."""
    if not model or 'weight_zero' not in model:
        return np.full(ranks.shape, np.nan)
    
    weight_zero = model['weight_zero']
    weight_one = model.get('weight_one', 0.0)
    
    out = np.zeros(ranks.shape)
    
    # Ranks <= weight_zero are assigned 0
    # Ranks > (1 - weight_one) are assigned 1
    # Rest are sampled from Beta
    
    one_threshold = 1.0 - weight_one
    
    one_mask = (ranks > one_threshold)
    out[one_mask] = 1.0
    
    beta_mask = (ranks > weight_zero) & (ranks <= one_threshold)
    
    if np.any(beta_mask) and model['beta_params']:
        a, b = model['beta_params']
        # Rescale rank to (0, 1) for the Beta portion
        # The range is (weight_zero, 1 - weight_one)
        denom = (one_threshold - weight_zero)
        if denom > 1e-6:
            rescaled_ranks = (ranks[beta_mask] - weight_zero) / denom
            # Clip rescaled_ranks for stability before ppf
            rescaled_ranks = np.clip(rescaled_ranks, 1e-6, 1-1e-6)
            out[beta_mask] = stats.beta.ppf(rescaled_ranks, a, b)
        else:
            # If no room between point masses, just use the mean or 0.5
            out[beta_mask] = 0.5
    
    return out

def process_scenario(args):
    scenario_type, period, ssp = args
    log(f"Processing Scenario: {scenario_type} | {period} | {ssp}")
    
    # 1. Load Conditioning Data
    group_path = f"{period}/{ssp}" if ssp else period
    ds_lu = xr.open_zarr(DATA_LU, group=group_path)
    ds_cluster = xr.open_zarr(DATA_CLUSTER, group=group_path)
    
    lu_map = ds_lu['LU_class'].values
    cluster_map = ds_cluster['cluster'].values
    lat, lon = ds_lu.lat, ds_lu.lon
    
    # Load Population or Anthrome
    pop_map = None
    anthrome_map = None
    
    if scenario_type == "standard":
        # Mapping year selection
        if period == "1901_1930": pop_year = 1915
        elif period == "1931_1960": pop_year = 1945
        elif period == "1961_1990": pop_year = 1975
        elif period == "1991_2020": pop_year = 2014 # latest hist
        elif period == "2041_2070": pop_year = 2055
        elif period == "2071_2099": pop_year = 2085
        else: pop_year = 2014

        if pop_year <= 2014:
            ds_pop = xr.open_zarr(f"{DATA_POP}/population_histsoc_30arcmin_annual_1901_2014_binned.zarr")
            pop_map = ds_pop['pop_bin'].sel(time=f"{pop_year}-01-01", method='nearest').values
        else:
            # Map ssp126 -> SSP1, ssp370 -> SSP3
            ssp_prefix = ssp.upper()
            if ssp_prefix.startswith("SSP"):
                ssp_prefix = ssp_prefix[:4] # e.g. SSP3
            
            pop_file = f"{DATA_POP}/{ssp_prefix}_{pop_year}_binned.nc"
            log(f"  Opening population: {pop_file}")
            ds_pop = xr.open_dataset(pop_file)
            pop_map = ds_pop['pop_bin'].values
            
        with open(f"{DATA_MODELS}/beta_mixture_models.json", 'r') as f:
            models = json.load(f)
            
    elif scenario_type in ["intact", "pre-industrial"]:
        ant_name = "anthrome_1700" if scenario_type == "pre-industrial" else "anthrome_2015"
        ds_ant = xr.open_dataset(f"{DATA_ANTHROME}/{ant_name}.nc")
        anthrome_map = ds_ant['anthrome'].values
        
        with open(f"{DATA_MODELS}/anthrome_mixture_models.json", 'r') as f:
            models = json.load(f)
    
    # 2. Estimation
    scenario_results = {}
    
    for feat in FEATURES:
        log(f"  Estimating {feat}...")
        rank_map = xr.open_dataset(f"{DATA_MODELS}/{feat}_rank_map.nc")[f"{feat}_rank"].values
        out_feat = np.full(lu_map.shape, np.nan, dtype=np.float32)
        
        unique_lus = np.unique(lu_map[lu_map > 0])
        for lu in unique_lus:
            for cluster in range(6):
                cond_val_range = range(10) if scenario_type == "standard" else np.unique(anthrome_map[anthrome_map > 0])
                
                for cond_val in cond_val_range:
                    if scenario_type == "standard":
                        mask = (lu_map == lu) & (cluster_map == cluster) & (pop_map == cond_val)
                    else:
                        mask = (lu_map == lu) & (cluster_map == cluster) & (anthrome_map == cond_val)
                    
                    if not np.any(mask):
                        continue
                    
                    bin_key = f"{int(lu)}_{int(cluster)}_{int(cond_val)}"
                    if bin_key in models[feat]:
                        out_feat[mask] = sample_from_beta_mixture(models[feat][bin_key], rank_map[mask])
                    else:
                        # Fallback to mean if bin missing? Or leave NaN.
                        pass
        
        # Post-processing: Forest Age (denormalize)
        if feat == 'forest_age':
            out_feat = out_feat * 300.0
            
        scenario_results[feat] = out_feat

    # 3. Soft Constraints & Saving
    # (irr + rain) <= Crop Margin (LU 22)
    # We allow 10% margin
    crop_mask = (lu_map == 22)
    total_frac = scenario_results['irr_frac'] + scenario_results['rain_frac']
    over_limit = (total_frac > 1.1) & crop_mask
    if np.any(over_limit):
        log(f"  Applying soft constraint to {np.sum(over_limit)} pixels...")
        scale = 1.1 / total_frac[over_limit]
        scenario_results['irr_frac'][over_limit] *= scale
        scenario_results['rain_frac'][over_limit] *= scale

    # Save to NetCDF
    if scenario_type == "intact":
        filename = "intact.nc"
    elif scenario_type == "pre-industrial":
        filename = "pre-industrial.nc"
    else:
        # For standard, we drop the 'standard_' prefix
        filename = f"{period}_{ssp if ssp else 'none'}.nc"
        
    out_path = f"{OUTPUT_DIR}/{filename}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ds_save = xr.Dataset(
        {f: (("lat", "lon"), scenario_results[f]) for f in FEATURES},
        coords={"lat": lat, "lon": lon}
    )
    ds_save.to_netcdf(out_path)
    log(f"  Saved {out_path}")

def main():
    log("Starting Estimation Phase...")
    
    tasks = []
    # Historical Standard
    for p in ['1901_1930', '1931_1960', '1961_1990', '1991_2020']:
        tasks.append(("standard", p, None))
    
    # Future Standard
    for p in ['2041_2070', '2071_2099']:
        for s in ['ssp126', 'ssp245', 'ssp370', 'ssp460']:
            tasks.append(("standard", p, s))
            
    # Special
    # Note: Pre-industrial/Intact usually paired with 1901_1930 climate/clusters?
    # User prompt implies "Pre-industrial" and "Intact" as separate LU scenarios.
    # We'll use 1901_1930 as the climate base for Pre-industrial.
    # We'll use 1991_2020 as the climate base for Intact (as an idealized state of now).
    tasks.append(("pre-industrial", "1901_1930", None))
    tasks.append(("intact", "1991_2020", None))

    # Using multiprocessing
    with mp.Pool(processes=32) as pool:
        pool.map(process_scenario, tasks)

if __name__ == "__main__":
    main()
