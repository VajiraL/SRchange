import os
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
LU_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
BOOTSTRAP_DIR = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/advanced/bootstrap_interactions')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/scenarios_probabilistic')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/scenarios_probabilistic')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SCENARIOS = {
    "1901_1930": {"clim_group": "1901_1930", "lu": "1901_1930.nc"},
    "1991_2020": {"clim_group": "1991_2020", "lu": "1991_2020.nc"},
    "2071_2099_ssp126": {"clim_group": "2071_2099/ssp126", "lu": "2071_2099_ssp126.nc"},
    "2071_2099_ssp370": {"clim_group": "2071_2099/ssp370", "lu": "2071_2099_ssp370.nc"}
}

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
# 2. Logic
# ---------------------------
def extract_threshold_contours(boot_ensemble_path):
    data = np.load(boot_ensemble_path)
    x1_grid = data['x1_grid']
    x2_grid = data['x2_grid']
    dy_all = data['dy_all'] # Shape: (N, len(x2), len(x1))
    
    n_boot = dy_all.shape[0]
    threshold_surfaces = [] # List of functions f(x2) -> x1*
    
    for i in range(n_boot):
        dy = dy_all[i]
        # For each x2, find where dy_df1 becomes small
        x1_star = []
        for j in range(len(x2_grid)):
            dy_j = dy[j, :]
            max_dy = np.max(np.abs(dy_j))
            idx_max = np.argmax(np.abs(dy_j))
            
            tail_dy = dy_j[idx_max:]
            tail_x1 = x1_grid[idx_max:]
            
            sat_idx = np.where(np.abs(tail_dy) < 0.1 * max_dy)[0]
            if len(sat_idx) > 0:
                x1_star.append(tail_x1[sat_idx[0]])
            else:
                x1_star.append(tail_x1[-1])
        
        # Create interpolator for this bootstrap iteration
        threshold_surfaces.append(interp1d(x2_grid, x1_star, bounds_error=False, fill_value="extrapolate"))
        
    return threshold_surfaces

def main():
    log("Loading bootstrap threshold surfaces...")
    surfaces = {}
    for cid, f1, f2 in INTERACTIONS:
        p = os.path.join(BOOTSTRAP_DIR, f"boot_surface_c{cid}_{f1}_{f2}.npz")
        if os.path.exists(p):
            surfaces[(cid, f1, f2)] = extract_threshold_contours(p)
            log(f"  Loaded surfaces for Cluster {cid}: {f1} x {f2}")

    log("Loading cluster map...")
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']

    for scn_name, paths in SCENARIOS.items():
        log(f"Processing Scenario: {scn_name}...")
        
        # Load necessary features for this scenario
        features = {}
        # We need aridity, gHM, forest_age
        try:
            clim = xr.open_zarr(CLIMATE_ZARR, group=paths['clim_group'])
            features['aridity_index'] = clim['aridity_index'].sel(lat=cl_ds.lat, lon=cl_ds.lon, method='nearest')
            
            lu = xr.open_dataset(os.path.join(LU_FEATURES_DIR, paths['lu']))
            features['gHM'] = lu['gHM'].sel(lat=cl_ds.lat, lon=cl_ds.lon, method='nearest')
            features['forest_age'] = lu['forest_age'].sel(lat=cl_ds.lat, lon=cl_ds.lon, method='nearest')
        except Exception as e:
            log(f"  Error loading data for {scn_name}: {e}")
            continue

        # For each interaction, calculate probability map
        for (cid, f1, f2), surface_fns in surfaces.items():
            log(f"  Calculating Prob(Saturated) for Cluster {cid}: {f1} x {f2}")
            
            # Mask for this cluster
            mask = (cl_ds == cid)
            if not mask.any(): continue
            
            x1_data = features[f1].values
            x2_data = features[f2].values
            
            # Flatten for speed
            m_flat = mask.values.flatten()
            x1_flat = x1_data.flatten()[m_flat]
            x2_flat = x2_data.flatten()[m_flat]
            
            # Probability calculation
            # For each bootstrap iteration i, is x1 > x1*(x2) ?
            saturated_count = np.zeros_like(x1_flat)
            for f_interp in surface_fns:
                x1_star = f_interp(x2_flat)
                saturated_count += (x1_flat > x1_star).astype(int)
            
            prob_flat = saturated_count / len(surface_fns)
            
            # Put back into map
            prob_map = xr.full_like(cl_ds, np.nan, dtype=float)
            data_full = prob_map.values.flatten()
            data_full[m_flat] = prob_flat
            prob_map.values = data_full.reshape(prob_map.shape)
            
            # Save
            prob_map.name = f"prob_sat_c{cid}_{f1}_{f2}_{scn_name}"
            prob_map.to_netcdf(os.path.join(OUTPUT_BASE, f"{prob_map.name}.nc"))

    log("Probabilistic regime mapping complete.")

if __name__ == "__main__":
    main()
