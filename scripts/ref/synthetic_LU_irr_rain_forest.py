import xarray as xr
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
import dask.array as da
from dask.distributed import Client

# -------------------------------------------------------------------------
# 1. SETUP & UTILITIES
# -------------------------------------------------------------------------

def setup_cluster():
    """Initialize Dask client to utilize the 364 cores."""
    # With 2.3TB RAM, we can be generous with memory_limit per worker
    client = Client(n_workers=64, threads_per_worker=4, memory_limit='30GB')
    print(f"Dask Dashboard: {client.dashboard_link}")
    return client

class BetaMixtureModel:
    """
    Fits a Beta distribution mixed with a point mass at 0.
    """
    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.weight_zero_ = None
        self.weight_nonzero_ = None

    def fit(self, X):
        # Remove NaNs
        X = X[~np.isnan(X)]
        if len(X) == 0:
            return
            
        X_zero = X[X == 0]
        X_nonzero = X[X > 0]
        
        self.weight_zero_ = len(X_zero) / len(X)
        self.weight_nonzero_ = 1 - self.weight_zero_
        
        # Fit Beta to nonzero (assuming 0-1 range)
        # We enforce floc=0, fscale=1 to keep it strictly 0-1
        if len(X_nonzero) > 10:
            try:
                self.a_, self.b_, _, _ = stats.beta.fit(X_nonzero, floc=0, fscale=1)
            except:
                # Fallback if fit fails
                self.a_, self.b_ = 1, 1
        else:
            self.a_, self.b_ = 1, 1

    def cdf(self, x):
        """Calculate Cumulative Density (Rank)"""
        # P(X <= x) = P(0) + P(NonZero) * BetaCDF(x)
        if self.a_ is None: return np.nan
        
        # Vectorized handling
        res = np.zeros_like(x, dtype=float)
        
        # If x < 0 (shouldn't happen in data), rank is 0
        # If x == 0, rank is weight_zero
        mask_zero = (x <= 0)
        mask_nonzero = (x > 0)
        
        res[mask_zero] = self.weight_zero_
        
        if np.any(mask_nonzero):
            b_cdf = stats.beta.cdf(x[mask_nonzero], self.a_, self.b_, loc=0, scale=1)
            res[mask_nonzero] = self.weight_zero_ + (self.weight_nonzero_ * b_cdf)
            
        return res

    def ppf(self, q):
        """Inverse CDF (Rank to Value)"""
        if self.a_ is None: return np.nan
        
        res = np.zeros_like(q, dtype=float)
        
        # If rank <= weight_zero, value is 0
        mask_zero = (q <= self.weight_zero_)
        mask_nonzero = (q > self.weight_zero_)
        
        res[mask_zero] = 0.0
        
        if np.any(mask_nonzero):
            # Adjust q to be relative to the nonzero portion
            # q_adj = (q - w_0) / w_nonzero
            q_adj = (q[mask_nonzero] - self.weight_zero_) / self.weight_nonzero_
            # Clip for numerical stability
            q_adj = np.clip(q_adj, 0, 1)
            res[mask_nonzero] = stats.beta.ppf(q_adj, self.a_, self.b_, loc=0, scale=1)
            
        return res

# -------------------------------------------------------------------------
# 2. DATA LOADING
# -------------------------------------------------------------------------

def load_data():
    # reference grid for interpolation
    sr = xr.open_dataarray("data/target_sr.nc")
    # Load actual lu
    lu_actual = xr.open_dataset("data/lu_actual.nc").interp_like(sr, method='nearest')
    # Load HILDA
    hilda = xr.open_dataset('data/hildaplus_states.nc', chunks='auto')
    
    # Load Anthromes and Climate Clusters (Placeholders - assuming you have these opened)
    # Ensure they are on the same grid as lu_actual
    # Load anthromes data for each year
    ANTHROME_YEARS = [1700, 1910, 1940, 1970, 2015]
    ANTHROME_BASE_PATH = 'data/anthromes_12K_full'
    anthromes = {}
    for year in ANTHROME_YEARS:
        path = f'{ANTHROME_BASE_PATH}/anthromes{year}AD.asc'
        da = xr.open_dataarray(path, engine='rasterio')
        da = da.drop_vars(['band', 'spatial_ref'], errors='ignore')
        da = da.rename({'x': 'lon', 'y': 'lat'})
        da = da.interp_like(sr, method='nearest').squeeze()
        da = da.rename(f'anthrome{year}')
        da = da.where(da != 70)  # Set no data value (70) to NaN
        anthromes[year] = da

    clim_k5 = xr.open_dataarray('output/sqrt_weight_ensemble/cluster_maps/cluster_k5.nc') # .interp_like(sr, method='nearest')
    clim_k5 = clim_k5.where(clim_k5 != -1)

    return lu_actual, hilda, clim_k5, anthromes

# -------------------------------------------------------------------------
# 3. IRRIGATION & RAINFED SIMULATION (The Statistical Approach)
# -------------------------------------------------------------------------

def simulate_fraction_variable(
    target_var_name, 
    lu_ds, 
    anthrome_clim_current, 
    anthrome_clim_past_dict,
    monotonic_constraint=False
):
    """
    target_var_name: 'irr_frac' or 'rain_frac'
    lu_ds: dataset containing the current variable
    anthrome_clim_current: DataArray of Combined Class ID for present
    anthrome_clim_past_dict: Dict {year: DataArray of Combined Class ID}
    monotonic_constraint: If True, ensures val_1920 <= val_1950 <= val_2020
    """
    
    print(f"Processing {target_var_name}...")
    
    # Get current values
    current_vals = lu_ds[target_var_name].load() # Load to RAM for speed
    
    # Identify all unique combined classes
    unique_classes = np.unique(anthrome_clim_current.values)
    unique_classes = unique_classes[~np.isnan(unique_classes)]
    
    # 1. Fit Models and Calculate Current Ranks
    models = {}
    rank_map_current = xr.full_like(current_vals, np.nan)
    
    # Convert to numpy for fast loop
    np_current = current_vals.values
    np_class_curr = anthrome_clim_current.values
    np_rank_curr = np.full(np_current.shape, np.nan, dtype=np.float32)

    for cls in unique_classes:
        mask = (np_class_curr == cls) & (~np.isnan(np_current))
        data_in_class = np_current[mask]
        
        if len(data_in_class) < 5: continue
        
        # Fit Model
        model = BetaMixtureModel()
        model.fit(data_in_class)
        models[cls] = model
        
        # Calculate Rank (Intrinsic Rank)
        np_rank_curr[mask] = model.cdf(data_in_class)

    # 2. Simulate Past Years
    simulated_maps = {}
    
    # Sort years descending to handle monotonicity constraint easily
    years = sorted(anthrome_clim_past_dict.keys(), reverse=True)
    
    # Start constraint with current values
    constraint_barrier = np_current.copy()
    
    for year in years:
        print(f"  Simulating year {year}...")
        past_class_map = anthrome_clim_past_dict[year].values
        sim_layer = np.full(np_current.shape, np.nan, dtype=np.float32)
        
        unique_past_classes = np.unique(past_class_map)
        unique_past_classes = unique_past_classes[~np.isnan(unique_past_classes)]
        
        for cls in unique_past_classes:
            if cls not in models: 
                # If a class exists in past but not present, fallback? 
                # Option: Use nearest neighbor class or generic fit. 
                # For now, skip (nan)
                continue
                
            model = models[cls]
            mask = (past_class_map == cls)
            
            # Use the rank from the *current* spatial configuration
            # This assumes spatial autocorrelation of 'intensity' persists
            ranks_at_loc = np_rank_curr[mask]
            
            # Where ranks are missing (e.g., current was NaN), generate random rank?
            # Better: fill nan ranks with 0.5 or random. Let's keep nan.
            valid_ranks = ~np.isnan(ranks_at_loc)
            
            if np.any(valid_ranks):
                pred_vals = model.ppf(ranks_at_loc[valid_ranks])
                
                # Assign back
                # Need careful indexing here
                flat_indices = np.where(mask.ravel())[0]
                valid_indices = flat_indices[valid_ranks]
                
                sim_layer.ravel()[valid_indices] = pred_vals

        # 3. Apply Monotonicity Constraint (if requested)
        if monotonic_constraint:
            # Past value cannot be higher than future value (constraint_barrier)
            # Handle NaNs: np.fmin ignores NaNs, but we want to keep NaNs if data is missing
            # Logic: If constraint is valid and sim is valid, take min.
            
            valid_mask = (~np.isnan(sim_layer)) & (~np.isnan(constraint_barrier))
            sim_layer[valid_mask] = np.minimum(
                sim_layer[valid_mask], 
                constraint_barrier[valid_mask]
            )
            
            # Update barrier for next older year
            # We use the simulated value as the new ceiling for the older year
            constraint_barrier = sim_layer
            
        simulated_maps[year] = sim_layer

    return simulated_maps

# -------------------------------------------------------------------------
# 4. FOREST AGE RECONSTRUCTION (HILDA+)
# -------------------------------------------------------------------------
'''
def process_forest_age(hilda_ds, lu_obs_ds, target_years=[1920, 1950, 1980, 2010]):
    """
    1. Regrids HILDA to target grid.
    2. Calculates continuous forest age from HILDA states.
    3. Bias corrects using observed forest age (2010).
    """
    print("Processing Forest Age...")
    
    # 1. Regrid HILDA to Analysis Grid
    # Using interp_like nearest as requested.
    # Note: With 2.3TB RAM, we can load the relevant subset of HILDA first to speed up interp.
    print("  Regridding HILDA (this may take time)...")
    
    # Optimization: Only load forest classes? No, we need full state history.
    hilda_regridded = hilda_ds['LULC_states'].interp_like(lu_obs_ds, method='nearest')
    
    # Load into memory (RAM is plentiful) to make the loop lightning fast
    hilda_cube = hilda_regridded.load() 
    
    # Forest Codes in HILDA: 40 (Unknown), 41, 42, 43, 44, 45
    forest_mask = (hilda_cube >= 40) & (hilda_cube <= 45)
    
    # 2. Calculate "HILDA Age" (Consecutive years as forest)
    # We create a 3D array of ages. 
    # Algorithm: Iterate forward. If forest, age++, else age=0.
    
    times = hilda_cube.time.values
    ni, nj = hilda_cube.shape[1], hilda_cube.shape[2]
    
    # Initialize age counter
    current_age = np.zeros((ni, nj), dtype=np.float32)
    
    # Store snapshots
    hilda_age_snapshots = {}
    
    print("  Calculating continuous forest history...")
    for t_idx, t_val in enumerate(times):
        is_forest_now = forest_mask.isel(time=t_idx).values
        
        # Increment age where forest, reset to 0 where not
        current_age[is_forest_now] += 1
        current_age[~is_forest_now] = 0
        
        # Check if this is a target year
        yr = int(t_val)
        if yr in target_years:
            hilda_age_snapshots[yr] = current_age.copy()
            
    # 3. Bias Correction / Harmonization
    # Formula: Final_Age_T = HILDA_Age_T + (Obs_Age_2010 - HILDA_Age_2010)
    # But only apply if pixel is forest at time T.
    
    obs_age_2010 = lu_obs_ds['forest_age'].values
    
    # Calculate Baseline Offset
    # If HILDA says age is 10, but Obs says 100, Offset is 90.
    # If HILDA says age is 0 (not forest), Offset is undefined/nan.
    
    hilda_2010 = hilda_age_snapshots[2010]
    
    # Mask where HILDA 2010 is 0 (not forest according to HILDA)
    valid_hilda_2010 = (hilda_2010 > 0)
    
    baseline_offset = np.full_like(hilda_2010, np.nan)
    baseline_offset[valid_hilda_2010] = obs_age_2010[valid_hilda_2010] - hilda_2010[valid_hilda_2010]
    
    # If Obs is NaN (no forest), offset is NaN.
    # If Obs < HILDA (HILDA overestimates age), offset is negative.
    
    final_ages = {}
    for yr in target_years:
        if yr == 2010:
            final_ages[yr] = obs_age_2010 # Trust observed for present
            continue
            
        h_age = hilda_age_snapshots[yr]
        
        # Apply offset
        corrected_age = h_age + baseline_offset
        
        # Logic Fixes:
        # 1. If pixel is not forest in HILDA at year T, Age should be Null (or 0)
        not_forest_at_t = (h_age == 0)
        corrected_age[not_forest_at_t] = np.nan 
        
        # 2. If pixel WAS forest at T, but NOT forest in 2010, we have no baseline offset.
        # Fallback: Use raw HILDA age, or set to max (300) if it was older than 1899?
        # Safe bet: Use Raw HILDA Age + an assumed offset? 
        # Or just Raw HILDA Age (conservative, assumes young forest).
        # Let's use Raw HILDA Age where offset is NaN.
        mask_no_offset = np.isnan(baseline_offset) & (~not_forest_at_t)
        corrected_age[mask_no_offset] = h_age[mask_no_offset]
        
        # 3. Cap max age (e.g., 300) and min age (0)
        corrected_age = np.clip(corrected_age, 0, 300)
        
        final_ages[yr] = corrected_age

    return final_ages
'''

def process_forest_age(hilda_ds, lu_obs_ds, climate_clusters, target_years=[1920, 1950, 1980, 2010]):
    """
    1. Regrids HILDA to target grid.
    2. Calculates continuous forest age from HILDA states.
    3. Bias corrects using observed forest age (2010).
    4. Fills 'Ghost Forests' (no present observation) using the median offset 
       of their Climate Cluster.
    """
    print("Processing Forest Age (with Ghost Forest handling)...")
    
    # 1. Regrid HILDA to Analysis Grid
    print("  Regridding HILDA...")
    # Load into memory (High RAM available)
    hilda_cube = hilda_ds['LULC_states'].interp_like(lu_obs_ds, method='nearest').load()
    
    # Forest Codes in HILDA: 40-45
    forest_mask = (hilda_cube >= 40) & (hilda_cube <= 45)
    
    # 2. Calculate "HILDA Age" (Years since 1899 or last disturbance)
    times = hilda_cube.time.values
    ni, nj = hilda_cube.shape[1], hilda_cube.shape[2]
    
    current_age = np.zeros((ni, nj), dtype=np.float32)
    hilda_age_snapshots = {}
    
    print("  Calculating continuous forest history...")
    for t_idx, t_val in enumerate(times):
        is_forest_now = forest_mask.isel(time=t_idx).values
        current_age[is_forest_now] += 1
        current_age[~is_forest_now] = 0 # Reset if forest is lost
        
        yr = int(t_val)
        if yr in target_years:
            hilda_age_snapshots[yr] = current_age.copy()
            
    # 3. Calculate Baseline Offset (The "Pre-1899 Age")
    # Offset = Observed_2010 - HILDA_Age_2010
    
    obs_age_2010 = lu_obs_ds['forest_age'].values
    hilda_2010 = hilda_age_snapshots[2010]
    
    # Calculate raw offset where we have both valid Obs and HILDA forest
    valid_comparison = (hilda_2010 > 0) & (~np.isnan(obs_age_2010))
    
    raw_offset = np.full_like(hilda_2010, np.nan)
    raw_offset[valid_comparison] = obs_age_2010[valid_comparison] - hilda_2010[valid_comparison]
    
    # 4. Fill Ghost Forests using Climate Cluster Median
    print("  Calculating regional offsets for Ghost Forests...")
    
    # Load climate clusters into numpy
    clim_cluster_map = climate_clusters.values 
    unique_clusters = np.unique(clim_cluster_map)
    unique_clusters = unique_clusters[~np.isnan(unique_clusters)]
    
    # Create a full map of offsets, filling NaNs with cluster medians
    filled_offset_map = raw_offset.copy()
    
    for k in unique_clusters:
        # Mask for this climate cluster
        cluster_mask = (clim_cluster_map == k)
        
        # Get all valid offsets in this cluster
        valid_offsets_in_cluster = raw_offset[cluster_mask & (~np.isnan(raw_offset))]
        
        if len(valid_offsets_in_cluster) > 0:
            # Use Median to avoid skew from outliers
            cluster_median = np.median(valid_offsets_in_cluster)
            
            # Identify pixels in this cluster that need an offset (NaNs)
            # but ONLY if they were actually forests in the past (handled in loop below)
            # Here we just fill the map entirely for the cluster
            target_fill_mask = cluster_mask & (np.isnan(filled_offset_map))
            filled_offset_map[target_fill_mask] = cluster_median
        else:
            # Fallback if cluster has no forests (unlikely): Global Median
            global_median = np.nanmedian(raw_offset)
            target_fill_mask = cluster_mask & (np.isnan(filled_offset_map))
            filled_offset_map[target_fill_mask] = global_median

    # 5. Apply to Past Years
    final_ages = {}
    print("  Reconstructing historical ages...")
    
    for yr in target_years:
        if yr == 2010:
            # For present, stick to observations, filling gaps with HILDA+Offset
            # or just strict observation? Usually better to keep observation where it exists.
            combined_2010 = obs_age_2010.copy()
            # If observation is missing but HILDA says forest, use reconstruction
            missing_obs = np.isnan(combined_2010) & (hilda_age_snapshots[2010] > 0)
            combined_2010[missing_obs] = hilda_age_snapshots[2010][missing_obs] + filled_offset_map[missing_obs]
            final_ages[yr] = combined_2010
            continue
            
        h_age = hilda_age_snapshots[yr]
        
        # Calculate: Historical Age = HILDA_Age_at_T + Pre_1899_Offset
        rec_age = h_age + filled_offset_map
        
        # Masking Logic:
        # If HILDA says it wasn't forest at time T, age is NaN
        not_forest_at_t = (h_age == 0)
        rec_age[not_forest_at_t] = np.nan
        
        # Clip to realistic bounds
        rec_age = np.clip(rec_age, 0, 300)
        
        final_ages[yr] = rec_age

    return final_ages

# -------------------------------------------------------------------------
# 5. MAIN EXECUTION BLOCK
# -------------------------------------------------------------------------

if __name__ == "__main__":
    client = setup_cluster()
    
    lu_actual, hilda, clim_k5, anthromes = load_data()

    clim_k5_2020 = clim_k5.sel(scenario='2020a')
    clim_k5_1990 = clim_k5.sel(scenario='1990a')
    clim_k5_1960 = clim_k5.sel(scenario='1960a')
    clim_k5_1930 = clim_k5.sel(scenario='1930a')
    
    anthrome1700 = anthromes[1700]
    anthrome1910 = anthromes[1910]
    anthrome1940 = anthromes[1940]
    anthrome1970 = anthromes[1970]
    anthrome2015 = anthromes[2015]

    # --- Example Usage for Rainfed/Irrigated ---
    # Construct combined class maps (Climate * 100 + Anthrome)
    # You need to load your anthrome/climate maps here
    # clim_k5 = xr.open_dataarray(...)
    # anth_2015 = xr.open_dataarray(...)
    # anth_1910 = xr.open_dataarray(...)
    
    comb_2015 = clim_k5_2020 * 100 + anthrome2015
    comb_1910 = clim_k5_1930 * 100 + anthrome1910 
    comb_1940 = clim_k5_1960 * 100 + anthrome1940
    comb_1970 = clim_k5_1990 * 100 + anthrome1970

    past_dict = {1910: comb_1910,
                 1940: comb_1940,
                 1970: comb_1970,
                 2010: comb_2015}
    
    # 1. Rainfed (No Monotonic constraint)
    rain_maps = simulate_fraction_variable('rain_frac', lu_actual, comb_2015, past_dict, monotonic_constraint=False)
    
    # 2. Irrigated (Monotonic constraint: Past <= Present)
    irr_maps = simulate_fraction_variable('irr_frac', lu_actual, comb_2015, past_dict, monotonic_constraint=True)
    
    # save to netcdf
    for yr in past_dict.keys():
        lu_actual[f'rain_frac_{yr}'] = (('lat', 'lon'), rain_maps[yr])
        lu_actual[f'irr_frac_{yr}'] = (('lat', 'lon'), irr_maps[yr])
        lu_actual.to_netcdf("data/synthetic_lu/lu_rain_irr.nc")
    # Assuming you have loaded:
    # hilda (the dataset)
    # lu_actual (containing 'forest_age' and coordinates)
    # clim_k5 (Your climate cluster DataArray, e.g., 0-5 values)

    forest_ages_dict = process_forest_age(
        hilda_ds=hilda, 
        lu_obs_ds=lu_actual, 
        climate_clusters=clim_k5_2020,
        target_years=[1920, 1950, 1980, 2010]
    )

    # Convert dictionary to Dataset for saving
    ds_forest = xr.Dataset()
    for yr, data in forest_ages_dict.items():
        ds_forest[f'forest_age_{yr}'] = (('lat', 'lon'), data)
        
    ds_forest.to_netcdf("data/synthetic_lu/forest_age.nc")