"""
explain_change.py
-----------------
Computes the marginal contributions (Symmetric Shapley Change Decomposition) 
of features and feature groups (Climate, Land Use, Static) to the change in 
predicted SR between two specified scenarios for any set of coordinates.

Decomposition formulation:
  Total Change (Δy) = pred_B - pred_A
  Symmetric Feature Contribution (Δphi_i):
    0.5 * [(phi_i_M_A(x_B) - phi_i_M_A(x_A)) + (phi_i_M_B(x_B) - phi_i_M_B(x_A))]
  Symmetric Model Shift Contribution (ΔModel):
    0.5 * [(M_B(x_B) - M_A(x_B)) + (M_B(x_A) - M_A(x_A))]

Supports single coordinate target inputs via CLI or batch processing via a CSV file.
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import shap
from datetime import datetime
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_hand_0p05.nc')
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/local_change')

STATIC_FEATURES = ['elev', 'slope', 'roughness', 'hand']
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
SCENARIO_LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + SCENARIO_LU_FEATURES

XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': 128,
    'device': 'cpu',
    'eval_metric': 'rmse',
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_clim_group(c_name: str) -> str:
    if 'ssp' in c_name:
        parts = c_name.split('_')
        return f"{parts[0]}_{parts[1]}/{parts[2]}"
    return c_name


# ---------------------------
# Training Representative Reference Models
# ---------------------------
def train_reference_models():
    """
    Trains a single set of representative reference models (1 Global + up to 7 Cluster models)
    using the 1991_2020 baseline dataset to produce SHAP explainers.
    """
    log("=== Commencing Reference Model Training ===")
    
    # 1. Load targets
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    
    # 2. Terrain (Static)
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    # 3. Climate (Baseline 1991-2020)
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    
    # 4. LU (Baseline 1991-2020)
    lu_ds = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, '1991_2020.nc'))
    lu = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]
    
    # 5. Clusters (Baseline 1991-2020)
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters = cl_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').values
    
    # 6. Create Flat training matrix
    features_3d = np.concatenate([terrain, clim, lu], axis=0)
    X_full = features_3d.reshape(len(ALL_FEATURES), -1).T
    y_full = sr_target.values.flatten()
    c_full = clusters.flatten()
    
    df_train = pd.DataFrame(X_full, columns=ALL_FEATURES)
    df_train['sr'] = y_full
    df_train['cluster'] = c_full
    df_train = df_train.dropna(subset=['sr'] + ALL_FEATURES)
    
    df_tr, df_va = train_test_split(df_train, test_size=0.2, random_state=42)
    
    # Train Global Model
    log("Training Representative Global Model...")
    dtr_g = xgb.DMatrix(df_tr[ALL_FEATURES], label=df_tr['sr'])
    dva_g = xgb.DMatrix(df_va[ALL_FEATURES], label=df_va['sr'])
    m_glob = xgb.train(XGB_PARAMS, dtr_g, 2000, evals=[(dva_g, 'val')], early_stopping_rounds=100, verbose_eval=False)
    
    # Initialize Global Explainer
    log("Initializing Global SHAP Explainer...")
    exp_glob = shap.TreeExplainer(m_glob)
    
    # Train Cluster Models
    cluster_models = {}
    cluster_explainers = {}
    for cid in range(7):
        m_tr = (df_tr['cluster'] == cid)
        m_va = (df_va['cluster'] == cid)
        if np.sum(m_tr) < 100: 
            continue
        
        log(f"Training Representative Cluster {cid} Model...")
        dtr_c = xgb.DMatrix(df_tr[ALL_FEATURES][m_tr], label=df_tr['sr'][m_tr])
        dva_c = xgb.DMatrix(df_va[ALL_FEATURES][m_va], label=df_va['sr'][m_va])
        m_clust = xgb.train(XGB_PARAMS, dtr_c, 1600, evals=[(dva_c, 'val')], early_stopping_rounds=100, verbose_eval=False)
        
        cluster_models[cid] = m_clust
        cluster_explainers[cid] = shap.TreeExplainer(m_clust)
        
    log("=== Reference Model Training Complete ===")
    return m_glob, exp_glob, cluster_models, cluster_explainers


# ---------------------------
# Helpers for Single-Row Predictions and Attributions
# ---------------------------
def get_pred(model, df: pd.DataFrame) -> float:
    d = xgb.DMatrix(df)
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        return float(model.predict(d, iteration_range=(0, best_iter + 1))[0])
    return float(model.predict(d)[0])


def get_shap(explainer, df: pd.DataFrame) -> np.ndarray:
    res = explainer.shap_values(df)
    if isinstance(res, list):
        res = res[0]
    if hasattr(res, "values"):
        res = res.values
    return np.array(res[0], dtype=np.float32)


# ---------------------------
# Vectorized Feature Extraction and Processing Pipeline
# ---------------------------
def execute_decomposition(
    coords_list: list[tuple[float, float]], 
    scenario_A: tuple[str, str], 
    scenario_B: tuple[str, str],
    m_glob, exp_glob, cluster_models, cluster_explainers
):
    clim_A, lu_A = scenario_A
    clim_B, lu_B = scenario_B
    
    log(f"Processing {len(coords_list)} coordinates...")
    log(f"Scenario A (Baseline): CLIM={clim_A}, LU={lu_A}")
    log(f"Scenario B (Change):   CLIM={clim_B}, LU={lu_B}")
    
    # 1. Snap coordinates to grid
    sr_target = xr.open_dataarray(SR_TARGET_PATH)
    lat_vals = sr_target.lat.values
    lon_vals = sr_target.lon.values
    
    snapped_coords = []
    for lat, lon in coords_list:
        lat_idx = np.argmin(np.abs(lat_vals - lat))
        lon_idx = np.argmin(np.abs(lon_vals - lon))
        snapped_lat = lat_vals[lat_idx]
        snapped_lon = lon_vals[lon_idx]
        flat_idx = lat_idx * len(lon_vals) + lon_idx
        snapped_coords.append((lat, lon, snapped_lat, snapped_lon, flat_idx))
        
    # Build vectorized selectors
    lat_da = xr.DataArray([c[2] for c in snapped_coords], dims="pixel")
    lon_da = xr.DataArray([c[3] for c in snapped_coords], dims="pixel")
    
    # 2. Extract Static features
    log("Loading and aligning static features...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain_sub = terrain_ds.sel(lat=lat_da, lon=lon_da, method="nearest").load()
    
    # Helper to retrieve 2D DataArrays from extra indices (e.g. squeezed bands)
    def clean_ds(ds):
        for dim in ['band', 'time']:
            if dim in ds.dims:
                ds = ds.isel({dim: 0})
        return ds

    terrain_sub = clean_ds(terrain_sub)
    
    # 3. Extract Scenario A features and clusters
    log("Extracting features for Scenario A...")
    clim_group_A = get_clim_group(clim_A)
    clim_ds_A = xr.open_zarr(CLIMATE_ZARR, group=clim_group_A)[CLIMATE_FEATURES]
    clim_sub_A = clean_ds(clim_ds_A.sel(lat=lat_da, lon=lon_da, method="nearest").load())
    
    cl_ds_A = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=clim_group_A)['cluster']
    cl_sub_A = cl_ds_A.sel(lat=lat_da, lon=lon_da, method="nearest").load()
    
    lu_ds_A = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{lu_A}.nc"))[SCENARIO_LU_FEATURES]
    lu_sub_A = clean_ds(lu_ds_A.sel(lat=lat_da, lon=lon_da, method="nearest").load())
    
    # 4. Extract Scenario B features and clusters
    log("Extracting features for Scenario B...")
    clim_group_B = get_clim_group(clim_B)
    clim_ds_B = xr.open_zarr(CLIMATE_ZARR, group=clim_group_B)[CLIMATE_FEATURES]
    clim_sub_B = clean_ds(clim_ds_B.sel(lat=lat_da, lon=lon_da, method="nearest").load())
    
    cl_ds_B = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=clim_group_B)['cluster']
    cl_sub_B = cl_ds_B.sel(lat=lat_da, lon=lon_da, method="nearest").load()
    
    lu_ds_B = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{lu_B}.nc"))[SCENARIO_LU_FEATURES]
    lu_sub_B = clean_ds(lu_ds_B.sel(lat=lat_da, lon=lon_da, method="nearest").load())
    
    # 5. Decomposition loop across pixels
    results = []
    log("Executing Symmetric Decomposition on selected coordinates...")
    for idx, (orig_lat, orig_lon, snap_lat, snap_lon, flat_idx) in enumerate(snapped_coords):
        
        # Build feature vectors
        try:
            feat_vals_A = {}
            feat_vals_B = {}
            for f in STATIC_FEATURES:
                val = float(terrain_sub[f].isel(pixel=idx).values)
                feat_vals_A[f] = val
                feat_vals_B[f] = val
                
            for f in CLIMATE_FEATURES:
                feat_vals_A[f] = float(clim_sub_A[f].isel(pixel=idx).values)
                feat_vals_B[f] = float(clim_sub_B[f].isel(pixel=idx).values)
                
            for f in SCENARIO_LU_FEATURES:
                feat_vals_A[f] = float(lu_sub_A[f].isel(pixel=idx).values)
                feat_vals_B[f] = float(lu_sub_B[f].isel(pixel=idx).values)
        except Exception as e:
            log(f"Skipping coordinate ({orig_lat}, {orig_lon}): Failed to extract values (likely NaN region). Error: {e}")
            continue

        x_A = np.array([feat_vals_A[f] for f in ALL_FEATURES])
        x_B = np.array([feat_vals_B[f] for f in ALL_FEATURES])
        
        # Check for NaNs
        if np.isnan(x_A).any() or np.isnan(x_B).any():
            log(f"Skipping coordinate ({orig_lat}, {orig_lon}) due to NaN values in features.")
            continue
            
        df_A = pd.DataFrame([x_A], columns=ALL_FEATURES)
        df_B = pd.DataFrame([x_B], columns=ALL_FEATURES)
        
        # Determine cluster-specific models
        try:
            cluster_val_A = cl_sub_A.isel(pixel=idx).values
            cluster_A = int(cluster_val_A) if not np.isnan(cluster_val_A) else -1
        except Exception:
            cluster_A = -1
            
        try:
            cluster_val_B = cl_sub_B.isel(pixel=idx).values
            cluster_B = int(cluster_val_B) if not np.isnan(cluster_val_B) else -1
        except Exception:
            cluster_B = -1

        # Model A selection
        if cluster_A in cluster_models:
            M_A = cluster_models[cluster_A]
            exp_A = cluster_explainers[cluster_A]
        else:
            M_A = m_glob
            exp_A = exp_glob
            
        # Model B selection
        if cluster_B in cluster_models:
            M_B = cluster_models[cluster_B]
            exp_B = cluster_explainers[cluster_B]
        else:
            M_B = m_glob
            exp_B = exp_glob
            
        # 1. Prediction Transitions
        y_AA = get_pred(M_A, df_A)  # M_A(x_A)
        y_AB = get_pred(M_A, df_B)  # M_A(x_B)
        y_BA = get_pred(M_B, df_A)  # M_B(x_A)
        y_BB = get_pred(M_B, df_B)  # M_B(x_B)
        
        # 2. SHAP Attribution Transitions
        shap_AA = get_shap(exp_A, df_A)  # phi_M_A(x_A)
        shap_AB = get_shap(exp_A, df_B)  # phi_M_A(x_B)
        shap_BA = get_shap(exp_B, df_A)  # phi_M_B(x_A)
        shap_BB = get_shap(exp_B, df_B)  # phi_M_B(x_B)
        
        # 3. Symmetric Change Decompositions
        delta_phi = 0.5 * ((shap_AB - shap_AA) + (shap_BB - shap_BA))
        delta_model = 0.5 * ((y_BB - y_AB) + (y_BA - y_AA))
        total_change = y_BB - y_AA
        
        # Group contributions
        climate_contrib = float(np.sum([delta_phi[ALL_FEATURES.index(f)] for f in CLIMATE_FEATURES]))
        lu_contrib = float(np.sum([delta_phi[ALL_FEATURES.index(f)] for f in SCENARIO_LU_FEATURES]))
        static_contrib = float(np.sum([delta_phi[ALL_FEATURES.index(f)] for f in STATIC_FEATURES]))
        
        # Assemble dictionary row
        row_res = {
            'input_lat': orig_lat,
            'input_lon': orig_lon,
            'snapped_lat': snap_lat,
            'snapped_lon': snap_lon,
            'flat_idx': flat_idx,
            'cluster_A': cluster_A,
            'cluster_B': cluster_B,
            'pred_A': y_AA,
            'pred_B': y_BB,
            'total_change': total_change,
            'group_climate_effect': climate_contrib,
            'group_lu_effect': lu_contrib,
            'group_static_effect': static_contrib,
            'cluster_shift_effect': delta_model
        }
        
        # Append individual feature values and contributions
        for i, feat in enumerate(ALL_FEATURES):
            row_res[f"{feat}_val_A"] = float(x_A[i])
            row_res[f"{feat}_val_B"] = float(x_B[i])
            row_res[f"{feat}_effect"] = float(delta_phi[i])
            
        results.append(row_res)
        
    return pd.DataFrame(results)


# ---------------------------
# Main Controller
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute symmetric change decomposition for SR changes between scenarios."
    )
    # Target scenario pair
    parser.add_argument("--scenario_A_clim", type=str, default="1901_1930", help="Baseline climate period")
    parser.add_argument("--scenario_A_lu", type=str, default="pre-industrial", help="Baseline Land Use period")
    parser.add_argument("--scenario_B_clim", type=str, default="1991_2020", help="Target climate period")
    parser.add_argument("--scenario_B_lu", type=str, default="1991_2020", help="Target Land Use period")
    
    # Coordinates input
    parser.add_argument("--lat", type=float, default=None, help="Target single coordinate latitude")
    parser.add_argument("--lon", type=float, default=None, help="Target single coordinate longitude")
    parser.add_argument("--coords_csv", type=str, default=None, help="Path to CSV containing lat, lon columns")
    parser.add_argument("--output_name", type=str, default=None, help="Name of output CSV file")
    
    args = parser.parse_args()
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # 1. Resolve coordinates
    coords_list = []
    if args.coords_csv:
        if os.path.exists(args.coords_csv):
            log(f"Reading target coordinates from: {args.coords_csv}")
            coords_df = pd.read_csv(args.coords_csv)
            if 'lat' in coords_df.columns and 'lon' in coords_df.columns:
                coords_list = list(zip(coords_df['lat'], coords_df['lon']))
            else:
                raise ValueError("CSV must contain 'lat' and 'lon' columns")
        else:
            raise FileNotFoundError(f"Provided coordinate CSV not found: {args.coords_csv}")
    elif args.lat is not None and args.lon is not None:
        coords_list = [(args.lat, args.lon)]
    else:
        # Fallback default location if nothing is specified (Example: Central Europe region)
        log("No coordinates provided. Using default coordinate: lat=58.77, lon=7.38")
        coords_list = [(58.77, 7.38)]
        
    # 2. Train baseline models once
    m_glob, exp_glob, cluster_models, cluster_explainers = train_reference_models()
    
    # 3. Compute explanations
    scenario_A = (args.scenario_A_clim, args.scenario_A_lu)
    scenario_B = (args.scenario_B_clim, args.scenario_B_lu)
    
    df_results = execute_decomposition(
        coords_list, scenario_A, scenario_B,
        m_glob, exp_glob, cluster_models, cluster_explainers
    )
    
    if len(df_results) == 0:
        log("No valid coordinates were successfully analyzed. Execution aborted.")
        return

    # 4. Save results to CSV
    filename = args.output_name or f"decomp_{args.scenario_A_clim}_to_{args.scenario_B_clim}.csv"
    output_path = os.path.join(OUTPUT_BASE, filename)
    df_results.to_csv(output_path, index=False)
    log(f"Marginal change decomposition successfully saved to: {output_path}")


if __name__ == "__main__":
    main()