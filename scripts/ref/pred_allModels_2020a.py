#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict all valid cells with all clustered models for one chosen k (default k=6),
for the present/training scenario only (2020a), using 3 ensemble seeds and 5 CV folds.

What it does
------------
For each ensemble seed and each CV fold:
  1) Load the 2020a features and present SR target
  2) Use the fold's test mask to define training cells
  3) Train one XGBoost model per cluster (for the chosen k)
  4) Predict ALL valid cells with EVERY cluster model
  5) Save outputs into one NetCDF per ensemble:
       - pred_cluster_<c>(fold, y, x)
       - assigned_pred_fold(fold, y, x)
       - cv_test_mosaic_assigned(y, x)
       - test_mask(fold, y, x)
       - assigned_cluster(y, x)

Notes
-----
- This is intentionally simplified to 2020a only.
- The heavy part is training; prediction is done for all valid cells once per trained model.
- Parallelization is across folds, using the same CPU strategy as your original script:
  5 folds in parallel, ~60 threads per XGBoost model. Adapt as needed. 
- IMPORTANT: You must adapt the 4 marked functions if your file layout differs:
    * load_target_sr()
    * load_2020a_features()
    * select_cluster_map_for_k()
    * load_test_mask_for_fold()
"""

import os
import gc
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from joblib import Parallel, delayed


# ============================================================
# Configuration
# ============================================================

# ---- Main analysis target
K = 6
SCENARIO = "2020a"

# ---- CV / ensemble setup
N_FOLDS = 5
N_ENSEMBLES = 3

# Use your preferred 3 ensemble seeds here.
# If your original script used a specific seed list, replace these.
ENSEMBLE_SEEDS = [101, 202, 303]

# ---- Parallelism
# Matches your original CPU strategy well: 5 folds * 60 threads/model = 300 threads
N_PARALLEL_FOLDS = 5
CORES_PER_MODEL = 60

# ---- XGBoost config
# IMPORTANT: set NUM_BOOST_ROUND to the value from your original training script.
NUM_BOOST_ROUND = 300

XGB_PARAMS_BASE = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',  # GPU acceleration
    'eval_metric': 'rmse',
}
# ---- Paths: ADAPT THESE
TARGET_FILE = "data/target-mergedSR_masked.nc"
TARGET_VAR = None   # set explicit var name if needed, else first 2D variable is used

CLUSTER_MAPS_FILE = "output/sqrt_weight_ensemble/cluster_maps.nc"
CLUSTER_VAR = None  # set explicit var name if needed, else auto-detection is attempted

# The present-only features for 2020a.
# Fill this list with the NetCDFs / variables you already use in your original script.
# Example:
# FEATURE_SOURCES = [
#     {"file": "data/climate_features/climate_1991_2020.nc", "vars": None},
#     {"file": "data/LU_features/lu_2020_actual_filled.nc", "vars": None},
#     {"file": "data/topography/topography_features.nc", "vars": None},
# ]
FEATURE_SOURCES = [
    # TODO: replace with your actual 2020a feature sources
]

MASK_ROOT = "data/random_splits_MaskedMergedSR"

OUTDIR_BASE = "output_all_models/k6_2020a"

# ---- Prediction controls
PRED_CHUNK_SIZE = 2_000_000   # rows per chunk during prediction
MIN_CLUSTER_TRAIN_SAMPLES = 100

# Compression for NetCDF output
NC_COMP = dict(zlib=True, complevel=4)


# ============================================================
# Logging
# ============================================================

def log(msg, log_file=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")


# ============================================================
# Generic helpers
# ============================================================

def infer_2d_var(ds: xr.Dataset, preferred=None) -> xr.DataArray:
    if isinstance(ds, xr.DataArray):
        return ds

    if preferred is not None:
        if preferred not in ds.data_vars:
            raise ValueError(f"Variable '{preferred}' not found. Available: {list(ds.data_vars)}")
        da = ds[preferred]
        if da.ndim != 2:
            raise ValueError(f"Variable '{preferred}' is not 2D; dims={da.dims}")
        return da

    for v in ds.data_vars:
        if ds[v].ndim == 2:
            return ds[v]

    raise ValueError("No 2D variable found in dataset.")


def first_2d_or_bool_var(ds: xr.Dataset) -> xr.DataArray:
    # Prefer boolean/integer 2D variables for masks
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim == 2 and (np.issubdtype(da.dtype, np.bool_) or np.issubdtype(da.dtype, np.integer)):
            return da
    # Fallback
    return infer_2d_var(ds, preferred=None)


def align_like_template(da: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    da, template = xr.align(da, template, join="inner")
    if da.shape != template.shape:
        raise ValueError(f"Alignment failed: {da.shape} vs template {template.shape}")
    return da


def make_2d_nan(shape, dtype=np.float32):
    arr = np.full(shape, np.nan, dtype=dtype)
    return arr


def predict_in_chunks(booster: xgb.Booster, X: np.ndarray, feature_names, chunk_size=PRED_CHUNK_SIZE):
    preds = np.empty(X.shape[0], dtype=np.float32)
    start = 0
    while start < X.shape[0]:
        end = min(start + chunk_size, X.shape[0])
        dmat = xgb.DMatrix(X[start:end], feature_names=feature_names)
        preds[start:end] = booster.predict(dmat).astype(np.float32)
        start = end
    return preds


# ============================================================
# 1) Target loading
# ============================================================

def load_target_sr():
    """
    Load present SR target (2020a training target).
    Adapt TARGET_VAR if needed.
    """
    ds = xr.open_dataset(TARGET_FILE)
    da = infer_2d_var(ds, preferred=TARGET_VAR)
    return da


# ============================================================
# 2) Feature loading (2020a only)
# ============================================================

def load_2020a_features(template_da: xr.DataArray):
    """
    Load all predictor variables for 2020a and stack them into a single DataFrame-like matrix.

    You MUST populate FEATURE_SOURCES with the same variables used in training:
      - present climate variables (1991-2020)
      - 2020 actual LU variables
      - static topographic variables

    Returns
    -------
    ds_features : xr.Dataset
        Aligned 2D predictor dataset
    """
    if len(FEATURE_SOURCES) == 0:
        raise ValueError(
            "FEATURE_SOURCES is empty. Fill it with the same feature files used in your original 2020a training."
        )

    var_dict = {}
    for src in FEATURE_SOURCES:
        file_path = src["file"]
        vars_keep = src.get("vars", None)

        ds = xr.open_dataset(file_path)

        if vars_keep is None:
            candidate_vars = [v for v in ds.data_vars if ds[v].ndim == 2]
        else:
            candidate_vars = vars_keep

        if len(candidate_vars) == 0:
            raise ValueError(f"No 2D vars found in {file_path}")

        for v in candidate_vars:
            da = ds[v]
            if da.ndim != 2:
                continue
            da = align_like_template(da, template_da)
            # Ensure unique variable names if repeated across files
            out_name = v
            if out_name in var_dict:
                prefix = os.path.splitext(os.path.basename(file_path))[0]
                out_name = f"{prefix}__{v}"
            var_dict[out_name] = da.astype(np.float32)

    ds_features = xr.Dataset(var_dict)
    return ds_features


def dataset_to_feature_matrix(ds_features: xr.Dataset, target_da: xr.DataArray):
    """
    Convert aligned feature dataset + target grid into:
      - X_all_valid: matrix for all cells with finite predictors
      - pred_valid_mask: 2D mask where all predictors are finite
      - X_train_valid / y_train_valid handled later per fold
      - feature_names
    """
    feature_names = list(ds_features.data_vars)
    feat_arrays = [ds_features[v].values for v in feature_names]

    # shape: (n_features, ny, nx) -> (ny, nx, n_features)
    feat_stack = np.stack(feat_arrays, axis=0).transpose(1, 2, 0).astype(np.float32)

    pred_valid_mask = np.all(np.isfinite(feat_stack), axis=2)
    X_all_valid = feat_stack[pred_valid_mask]

    return X_all_valid, pred_valid_mask, feature_names, feat_stack


# ============================================================
# 3) Cluster-map loading / selection
# ============================================================

def select_cluster_map_for_k(template_da: xr.DataArray, k=6, scenario="2020a"):
    """
    Try to extract the cluster map for the requested k and scenario from CLUSTER_MAPS_FILE.

    This function attempts several common layouts:
      A) variable named like 'k6_2020a' or 'cluster_k6_2020a'
      B) dims include ('k', 'scenario', y, x)
      C) dims include ('K', 'scenario', y, x)
      D) file itself is already the desired 2D cluster map

    ADAPT THIS if your cluster_maps.nc uses a different structure.
    """
    ds = xr.open_dataset(CLUSTER_MAPS_FILE)

    # Case D: file directly contains one 2D cluster map
    two_d_vars = [v for v in ds.data_vars if ds[v].ndim == 2]
    if CLUSTER_VAR is not None and CLUSTER_VAR in ds.data_vars and ds[CLUSTER_VAR].ndim == 2:
        da = ds[CLUSTER_VAR]
        return align_like_template(da, template_da)

    # Case A: variable naming patterns
    candidate_names = [
        f"k{k}_{scenario}",
        f"cluster_k{k}_{scenario}",
        f"clusters_k{k}_{scenario}",
        f"{scenario}_k{k}",
        f"{scenario}_clusters_k{k}",
    ]
    if CLUSTER_VAR is not None:
        candidate_names = [CLUSTER_VAR] + candidate_names

    for v in candidate_names:
        if v in ds.data_vars and ds[v].ndim == 2:
            da = ds[v]
            return align_like_template(da, template_da)

    # Case B/C: dimensions include k and scenario
    for v in ds.data_vars:
        da = ds[v]
        dims = set(da.dims)

        if da.ndim < 2:
            continue

        has_k = ("k" in dims) or ("K" in dims)
        has_scen = ("scenario" in dims) or ("scen" in dims)

        if has_k and has_scen:
            kdim = "k" if "k" in da.dims else "K"
            sdim = "scenario" if "scenario" in da.dims else "scen"

            # scenario coord might be bytes/strings
            if scenario in da[sdim].values:
                da_sel = da.sel({kdim: k, sdim: scenario})
                if da_sel.ndim == 2:
                    return align_like_template(da_sel, template_da)

    # Last fallback: first 2D var
    if len(two_d_vars) == 1:
        da = ds[two_d_vars[0]]
        return align_like_template(da, template_da)

    raise ValueError(
        "Could not determine the cluster map automatically. "
        "Please adapt select_cluster_map_for_k() to your cluster_maps.nc structure."
    )


# ============================================================
# 4) Fold test-mask loading
# ============================================================

def load_test_mask_for_fold(template_da: xr.DataArray, ens_idx: int, fold_idx: int) -> xr.DataArray:
    """
    Load the precomputed test mask for one ensemble/fold.

    Because I don't know your exact file layout, this tries a few common patterns.
    If none work, adapt this function directly.

    Expected return:
      2D boolean DataArray aligned to template_da
      True = test cell, False = train cell
    """
    ens1 = ens_idx + 1
    fold1 = fold_idx + 1

    candidate_patterns = [
        os.path.join(MASK_ROOT, f"ensemble_{ens1}", f"*fold*{fold1}*.nc"),
        os.path.join(MASK_ROOT, f"ensemble_{ens1}", f"*{fold1}*.nc"),
        os.path.join(MASK_ROOT, f"ens_{ens1}", f"*fold*{fold1}*.nc"),
        os.path.join(MASK_ROOT, f"*ensemble*{ens1}*fold*{fold1}*.nc"),
        os.path.join(MASK_ROOT, f"*ens*{ens1}*fold*{fold1}*.nc"),
    ]

    candidates = []
    for pat in candidate_patterns:
        candidates.extend(sorted(glob.glob(pat)))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not find mask file for ensemble={ens1}, fold={fold1}. "
            f"Please adapt load_test_mask_for_fold()."
        )

    # Use first match
    mask_file = candidates[0]
    ds = xr.open_dataset(mask_file)
    da = first_2d_or_bool_var(ds)
    da = align_like_template(da, template_da)

    # Convert to bool
    mask = da.values
    if mask.dtype != bool:
        # Assume non-zero = test
        mask = mask != 0

    out = xr.DataArray(mask, dims=template_da.dims, coords=template_da.coords, name="test_mask")
    return out


# ============================================================
# Training / prediction per fold
# ============================================================

def run_one_fold(
    ens_idx,
    fold_idx,
    ensemble_seed,
    X_all_valid,
    pred_valid_mask,
    feat_stack,
    feature_names,
    target_da,
    cluster_da,
    cluster_ids,
    out_shape
):
    """
    Train models for one fold and predict all valid cells with all cluster models.
    Returns a dict with:
      - preds_by_cluster: {cluster_id: 2D array}
      - assigned_pred_2d: hard-assigned prediction for this fold
      - test_mask_2d
      - stats
    """
    y_full = target_da.values.astype(np.float32)
    clusters_full = cluster_da.values.astype(np.float32)

    # Load fold mask
    test_mask_da = load_test_mask_for_fold(target_da, ens_idx, fold_idx)
    test_mask = test_mask_da.values.astype(bool)

    # Training labels must be finite, features finite, cluster valid, and NOT in test mask
    valid_label_mask = np.isfinite(y_full)
    valid_cluster_mask = np.isfinite(clusters_full) & np.isin(clusters_full.astype(np.int32), cluster_ids)
    train_mask = (~test_mask) & valid_label_mask & valid_cluster_mask & pred_valid_mask

    # Flatten training data
    X_train = feat_stack[train_mask].astype(np.float32)
    y_train = y_full[train_mask].astype(np.float32)
    c_train = clusters_full[train_mask].astype(np.int32)

    # Prepare outputs
    preds_by_cluster = {}
    assigned_pred_2d = np.full(out_shape, np.nan, dtype=np.float32)

    # Train one model per cluster and predict all valid cells
    stats_rows = []

    for c in cluster_ids:
        idx = (c_train == c)
        n_train = int(idx.sum())

        if n_train < MIN_CLUSTER_TRAIN_SAMPLES:
            preds_by_cluster[c] = np.full(out_shape, np.nan, dtype=np.float32)
            stats_rows.append({
                "ensemble": ens_idx + 1,
                "fold": fold_idx + 1,
                "cluster": int(c),
                "n_train": n_train,
                "status": "skipped_too_few_samples"
            })
            continue

        params = dict(XGB_PARAMS_BASE)
        params["seed"] = int(ensemble_seed * 100 + (fold_idx + 1) * 10 + c)

        dtrain = xgb.DMatrix(X_train[idx], label=y_train[idx], feature_names=feature_names)
        booster = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)

        pred_vec = predict_in_chunks(booster, X_all_valid, feature_names=feature_names)
        pred_2d = np.full(out_shape, np.nan, dtype=np.float32)
        pred_2d[pred_valid_mask] = pred_vec
        preds_by_cluster[c] = pred_2d

        stats_rows.append({
            "ensemble": ens_idx + 1,
            "fold": fold_idx + 1,
            "cluster": int(c),
            "n_train": n_train,
            "status": "ok"
        })

        # Free booster memory explicitly
        del booster, dtrain, pred_vec, pred_2d
        gc.collect()

    # Hard-assigned prediction for this fold
    for c in cluster_ids:
        arr = preds_by_cluster[c]
        sel = (clusters_full.astype(np.int32) == c) & np.isfinite(arr)
        assigned_pred_2d[sel] = arr[sel]

    return {
        "fold_idx": fold_idx,
        "preds_by_cluster": preds_by_cluster,
        "assigned_pred_2d": assigned_pred_2d,
        "test_mask_2d": test_mask.astype(np.int8),
        "stats_df": pd.DataFrame(stats_rows)
    }


# ============================================================
# NetCDF assembly
# ============================================================

def build_ensemble_dataset(
    ensemble_number,
    fold_results,
    cluster_ids,
    target_da,
    cluster_da
):
    """
    Build one xarray.Dataset for a single ensemble.
    Variables:
      - pred_cluster_<c>(fold, y, x)
      - assigned_pred_fold(fold, y, x)
      - cv_test_mosaic_assigned(y, x)
      - test_mask(fold, y, x)
      - assigned_cluster(y, x)
    """
    ydim, xdim = target_da.dims
    ycoords = target_da[ydim]
    xcoords = target_da[xdim]
    fold_coord = np.arange(1, N_FOLDS + 1, dtype=np.int32)

    # sort fold results by fold_idx
    fold_results = sorted(fold_results, key=lambda r: r["fold_idx"])

    data_vars = {}

    # One variable per cluster model: dims (fold, y, x)
    for c in cluster_ids:
        stack = np.stack([fr["preds_by_cluster"][c] for fr in fold_results], axis=0)
        data_vars[f"pred_cluster_{c}"] = xr.DataArray(
            stack,
            dims=("fold", ydim, xdim),
            coords={"fold": fold_coord, ydim: ycoords, xdim: xcoords}
        )

    # Assigned prediction for each fold
    assigned_stack = np.stack([fr["assigned_pred_2d"] for fr in fold_results], axis=0)
    data_vars["assigned_pred_fold"] = xr.DataArray(
        assigned_stack,
        dims=("fold", ydim, xdim),
        coords={"fold": fold_coord, ydim: ycoords, xdim: xcoords}
    )

    # Test masks
    test_mask_stack = np.stack([fr["test_mask_2d"] for fr in fold_results], axis=0)
    data_vars["test_mask"] = xr.DataArray(
        test_mask_stack,
        dims=("fold", ydim, xdim),
        coords={"fold": fold_coord, ydim: ycoords, xdim: xcoords}
    )

    # Stitch the 20% held-out assigned predictions into one full CV mosaic
    cv_mosaic = np.full(target_da.shape, np.nan, dtype=np.float32)
    for fr in fold_results:
        m = fr["test_mask_2d"].astype(bool)
        cv_mosaic[m] = fr["assigned_pred_2d"][m]

    data_vars["cv_test_mosaic_assigned"] = xr.DataArray(
        cv_mosaic,
        dims=(ydim, xdim),
        coords={ydim: ycoords, xdim: xcoords}
    )

    # Assigned cluster map
    data_vars["assigned_cluster"] = xr.DataArray(
        cluster_da.values.astype(np.int16),
        dims=(ydim, xdim),
        coords={ydim: ycoords, xdim: xcoords}
    )

    ds_out = xr.Dataset(data_vars)
    ds_out.attrs["scenario"] = SCENARIO
    ds_out.attrs["k"] = int(K)
    ds_out.attrs["ensemble"] = int(ensemble_number)
    ds_out.attrs["n_folds"] = int(N_FOLDS)
    ds_out.attrs["num_boost_round"] = int(NUM_BOOST_ROUND)
    ds_out.attrs["xgb_params"] = json.dumps(XGB_PARAMS_BASE)

    return ds_out


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTDIR_BASE, exist_ok=True)
    master_log = os.path.join(OUTDIR_BASE, "run.log")

    log("=" * 70, master_log)
    log("START: all-cells x all-cluster-model prediction for 2020a", master_log)
    log(f"K={K}, scenario={SCENARIO}", master_log)

    # --------------------------------------------------------
    # Load target / features / cluster map once
    # --------------------------------------------------------
    log("Loading target SR...", master_log)
    target_da = load_target_sr()

    log("Loading 2020a features...", master_log)
    ds_features = load_2020a_features(target_da)

    log("Building feature matrix...", master_log)
    X_all_valid, pred_valid_mask, feature_names, feat_stack = dataset_to_feature_matrix(ds_features, target_da)

    log("Loading cluster map...", master_log)
    cluster_da = select_cluster_map_for_k(target_da, k=K, scenario=SCENARIO)

    cluster_vals = cluster_da.values
    cluster_ids = sorted([int(c) for c in np.unique(cluster_vals[np.isfinite(cluster_vals)]) if int(c) >= 0])

    if len(cluster_ids) != K:
        log(
            f"WARNING: found {len(cluster_ids)} non-negative cluster IDs in the selected map: {cluster_ids}. "
            f"This does not equal requested K={K}. Proceeding with found IDs.",
            master_log
        )

    log(f"Cluster IDs used for modeling: {cluster_ids}", master_log)
    log(f"Number of valid prediction cells: {int(pred_valid_mask.sum())}", master_log)

    # --------------------------------------------------------
    # Run per ensemble
    # --------------------------------------------------------
    for ens_idx, ens_seed in enumerate(ENSEMBLE_SEEDS):
        ens_num = ens_idx + 1
        ens_dir = os.path.join(OUTDIR_BASE, f"ensemble_{ens_num}")
        os.makedirs(ens_dir, exist_ok=True)
        log_file = os.path.join(ens_dir, "ensemble.log")

        log("-" * 70, log_file)
        log(f"Ensemble {ens_num}/{N_ENSEMBLES} | seed={ens_seed}", log_file)

        # Parallel over folds
        fold_results = Parallel(n_jobs=N_PARALLEL_FOLDS, backend="loky", verbose=10)(
            delayed(run_one_fold)(
                ens_idx=ens_idx,
                fold_idx=fold_idx,
                ensemble_seed=ens_seed,
                X_all_valid=X_all_valid,
                pred_valid_mask=pred_valid_mask,
                feat_stack=feat_stack,
                feature_names=feature_names,
                target_da=target_da,
                cluster_da=cluster_da,
                cluster_ids=cluster_ids,
                out_shape=target_da.shape
            )
            for fold_idx in range(N_FOLDS)
        )

        # Collect training stats
        stats_df = pd.concat([fr["stats_df"] for fr in fold_results], ignore_index=True)
        stats_csv = os.path.join(ens_dir, f"training_stats_k{K:02d}_{SCENARIO}_ensemble_{ens_num}.csv")
        stats_df.to_csv(stats_csv, index=False)

        # Build output dataset
        log("Assembling NetCDF output...", log_file)
        ds_out = build_ensemble_dataset(
            ensemble_number=ens_num,
            fold_results=fold_results,
            cluster_ids=cluster_ids,
            target_da=target_da,
            cluster_da=cluster_da
        )

        # Encoding
        encoding = {}
        for v in ds_out.data_vars:
            encoding[v] = NC_COMP.copy()
            if np.issubdtype(ds_out[v].dtype, np.floating):
                encoding[v]["dtype"] = "float32"
                encoding[v]["_FillValue"] = np.float32(np.nan)
            else:
                encoding[v]["dtype"] = ds_out[v].dtype

        out_nc = os.path.join(
            ens_dir,
            f"all_models_k{K:02d}_{SCENARIO}_ensemble_{ens_num}.nc"
        )
        ds_out.to_netcdf(out_nc, encoding=encoding)

        # Provenance / metadata
        meta = {
            "scenario": SCENARIO,
            "k": K,
            "ensemble": ens_num,
            "seed": ens_seed,
            "n_folds": N_FOLDS,
            "feature_names": feature_names,
            "cluster_ids": cluster_ids,
            "target_file": TARGET_FILE,
            "cluster_maps_file": CLUSTER_MAPS_FILE,
            "mask_root": MASK_ROOT,
            "feature_sources": FEATURE_SOURCES,
            "num_boost_round": NUM_BOOST_ROUND,
            "xgb_params_base": XGB_PARAMS_BASE,
            "n_parallel_folds": N_PARALLEL_FOLDS,
            "cores_per_model": CORES_PER_MODEL
        }
        with open(os.path.join(ens_dir, "run_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

        log(f"Wrote: {out_nc}", log_file)
        log(f"Wrote: {stats_csv}", log_file)

        # Memory cleanup
        del ds_out, fold_results, stats_df
        gc.collect()

    log("DONE.", master_log)
    log("=" * 70, master_log)


if __name__ == "__main__":
    main()