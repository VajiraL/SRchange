#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Truth-based gating analysis for 2020a using aggregated ensemble means.
Inputs
------
- Aggregated predictions from Script A:
    output_all_models_k6_agg/agg_2020a.nc
- Ground-truth SR:
    data/target-mergedSR_masked.nc

Outputs
-------
- model_metrics.csv
- summary_by_cluster.csv
- summary_by_boundary_distance.csv
- assigned_rank_histogram.csv
- analysis_maps_2020a.nc
"""
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

AGG_2020A_FILE = "output_all_models_k6_agg/agg_2020a.nc"
TRUTH_FILE = "data/target-mergedSR_masked.nc"
OUTDIR = "clusterModelAnalysisMetricsV2"

K = 6
COMP = dict(zlib=True, complevel=4)

BOUNDARY_BINS = [0, 1, 2, 3, 5, 10, 20, np.inf]
BOUNDARY_LABELS = ["0", "1", "2", "3-4", "5-9", "10-19", "20+"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_truth_like(template_ds):
    sr = xr.open_dataarray(TRUTH_FILE).where(lambda x: x.lat < 74.42, drop=True)
    sr = sr.interp_like(template_ds["assigned_cluster"], method="nearest")
    return sr


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "r2": np.nan
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    err = yp - yt
    sse = np.sum(err ** 2)
    sst = np.sum((yt - yt.mean()) ** 2)

    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    r2 = np.nan if sst == 0 else 1.0 - sse / sst

    return {
        "n": n,
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "r2": float(r2)
    }

def make_boundary_mask(cluster_map):
    """
    Boundary cells are any cells whose right/down/up/left neighbor has a different cluster.
    """
    arr = cluster_map.copy()
    valid = np.isfinite(arr)

    boundary = np.zeros(arr.shape, dtype=bool)

    # right/left
    diff_x = np.zeros(arr.shape, dtype=bool)
    diff_x[:, :-1] = valid[:, :-1] & valid[:, 1:] & (arr[:, :-1] != arr[:, 1:])
    diff_x[:, 1:] |= valid[:, :-1] & valid[:, 1:] & (arr[:, :-1] != arr[:, 1:])

    # up/down
    diff_y = np.zeros(arr.shape, dtype=bool)
    diff_y[:-1, :] = valid[:-1, :] & valid[1:, :] & (arr[:-1, :] != arr[1:, :])
    diff_y[1:, :] |= valid[:-1, :] & valid[1:, :] & (arr[:-1, :] != arr[1:, :])

    boundary = diff_x | diff_y
    return boundary

def boundary_distance_cells(cluster_map):
    boundary = make_boundary_mask(cluster_map)
    # distance to nearest boundary cell, in grid-cell units
    dist = distance_transform_edt(~boundary).astype(np.float32)
    dist[~np.isfinite(cluster_map)] = np.nan
    return dist

def flatten_valid(*arrays):
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return [a[mask] for a in arrays], mask

# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading aggregated predictions...")
    ds = xr.open_dataset(AGG_2020A_FILE)

    print("Loading truth...")
    sr_true = load_truth_like(ds)
    y_true = sr_true.values.astype(np.float32)

    assigned_cluster = ds["assigned_cluster"].values.astype(np.float32)

    pred_assigned = ds["pred_assigned_mean"].values.astype(np.float32)
    pred_global = ds["pred_global_mean"].values.astype(np.float32)

    pred_assigned_std = ds["pred_assigned_std"].values.astype(np.float32)
    pred_global_std = ds["pred_global_std"].values.astype(np.float32)

    pred_assigned_count = ds["pred_assigned_count"].values.astype(np.float32)

    
    cluster_preds = []
    cluster_stds = []
    for c in range(K):
        v = f"pred_cluster_{c}_mean"
        if v not in ds.data_vars:
            raise ValueError(f"Missing variable: {v}")
        cluster_preds.append(ds[v].values.astype(np.float32))
        cluster_stds.append(ds[f"pred_cluster_{c}_std"].values.astype(np.float32))
    
    # --------------------------------------------------------
    # Boundary distance
    # --------------------------------------------------------
    print("Computing boundary distance...")
    dist_cells = boundary_distance_cells(assigned_cluster)

    # --------------------------------------------------------
    # Global model metrics
    # --------------------------------------------------------
    print("Computing global metrics...")
    model_rows = []

    metrics_assigned = compute_metrics(y_true, pred_assigned)
    metrics_assigned["model"] = "assigned"
    model_rows.append(metrics_assigned)

    metrics_global = compute_metrics(y_true, pred_global)
    metrics_global["model"] = "global"
    model_rows.append(metrics_global)

    for c in range(K):
        m = compute_metrics(y_true, cluster_preds[c])
        m["model"] = f"cluster_{c}"
        model_rows.append(m)

    model_metrics = pd.DataFrame(model_rows)[["model", "n", "rmse", "mae", "bias", "r2"]]
    model_metrics.to_csv(os.path.join(OUTDIR, "model_metrics.csv"), index=False)

    # --------------------------------------------------------
    # Per-cell diagnostics (only where truth and assigned exist)
    # --------------------------------------------------------
    print("Computing per-cell diagnostics...")

    # valid cells for main analysis
    valid_main = np.isfinite(y_true) & np.isfinite(pred_assigned) & (pred_assigned_count > 0)

    # cluster prediction stack on valid cells
    yv = y_true[valid_main]
    assignedv = pred_assigned[valid_main]
    globalv = pred_global[valid_main]
    cluster_map_v = assigned_cluster[valid_main].astype(np.int16)
    dist_v = dist_cells[valid_main]
    assigned_std_v = pred_assigned_std[valid_main]
    global_std_v = pred_global_std[valid_main]

    #cluster_mat = np.stack([cp[valid_main] for cp in cluster_preds], axis=1)  # (n, K)
    # Build cluster prediction matrix with guaranteed ID alignment
    cluster_mat = np.full((np.sum(valid_main), K), np.nan, dtype=np.float32)

    for c in range(K):
        arr = ds[f"pred_cluster_{c}_mean"].values.astype(np.float32)
        cluster_mat[:, c] = arr[valid_main]

    # absolute errors
    ae_assigned = np.abs(assignedv - yv)
    ae_global = np.abs(globalv - yv)
    ae_clusters = np.abs(cluster_mat - yv[:, None])
    ae_clusters[~np.isfinite(ae_clusters)] = np.inf

    best_cluster_id = np.argmin(ae_clusters, axis=1).astype(np.int16)
    best_cluster_ae = np.min(ae_clusters, axis=1)

    # rank of assigned cluster model among cluster-only models
    row_idx = np.arange(ae_clusters.shape[0])
    assigned_cluster_ae = ae_clusters[row_idx, cluster_map_v]

    # rank = 1 + number of models with strictly smaller AE
    assigned_rank = 1 + np.sum(ae_clusters < assigned_cluster_ae[:, None], axis=1)

    assigned_is_best = (best_cluster_id == cluster_map_v)
    assigned_beats_global = ae_assigned < ae_global

    ae_diff_assigned_minus_global = ae_assigned - ae_global
    ae_diff_assigned_minus_best_cluster = assigned_cluster_ae - best_cluster_ae

    # --------------------------------------------------------
    # Summary by assigned cluster
    # --------------------------------------------------------
    print("Summarizing by assigned cluster...")
    rows_cluster = []
    for c in range(K):
        m = cluster_map_v == c
        if m.sum() == 0:
            continue

        rows_cluster.append({
            "assigned_cluster": c,
            "n": int(m.sum()),
            "mae_assigned": float(np.mean(ae_assigned[m])),
            "mae_global": float(np.mean(ae_global[m])),
            "mean_ae_diff_assigned_minus_global": float(np.mean(ae_diff_assigned_minus_global[m])),
            "median_ae_diff_assigned_minus_global": float(np.median(ae_diff_assigned_minus_global[m])),
            "assigned_beats_global_rate": float(np.mean(assigned_beats_global[m])),
            "assigned_is_best_cluster_rate": float(np.mean(assigned_is_best[m])),
            "mean_assigned_rank_cluster_only": float(np.mean(assigned_rank[m])),
            "mean_pred_assigned_std": float(np.mean(assigned_std_v[m])),
            "mean_pred_global_std": float(np.mean(global_std_v[m])),
        })

    df_cluster = pd.DataFrame(rows_cluster)
    df_cluster.to_csv(os.path.join(OUTDIR, "summary_by_cluster.csv"), index=False)

    # --------------------------------------------------------
    # Summary by boundary distance
    # --------------------------------------------------------
    print("Summarizing by boundary distance...")
    dist_bin = pd.cut(dist_v, bins=BOUNDARY_BINS, labels=BOUNDARY_LABELS, right=False)

    rows_dist = []
    for label in BOUNDARY_LABELS:
        m = (dist_bin == label)
        if np.sum(m) == 0:
            continue

        rows_dist.append({
            "boundary_distance_bin": label,
            "n": int(np.sum(m)),
            "mae_assigned": float(np.mean(ae_assigned[m])),
            "mae_global": float(np.mean(ae_global[m])),
            "mean_ae_diff_assigned_minus_global": float(np.mean(ae_diff_assigned_minus_global[m])),
            "median_ae_diff_assigned_minus_global": float(np.median(ae_diff_assigned_minus_global[m])),
            "assigned_beats_global_rate": float(np.mean(assigned_beats_global[m])),
            "assigned_is_best_cluster_rate": float(np.mean(assigned_is_best[m])),
            "mean_assigned_rank_cluster_only": float(np.mean(assigned_rank[m])),
            "mean_pred_assigned_std": float(np.mean(assigned_std_v[m])),
            "mean_pred_global_std": float(np.mean(global_std_v[m])),
        })

    df_dist = pd.DataFrame(rows_dist)
    df_dist.to_csv(os.path.join(OUTDIR, "summary_by_boundary_distance.csv"), index=False)

    # --------------------------------------------------------
    # Rank histogram
    # --------------------------------------------------------
    print("Writing rank histogram...")
    rank_hist = (
        pd.Series(assigned_rank, name="assigned_rank")
        .value_counts()
        .sort_index()
        .rename_axis("rank")
        .reset_index(name="count")
    )
    rank_hist["fraction"] = rank_hist["count"] / rank_hist["count"].sum()
    rank_hist.to_csv(os.path.join(OUTDIR, "assigned_rank_histogram.csv"), index=False)

    # --------------------------------------------------------
    # Save analysis maps
    # --------------------------------------------------------
    print("Saving analysis maps...")

    # initialize full-grid maps
    shape = y_true.shape
    nan32 = lambda: np.full(shape, np.nan, dtype=np.float32)
    nan16 = lambda: np.full(shape, -1, dtype=np.int16)
    nan8 = lambda: np.full(shape, -1, dtype=np.int8)

    ae_assigned_map = nan32()
    ae_global_map = nan32()
    ae_diff_ag_map = nan32()
    ae_diff_abest_map = nan32()
    rank_map = nan32()
    best_cluster_map = nan16()
    assigned_best_map = nan8()

    ae_assigned_map[valid_main] = ae_assigned.astype(np.float32)
    ae_global_map[valid_main] = ae_global.astype(np.float32)
    ae_diff_ag_map[valid_main] = ae_diff_assigned_minus_global.astype(np.float32)
    ae_diff_abest_map[valid_main] = ae_diff_assigned_minus_best_cluster.astype(np.float32)
    rank_map[valid_main] = assigned_rank.astype(np.float32)
    best_cluster_map[valid_main] = best_cluster_id.astype(np.int16)
    assigned_best_map[valid_main] = assigned_is_best.astype(np.int8)

    ds_maps = xr.Dataset(
        {
            "sr_true": (("lat", "lon"), y_true.astype(np.float32)),
            "pred_assigned_mean": (("lat", "lon"), pred_assigned.astype(np.float32)),
            "pred_global_mean": (("lat", "lon"), pred_global.astype(np.float32)),
            "pred_assigned_std": (("lat", "lon"), pred_assigned_std.astype(np.float32)),
            "pred_global_std": (("lat", "lon"), pred_global_std.astype(np.float32)),
            "assigned_cluster": (("lat", "lon"), assigned_cluster.astype(np.int16)),
            "boundary_distance_cells": (("lat", "lon"), dist_cells.astype(np.float32)),
            "ae_assigned": (("lat", "lon"), ae_assigned_map),
            "ae_global": (("lat", "lon"), ae_global_map),
            "ae_diff_assigned_minus_global": (("lat", "lon"), ae_diff_ag_map),
            "ae_diff_assigned_minus_best_cluster": (("lat", "lon"), ae_diff_abest_map),
            "assigned_rank_cluster_only": (("lat", "lon"), rank_map),
            "best_cluster_id": (("lat", "lon"), best_cluster_map),
            "assigned_is_best_cluster": (("lat", "lon"), assigned_best_map),
        },
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values}
    )

    out_nc = os.path.join(OUTDIR, "analysis_maps_2020a.nc")
    encoding = {}
    for v in ds_maps.data_vars:
        encoding[v] = COMP.copy()
        if np.issubdtype(ds_maps[v].dtype, np.floating):
            encoding[v]["dtype"] = "float32"
            encoding[v]["_FillValue"] = np.float32(np.nan)
        else:
            encoding[v]["dtype"] = ds_maps[v].dtype

    ds_maps.to_netcdf(out_nc, encoding=encoding)

    # --------------------------------------------------------
    # Short console summary
    # --------------------------------------------------------
    print("\nDone.")
    print("\nModel metrics:")
    print(model_metrics.to_string(index=False))

    print("\nHeadline gating diagnostics:")
    print(f"Assigned beats global rate: {np.mean(assigned_beats_global):.3f}")
    print(f"Assigned is best cluster rate: {np.mean(assigned_is_best):.3f}")
    print(f"Mean assigned rank (cluster-only): {np.mean(assigned_rank):.3f}")
    print(f"Median AE diff assigned-global: {np.median(ae_diff_assigned_minus_global):.3f}")

    ds.close()
    ds_maps.close()
    gc.collect()

if __name__ == "__main__":
    main()