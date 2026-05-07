#!/usr/bin/env python3

"""
Aggregate all-model predictions across folds/ensembles.

Rules
-----
- For 2020a: use only TEST cells from each fold, so each valid cell should get 3 predictions
  (1 per ensemble; each ensemble contributes that cell from exactly 1 of 5 folds).
- For all other scenarios: use all predictions from all 15 runs (3 ensembles x 5 folds).

Outputs
-------
One NetCDF per scenario with:
  pred_global_mean, pred_global_std, pred_global_count
  pred_assigned_mean, pred_assigned_std, pred_assigned_count
  pred_cluster_0_mean, ... pred_cluster_5_mean (+ std/count)
  assigned_cluster
"""

import os
import gc
import numpy as np
import xarray as xr

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_BASE = "output_all_models_k6"
OUTPUT_BASE = "output_all_models_k6_agg2"

N_ENSEMBLES = 3
N_FOLDS = 5

SCENARIOS = [
    '1930a', '1930i', '1930c', '1930p', '1930_1960LU', '1930_1990LU',
    '1960a', '1960i', '1960c', '1960p', '1960_1930LU', '1960_1990LU',
    '2020a', '2020i', '2020p', '2020_1930LU', '2020_1960LU', '2020_1990LU',
    '1990i', '1990c', '1990p', '1990a', '1990_1930LU', '1990_1960LU',
]


COMP = dict(zlib=True, complevel=4)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def open_pred_file(ens, fold, scen):
    path = os.path.join(INPUT_BASE, f"ensemble_{ens}", f"fold_{fold}", f"preds_{scen}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)


def open_test_mask(ens, fold):
    path = os.path.join(INPUT_BASE, f"ensemble_{ens}", f"fold_{fold}", "test_mask.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ds = xr.open_dataset(path)
    if "test_mask" in ds.data_vars:
        da = ds["test_mask"]
    else:
        # fallback if saved as DataArray
        if len(ds.data_vars) != 1:
            raise ValueError(f"Cannot determine test_mask variable in {path}")
        da = list(ds.data_vars.values())[0]
    return da


def get_prediction_vars(sample_ds):
    return [v for v in sample_ds.data_vars if v.startswith("pred_")]


def aggregate_scenario(scen):
    print(f"\nAggregating scenario: {scen}")

    # use first file as template
    sample = open_pred_file(1, 1, scen)
    lat = sample["lat"].values
    lon = sample["lon"].values
    shape = (len(lat), len(lon))

    pred_vars = get_prediction_vars(sample)
    if "assigned_cluster" not in sample.data_vars:
        raise ValueError("assigned_cluster not found in prediction file.")

    assigned_cluster = sample["assigned_cluster"].values.astype(np.int16)
    sample.close()

    # accumulate per variable
    sum_dict = {v: np.zeros(shape, dtype=np.float64) for v in pred_vars}
    sumsq_dict = {v: np.zeros(shape, dtype=np.float64) for v in pred_vars}
    count_dict = {v: np.zeros(shape, dtype=np.uint16) for v in pred_vars}

    for ens in range(1, N_ENSEMBLES + 1):
        for fold in range(1, N_FOLDS + 1):
            print(f"  ensemble={ens}, fold={fold}")
            ds = open_pred_file(ens, fold, scen)

            if scen == "2020a":
                mask_da = open_test_mask(ens, fold)
                test_mask = (mask_da.values == 1)
                mask_da.close()
            else:
                test_mask = None

            for v in pred_vars:
                arr = ds[v].values.astype(np.float32)

                if scen == "2020a":
                    valid = test_mask & np.isfinite(arr)
                else:
                    valid = np.isfinite(arr)

                if np.any(valid):
                    vals = arr[valid].astype(np.float64)
                    sum_dict[v][valid] += vals
                    sumsq_dict[v][valid] += vals * vals
                    count_dict[v][valid] += 1

            ds.close()
            gc.collect()

    # build output dataset
    out_vars = {
        "assigned_cluster": (("lat", "lon"), assigned_cluster)
    }

    for v in pred_vars:
        cnt = count_dict[v].astype(np.float32)
        mean = np.full(shape, np.nan, dtype=np.float32)
        std = np.full(shape, np.nan, dtype=np.float32)

        valid = cnt > 0
        if np.any(valid):
            mean_vals = sum_dict[v][valid] / cnt[valid]
            var_vals = (sumsq_dict[v][valid] / cnt[valid]) - (mean_vals ** 2)
            var_vals = np.maximum(var_vals, 0.0)
            std_vals = np.sqrt(var_vals)

            mean[valid] = mean_vals.astype(np.float32)
            std[valid] = std_vals.astype(np.float32)

        out_vars[f"{v}_mean"] = (("lat", "lon"), mean)
        out_vars[f"{v}_std"] = (("lat", "lon"), std)
        out_vars[f"{v}_count"] = (("lat", "lon"), count_dict[v].astype(np.uint16))

    ds_out = xr.Dataset(
        out_vars,
        coords={"lat": lat, "lon": lon}
    )

    ds_out.attrs["scenario"] = scen
    ds_out.attrs["aggregation_mode"] = (
        "2020a: test-only across 3 ensembles"
        if scen == "2020a"
        else "all 15 predictions (3 ensembles x 5 folds)"
    )

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    out_path = os.path.join(OUTPUT_BASE, f"agg_{scen}.nc")

    encoding = {}
    for v in ds_out.data_vars:
        encoding[v] = COMP.copy()
        if np.issubdtype(ds_out[v].dtype, np.floating):
            encoding[v]["dtype"] = "float32"
            encoding[v]["_FillValue"] = np.float32(np.nan)
        else:
            encoding[v]["dtype"] = ds_out[v].dtype

    ds_out.to_netcdf(out_path, encoding=encoding)
    print(f" wrote {out_path}")

    # free memory
    del ds_out, sum_dict, sumsq_dict, count_dict
    gc.collect()


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    for scen in SCENARIOS:
        aggregate_scenario(scen)


if __name__ == "__main__":
    main()

