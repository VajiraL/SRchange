"""
Out-of-Domain Cross-Validation: Objective Function Comparison for SR Modelling
===============================================================================
Strategy: Train on the middle 80% of the joint climate space (annual_precip x
mean_annual_tmp), validate on the extreme 20% — simulating prediction into an
unfamiliar climate state.

Models evaluated: global + 6 climate-clustered (7 total per objective)
Objectives compared: 10 variants across 7 objective types
Metrics: R², MAE, RMSE, pBIAS, Global Moran's I
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from datetime import datetime
from scipy.spatial import KDTree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Configuration
# ---------------------------
TRAIN_SAMPLE   = 3_000_000   # samples drawn from the inner 80% climate zone for training
TEST_SAMPLE    = 1_000_000    # samples drawn from the outer 20% climate zone for testing
MORAN_SAMPLES  = 200_000    # subsample used for Moran's I computation (expensive)
CORES          = 128        # nthread per XGBoost model
RANDOM_STATE   = 29

# --- Paths (mirror predict_sr_xgb_cpu.py) ---
DIR_ROOT               = "/home/vajira/SRchange"
SR_TARGET_PATH         = os.path.join(DIR_ROOT, "data/SR-target/merged_SR.nc")
CLIMATE_ZARR           = os.path.join(DIR_ROOT, "data/climate_features.zarr")
CLIMATE_CLUSTER_ZARR   = os.path.join(DIR_ROOT, "data/climate_clusters.zarr")
TERRAIN_FILE           = os.path.join(DIR_ROOT, "data/terrain_hand_0p05.nc")
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, "data/LU_features_2")
OUTPUT_DIR             = os.path.join(DIR_ROOT, "output_SR/ood_cv_2")
LOG_FILE               = os.path.join(OUTPUT_DIR, "ood_cv.log")

# --- Features (identical to production script) ---
STATIC_FEATURES  = ["elev", "slope", "roughness", "hand"]
CLIMATE_FEATURES = [
    "annual_precip", "aridity_index", "frost_months", "gdm_0", "gdm_10", "gdm_5",
    "longest_dry_spell_months", "max_monthly_precip", "max_monthly_tmean",
    "mean_annual_tmp", "min_monthly_precip", "min_monthly_tmean",
    "monthly_precip_std", "monthly_tmean_range", "monthly_tmean_std",
    "months_gt_25C", "months_gt_30C", "pet_annual", "precip_coldest_quarter",
    "precip_cv", "precip_deficit_annual", "precip_iqr", "precip_p10", "precip_p25",
    "precip_p5", "precip_p50", "precip_p75", "precip_p90", "precip_p95",
    "precip_warmest_quarter", "temp_driest_quarter", "temp_wettest_quarter",
    "tmean_iqr", "tmean_p10", "tmean_p25", "tmean_p5", "tmean_p50", "tmean_p75",
    "tmean_p90", "tmean_p95",
]
SCENARIO_LU_FEATURES = ["gHM", "irr_frac", "rain_frac", "forest_age"]
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + SCENARIO_LU_FEATURES

# --- Objectives to compare ---
# Format: (label, objective_string, extra_params_dict)
OBJECTIVES = [
    ("squarederror",       "reg:squarederror",     {}),
    ("absoluteerror",      "reg:absoluteerror",    {}),
#    ("pseudohubererror",   "reg:pseudohubererror", {}),
    ("squaredlogerror",    "reg:squaredlogerror",  {}),
    ("poisson",            "count:poisson",        {}),
    ("gamma",              "reg:gamma",            {}),
#   ("tweedie_1.1",        "reg:tweedie",          {"tweedie_variance_power": 1.1}),
    ("tweedie_1.5",        "reg:tweedie",          {"tweedie_variance_power": 1.5}),
#    ("tweedie_1.9",        "reg:tweedie",          {"tweedie_variance_power": 1.9}),
]

BASE_XGB_PARAMS = {
    "max_depth":        5,
    "eta":              0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "nthread":          CORES,
    "device":           "cpu",
    "eval_metric":      "rmse",
}

N_ROUNDS           = 1200
EARLY_STOPPING     = 100

# ---------------------------
# Helpers
# ---------------------------
def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return R², MAE, RMSE, pBIAS (%)."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "pBIAS": pbias}


def calculate_morans_i(coords: np.ndarray, residuals: np.ndarray,
                       n_samples: int = MORAN_SAMPLES) -> float:
    """
    Global Moran's I on a random subsample.
    coords   : (N, 2) array of [lon, lat]
    residuals: (N,)   array of prediction errors
    """
    if len(residuals) > n_samples:
        idx      = np.random.choice(len(residuals), n_samples, replace=False)
        coords   = coords[idx]
        residuals = residuals[idx]

    z     = residuals - np.mean(residuals)
    tree  = KDTree(coords)
    pairs = tree.query_pairs(r=1.0)   # ~1° neighbourhood

    w_sum     = 0.0
    numerator = 0.0
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist > 0:
            w          = 1.0 / dist
            numerator += w * z[i] * z[j]
            w_sum     += w

    denominator = np.sum(z ** 2)
    if w_sum == 0 or denominator == 0:
        return 0.0

    n = len(residuals)
    return (n / (2.0 * w_sum)) * (numerator / denominator)


# ---------------------------
# Data loading
# ---------------------------
def load_full_dataframe() -> pd.DataFrame:
    """Load all features and SR target into a flat DataFrame."""
    log("Loading SR target...")
    sr_target           = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals  = sr_target.lat.values, sr_target.lon.values

    log("Loading terrain features...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain    = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values.squeeze()
    if terrain.ndim == 4:
        terrain = terrain[:, 0, :, :]

    log("Loading 1991-2020 climate features...")
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group="1991_2020")[CLIMATE_FEATURES]
    clim    = clim_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if clim.ndim == 4:
        clim = clim[:, 0, :, :]

    log("Loading 1991-2020 LU features...")
    lu_ds = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, "1991_2020.nc"))
    lu    = lu_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if lu.ndim == 4:
        lu = lu[:, 0, :, :]

    log("Loading climate cluster labels (1991-2020)...")
    cl_da    = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group="1991_2020")["cluster"]
    clusters = cl_da.sel(lat=lat_vals, lon=lon_vals, method="nearest").values.flatten()

    log("Assembling DataFrame...")
    X = np.concatenate([terrain, clim, lu], axis=0).reshape(len(ALL_FEATURES), -1).T
    y = sr_target.values.flatten()

    df          = pd.DataFrame(X, columns=ALL_FEATURES)
    df["sr"]    = y
    df["lat"]   = np.repeat(lat_vals, len(lon_vals))
    df["lon"]   = np.tile(lon_vals, len(lat_vals))
    df["cluster"] = clusters.astype(float)

    df = df.dropna(subset=["sr"] + ALL_FEATURES)
    df = df[df["sr"] > 0].reset_index(drop=True)   # Poisson/Gamma/log require y > 0
    log(f"DataFrame ready: {len(df):,} valid pixels")
    return df


# ---------------------------
# Climate-space split
# ---------------------------
def make_ood_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pixels into TRAIN (inner 80% of joint climate space) and
    TEST (outer 20% = extremes) using quantile-based bounding box on
    annual_precip × mean_annual_tmp.

    'Outer 20%' = pixels outside the [10th, 90th] percentile range
    of BOTH climate axes (union of the four tails).
    """
    p_lo, p_hi = np.percentile(df["annual_precip"],  [10, 90])
    t_lo, t_hi = np.percentile(df["mean_annual_tmp"], [10, 90])

    inner_mask = (
        (df["annual_precip"]  >= p_lo) & (df["annual_precip"]  <= p_hi) &
        (df["mean_annual_tmp"] >= t_lo) & (df["mean_annual_tmp"] <= t_hi)
    )

    df_inner = df[inner_mask].reset_index(drop=True)
    df_outer = df[~inner_mask].reset_index(drop=True)

    log(f"Climate split — inner (train pool): {len(df_inner):,} | "
        f"outer (test pool): {len(df_outer):,}")
    log(f"  Precip range kept : [{p_lo:.1f}, {p_hi:.1f}]")
    log(f"  Temp   range kept : [{t_lo:.2f}, {t_hi:.2f}]")

    # Sample
    n_tr = min(TRAIN_SAMPLE, len(df_inner))
    n_te = min(TEST_SAMPLE,  len(df_outer))
    df_tr = df_inner.sample(n=n_tr, random_state=RANDOM_STATE)
    df_te = df_outer.sample(n=n_te, random_state=RANDOM_STATE)
    log(f"Sampled {n_tr:,} train | {n_te:,} test")
    return df_tr.reset_index(drop=True), df_te.reset_index(drop=True)


# ---------------------------
# Model training + evaluation
# ---------------------------
def train_and_eval(obj_label: str, objective: str, extra_params: dict,
                   df_tr: pd.DataFrame, df_va: pd.DataFrame,
                   df_te: pd.DataFrame) -> list[dict]:
    """
    Train global model + up to 6 cluster models, evaluate on OOD test set.
    Returns list of metric dicts (one per sub-model: 'global' + cluster ids).
    """
    params = {**BASE_XGB_PARAMS, "objective": objective, **extra_params}

    # squaredlogerror requires y > 0 (already filtered), but cap predictions
    results = []

    X_tr = df_tr[ALL_FEATURES].values
    y_tr = df_tr["sr"].values
    X_va = df_va[ALL_FEATURES].values
    y_va = df_va["sr"].values
    X_te = df_te[ALL_FEATURES].values
    y_te = df_te["sr"].values
    coords_te = df_te[["lon", "lat"]].values

    # ---- Global model ----
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=ALL_FEATURES)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=ALL_FEATURES)
    dte = xgb.DMatrix(X_te, feature_names=ALL_FEATURES)

    m_glob = xgb.train(
        params, dtr, N_ROUNDS,
        evals=[(dva, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )

    best_iter     = m_glob.best_iteration + 1
    pred_glob     = m_glob.predict(dte, iteration_range=(0, best_iter))
    pred_glob     = np.clip(pred_glob, 0, None)
    residuals_g   = y_te - pred_glob
    moran_g       = calculate_morans_i(coords_te, residuals_g)

    row = {"objective": obj_label, "model": "global",
           **metrics(y_te, pred_glob), "MoranI": moran_g,
           "best_iter": best_iter, 'n_samples': len(y_te)}
    results.append(row)
    log(f"  [{obj_label}] global  R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}  "
        f"pBIAS={row['pBIAS']:.2f}%  MoranI={moran_g:.4f}")

    # ---- Cluster models ----
    cl_tr = df_tr["cluster"].values
    cl_te = df_te["cluster"].values

    for cid in range(7):
        mask_tr = cl_tr == cid
        mask_te = cl_te == cid

        n_tr_c = np.sum(mask_tr)
        n_te_c = np.sum(mask_te)

        if n_tr_c < 50 or n_te_c < 10:
            log(f"  [{obj_label}] cluster {cid} skipped (train={n_tr_c}, test={n_te_c})")
            continue

        dtr_c = xgb.DMatrix(X_tr[mask_tr], label=y_tr[mask_tr], feature_names=ALL_FEATURES)
        dva_c_X = X_va[df_va["cluster"].values == cid]
        dva_c_y = y_va[df_va["cluster"].values == cid]

        if len(dva_c_y) < 10:
            # Fall back to full val set if cluster has too few val points
            dva_c = dva
        else:
            dva_c = xgb.DMatrix(dva_c_X, label=dva_c_y, feature_names=ALL_FEATURES)

        m_c = xgb.train(
            params, dtr_c, N_ROUNDS,
            evals=[(dva_c, "val")],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False,
        )

        dte_c    = xgb.DMatrix(X_te[mask_te], feature_names=ALL_FEATURES)
        best_c   = m_c.best_iteration + 1
        pred_c   = np.clip(m_c.predict(dte_c, iteration_range=(0, best_c)), 0, None)
        res_c    = y_te[mask_te] - pred_c
        moran_c  = calculate_morans_i(coords_te[mask_te], res_c)

        row_c = {"objective": obj_label, "model": f"cluster_{cid}",
                 **metrics(y_te[mask_te], pred_c), "MoranI": moran_c,
                 "best_iter": best_c, 'n_samples': len(y_te[mask_te])}
        results.append(row_c)
        log(f"  [{obj_label}] cluster {cid} R²={row_c['R2']:.4f}  "
            f"RMSE={row_c['RMSE']:.4f}  pBIAS={row_c['pBIAS']:.2f}%  "
            f"MoranI={moran_c:.4f}  (n_tr={n_tr_c}, n_te={n_te_c})")

        del m_c, dtr_c, dva_c, dte_c
        gc.collect()

    del m_glob, dtr, dva, dte
    gc.collect()
    return results


# ---------------------------
# Summary printing
# ---------------------------
def print_summary(all_results: list[dict]):
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 90)
    print("OUT-OF-DOMAIN CV RESULTS  (train=inner 80% climate space, test=outer 20%)")
    print("=" * 90)

    # Global model summary
    print("\n--- Global model ---")
    glob_df = df[df["model"] == "global"].copy()
    print(glob_df[["objective", "R2", "MAE", "RMSE", "pBIAS", "MoranI", "best_iter"]]
          .sort_values("R2", ascending=False)
          .to_string(index=False, float_format="{:.4f}".format))

    # Per-cluster aggregated summary (mean across clusters)
    print("\n--- Cluster models (mean across clusters) ---")
    clust_df = df[df["model"] != "global"].copy()
    agg = (clust_df
           .groupby("objective")[["R2", "MAE", "RMSE", "pBIAS", "MoranI"]]
           .mean()
           .reset_index()
           .sort_values("R2", ascending=False))
    print(agg.to_string(index=False, float_format="{:.4f}".format))

    # Full table
    print("\n--- Full results table ---")
    print(df[["objective", "model", "R2", "MAE", "RMSE", "pBIAS", "MoranI", "best_iter"]]
          .sort_values(["objective", "model"])
          .to_string(index=False, float_format="{:.4f}".format))
    print("=" * 90)


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("=" * 60)
    log("OOD Climate CV — Objective Function Comparison")
    log("=" * 60)

    # 1. Load data
    df = load_full_dataframe()

    # 2. Climate-space OOD split
    df_tr_pool, df_te = make_ood_split(df)
    del df; gc.collect()

    # 3. Hold out 20% of inner zone as validation (for early stopping)
    val_frac = 0.2
    n_va = int(len(df_tr_pool) * val_frac)
    idx_perm = np.random.default_rng(RANDOM_STATE).permutation(len(df_tr_pool))
    df_va = df_tr_pool.iloc[idx_perm[:n_va]].reset_index(drop=True)
    df_tr = df_tr_pool.iloc[idx_perm[n_va:]].reset_index(drop=True)
    log(f"Final train: {len(df_tr):,} | val (early-stop): {len(df_va):,} | "
        f"OOD test: {len(df_te):,}")

    # 4. Run all objectives
    all_results = []
    for obj_label, objective, extra in OBJECTIVES:
        log(f"\n{'─' * 50}")
        log(f"Objective: {obj_label}")
        t0 = time.time()
        try:
            rows = train_and_eval(obj_label, objective, extra, df_tr, df_va, df_te)
            all_results.extend(rows)
        except Exception as exc:
            log(f"  ERROR in {obj_label}: {exc}")
        log(f"  Done in {time.time() - t0:.1f}s")

    # 5. Save and print results
    results_df = pd.DataFrame(all_results)
    out_csv    = os.path.join(OUTPUT_DIR, "ood_cv_results.csv")
    results_df.to_csv(out_csv, index=False)
    log(f"\nResults saved → {out_csv}")

    print_summary(all_results)
    log("OOD CV complete.")


if __name__ == "__main__":
    main()
