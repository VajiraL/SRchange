"""
OOD Cluster-Shift Assessment: Global vs Static-Cluster vs Shift-Cluster
========================================================================
Tests whether the production cluster-shift design (pixels reassigned to
clusters from the TARGET climate period) outperforms a global model on
out-of-domain climate conditions.

Three strategies compared, Poisson objective only:
  A. global        — single model, no cluster routing
  B. cluster_static — clusters from training period (1991-2020) for both
                      train and test pixels
  C. cluster_shift  — clusters from training period for train pixels,
                      clusters from each TARGET period for test pixels

Sub-experiment: run C for every CLIM_SCENARIO period in the zarr, giving
a performance curve across increasing climate distance from training.

Stratified analysis: for shift strategy, also report metrics split by
whether a pixel's cluster identity changed between periods.
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

warnings.filterwarnings("ignore")

# ---------------------------
# Configuration
# ---------------------------
TRAIN_SAMPLE  = 100_000
TEST_SAMPLE   = 30_000
MORAN_SAMPLES = 10_000
CORES         = 32
RANDOM_STATE  = 42

DIR_ROOT               = "/home/vajira/SRchange"
SR_TARGET_PATH         = os.path.join(DIR_ROOT, "data/SR-target/merged_SR.nc")
CLIMATE_ZARR           = os.path.join(DIR_ROOT, "data/climate_features.zarr")
CLIMATE_CLUSTER_ZARR   = os.path.join(DIR_ROOT, "data/climate_clusters.zarr")
TERRAIN_FILE           = os.path.join(DIR_ROOT, "data/terrain_hand_0p05.nc")
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, "data/LU_features")
OUTPUT_DIR             = os.path.join(DIR_ROOT, "output_SR/ood_cluster_shift")
LOG_FILE               = os.path.join(OUTPUT_DIR, "ood_cluster_shift.log")

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

# All climate periods whose cluster labels will be tested as "shift" targets
CLIM_SCENARIOS = [
    "1901_1930", "1931_1960", "1961_1990", "1991_2020",
    "2041_2070_ssp126", "2041_2070_ssp245", "2041_2070_ssp370", "2041_2070_ssp460",
    "2071_2099_ssp126", "2071_2099_ssp245", "2071_2099_ssp370", "2071_2099_ssp460",
]
TRAINING_PERIOD = "1991_2020"   # cluster labels used during training

XGB_PARAMS = {
    "objective":        "count:poisson",
    "max_depth":        5,
    "eta":              0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "nthread":          CORES,
    "device":           "cpu",
    "eval_metric":      "rmse",
}
N_ROUNDS       = 2000
EARLY_STOPPING = 100

# ---------------------------
# Helpers
# ---------------------------
def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")


def compute_metrics(y_true, y_pred):
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "pBIAS": pbias}


def calculate_morans_i(coords, residuals, n_samples=MORAN_SAMPLES):
    if len(residuals) > n_samples:
        idx       = np.random.choice(len(residuals), n_samples, replace=False)
        coords    = coords[idx]
        residuals = residuals[idx]
    z     = residuals - np.mean(residuals)
    tree  = KDTree(coords)
    pairs = tree.query_pairs(r=1.0)
    w_sum = numerator = 0.0
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist > 0:
            w          = 1.0 / dist
            numerator += w * z[i] * z[j]
            w_sum     += w
    denom = np.sum(z ** 2)
    if w_sum == 0 or denom == 0:
        return 0.0
    return (len(residuals) / (2.0 * w_sum)) * (numerator / denom)


def score_predictions(y_true, y_pred, coords, label_dict):
    """Full metric dict including Moran's I."""
    m = compute_metrics(y_true, y_pred)
    m["MoranI"] = calculate_morans_i(coords, y_true - y_pred)
    m.update(label_dict)
    return m


# ---------------------------
# Data loading
# ---------------------------
def get_clim_group(c_name):
    if "ssp" in c_name:
        parts = c_name.split("_")
        return f"{parts[0]}_{parts[1]}/{parts[2]}"
    return c_name


def load_full_dataframe():
    log("Loading SR target...")
    sr_target          = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values

    log("Loading terrain...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain    = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values.squeeze()
    if terrain.ndim == 4:
        terrain = terrain[:, 0, :, :]

    log(f"Loading climate features ({TRAINING_PERIOD})...")
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group=TRAINING_PERIOD)[CLIMATE_FEATURES]
    clim    = clim_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if clim.ndim == 4:
        clim = clim[:, 0, :, :]

    log(f"Loading LU features ({TRAINING_PERIOD})...")
    lu_ds = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{TRAINING_PERIOD}.nc"))
    lu    = lu_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if lu.ndim == 4:
        lu = lu[:, 0, :, :]

    log(f"Loading cluster labels ({TRAINING_PERIOD})...")
    cl_da      = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=TRAINING_PERIOD)["cluster"]
    clusters   = cl_da.sel(lat=lat_vals, lon=lon_vals, method="nearest").values.flatten()

    X = np.concatenate([terrain, clim, lu], axis=0).reshape(len(ALL_FEATURES), -1).T
    y = sr_target.values.flatten()

    df              = pd.DataFrame(X, columns=ALL_FEATURES)
    df["sr"]        = y
    df["lat"]       = np.repeat(lat_vals, len(lon_vals))
    df["lon"]       = np.tile(lon_vals, len(lat_vals))
    df["cluster_train"] = clusters.astype(float)

    df = df.dropna(subset=["sr"] + ALL_FEATURES)
    df = df[df["sr"] > 0].reset_index(drop=True)
    log(f"DataFrame ready: {len(df):,} valid pixels")
    return df, lat_vals, lon_vals


def load_cluster_labels_for_period(period, lat_vals, lon_vals):
    """Load cluster labels for a given climate period, aligned to lat/lon grid."""
    group = get_clim_group(period)
    cl_da = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=group)["cluster"]
    labels = cl_da.sel(lat=lat_vals, lon=lon_vals, method="nearest").values.flatten()
    return labels


# ---------------------------
# OOD climate-space split
# ---------------------------
def make_ood_split(df):
    p_lo, p_hi = np.percentile(df["annual_precip"],   [10, 90])
    t_lo, t_hi = np.percentile(df["mean_annual_tmp"], [10, 90])
    inner = (
        (df["annual_precip"]   >= p_lo) & (df["annual_precip"]   <= p_hi) &
        (df["mean_annual_tmp"] >= t_lo) & (df["mean_annual_tmp"] <= t_hi)
    )
    df_inner = df[inner].reset_index(drop=True)
    df_outer = df[~inner].reset_index(drop=True)
    log(f"Climate split — inner: {len(df_inner):,} | outer (OOD test pool): {len(df_outer):,}")
    log(f"  Precip [{p_lo:.1f}, {p_hi:.1f}] | Temp [{t_lo:.2f}, {t_hi:.2f}]")

    n_tr = min(TRAIN_SAMPLE, len(df_inner))
    n_te = min(TEST_SAMPLE,  len(df_outer))
    rng   = np.random.default_rng(RANDOM_STATE)

    df_tr = df_inner.sample(n=n_tr, random_state=RANDOM_STATE).reset_index(drop=True)
    df_te = df_outer.iloc[rng.permutation(len(df_outer))[:n_te]].reset_index(drop=True)
    log(f"Sampled {len(df_tr):,} train | {len(df_te):,} OOD test")
    return df_tr, df_te


# ---------------------------
# Model training
# ---------------------------
def train_models(df_tr, df_va):
    """
    Train global model + cluster models on training data.
    Returns (global_model, cluster_models_dict).
    """
    log("Training global model...")
    dtr_g = xgb.DMatrix(df_tr[ALL_FEATURES].values, label=df_tr["sr"].values, feature_names=ALL_FEATURES)
    dva_g = xgb.DMatrix(df_va[ALL_FEATURES].values, label=df_va["sr"].values, feature_names=ALL_FEATURES)
    m_glob = xgb.train(
        XGB_PARAMS, dtr_g, N_ROUNDS,
        evals=[(dva_g, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )
    log(f"  Global best iter: {m_glob.best_iteration}")

    log("Training cluster models...")
    cluster_models = {}
    cl_tr = df_tr["cluster_train"].values
    cl_va = df_va["cluster_train"].values

    for cid in range(7):
        m_tr = cl_tr == cid
        m_va = cl_va == cid
        n_tr_c = np.sum(m_tr)
        if n_tr_c < 100:
            log(f"  Cluster {cid} skipped (n_train={n_tr_c})")
            continue

        dtr_c = xgb.DMatrix(df_tr[ALL_FEATURES].values[m_tr], label=df_tr["sr"].values[m_tr], feature_names=ALL_FEATURES)
        n_va_c = np.sum(m_va)
        if n_va_c >= 10:
            dva_c = xgb.DMatrix(df_va[ALL_FEATURES].values[m_va], label=df_va["sr"].values[m_va], feature_names=ALL_FEATURES)
        else:
            dva_c = dva_g  # fallback to global val set

        m_c = xgb.train(
            XGB_PARAMS, dtr_c, N_ROUNDS,
            evals=[(dva_c, "val")],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False,
        )
        cluster_models[cid] = m_c
        log(f"  Cluster {cid}: n_train={n_tr_c}, best_iter={m_c.best_iteration}")

    return m_glob, cluster_models


# ---------------------------
# Prediction strategies
# ---------------------------
def predict_global(m_glob, X_te):
    dte = xgb.DMatrix(X_te, feature_names=ALL_FEATURES)
    pred = m_glob.predict(dte, iteration_range=(0, m_glob.best_iteration + 1))
    return np.clip(pred, 0, None)


def predict_cluster(m_glob, cluster_models, X_te, cluster_labels_te):
    """
    Cluster-routed prediction. Uses cluster_labels_te to route each pixel.
    Falls back to global for clusters without a trained model.
    """
    # Start with global predictions as fallback
    pred = predict_global(m_glob, X_te)

    for cid, mod in cluster_models.items():
        mask = cluster_labels_te == cid
        if not np.any(mask):
            continue
        d_c  = xgb.DMatrix(X_te[mask], feature_names=ALL_FEATURES)
        pred[mask] = np.clip(
            mod.predict(d_c, iteration_range=(0, mod.best_iteration + 1)), 0, None
        )
    return pred


# ---------------------------
# Main evaluation loop
# ---------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("=" * 65)
    log("OOD Cluster-Shift Assessment  (Poisson objective)")
    log("=" * 65)

    # 1. Load data
    df, lat_vals, lon_vals = load_full_dataframe()

    # 2. OOD split
    df_tr_pool, df_te = make_ood_split(df)

    # 3. Val split (for early stopping only, from inner zone)
    rng   = np.random.default_rng(RANDOM_STATE)
    idx   = rng.permutation(len(df_tr_pool))
    n_va  = int(len(df_tr_pool) * 0.2)
    df_va = df_tr_pool.iloc[idx[:n_va]].reset_index(drop=True)
    df_tr = df_tr_pool.iloc[idx[n_va:]].reset_index(drop=True)
    log(f"Train: {len(df_tr):,}  Val: {len(df_va):,}  OOD test: {len(df_te):,}")
    del df, df_tr_pool; gc.collect()

    # 4. Pre-compute flat index of test pixels in the full grid
    #    (needed to look up per-period cluster labels)
    n_lon   = len(lon_vals)
    # Nearest-grid-cell index for each test pixel
    lat_idx = np.array([np.argmin(np.abs(lat_vals - la)) for la in df_te["lat"].values])
    lon_idx = np.array([np.argmin(np.abs(lon_vals - lo)) for lo in df_te["lon"].values])
    flat_idx = lat_idx * n_lon + lon_idx

    # Training-period cluster labels for test pixels (for static comparison)
    cl_train_te = df_te["cluster_train"].values
    coords_te   = df_te[["lon", "lat"]].values
    X_te        = df_te[ALL_FEATURES].values
    y_te        = df_te["sr"].values

    # 5. Train models
    m_glob, cluster_models = train_models(df_tr, df_va)
    del df_tr, df_va; gc.collect()

    all_results = []

    # 6. Strategy A — Global
    log("\n--- Strategy: global ---")
    pred_g = predict_global(m_glob, X_te)
    row    = score_predictions(y_te, pred_g, coords_te,
                               {"strategy": "global", "period": "N/A",
                                "n_shifted": 0, "frac_shifted": 0.0})
    all_results.append(row)
    log(f"  R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}  "
        f"pBIAS={row['pBIAS']:.2f}%  MoranI={row['MoranI']:.4f}")

    # 7. Strategy B — Cluster-static (training-period labels for test pixels)
    log("\n--- Strategy: cluster_static ---")
    pred_cs = predict_cluster(m_glob, cluster_models, X_te, cl_train_te)
    row     = score_predictions(y_te, pred_cs, coords_te,
                                {"strategy": "cluster_static", "period": TRAINING_PERIOD,
                                 "n_shifted": 0, "frac_shifted": 0.0})
    all_results.append(row)
    log(f"  R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}  "
        f"pBIAS={row['pBIAS']:.2f}%  MoranI={row['MoranI']:.4f}")

    # 8. Strategy C — Cluster-shift (one run per climate period)
    log("\n--- Strategy: cluster_shift (per period) ---")
    for period in CLIM_SCENARIOS:
        log(f"  Loading cluster labels: {period}")
        try:
            cl_all = load_cluster_labels_for_period(period, lat_vals, lon_vals)
        except Exception as e:
            log(f"  FAILED to load {period}: {e}")
            continue

        cl_period_te = cl_all[flat_idx]

        # Count how many test pixels changed cluster vs training period
        shifted    = cl_period_te != cl_train_te
        n_shifted  = int(np.sum(shifted))
        frac_shift = n_shifted / len(cl_train_te)

        pred_shift = predict_cluster(m_glob, cluster_models, X_te, cl_period_te)
        row        = score_predictions(y_te, pred_shift, coords_te,
                                       {"strategy": "cluster_shift", "period": period,
                                        "n_shifted": n_shifted, "frac_shifted": frac_shift})
        all_results.append(row)
        log(f"  [{period}]  shift_frac={frac_shift:.3f}  "
            f"R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}  "
            f"pBIAS={row['pBIAS']:.2f}%  MoranI={row['MoranI']:.4f}")

        # --- Stratified analysis: shifted vs non-shifted pixels ---
        for label, mask in [("shifted", shifted), ("stable", ~shifted)]:
            if np.sum(mask) < 20:
                continue
            sub_row = score_predictions(
                y_te[mask], pred_shift[mask], coords_te[mask],
                {"strategy": f"cluster_shift_{label}", "period": period,
                 "n_shifted": int(np.sum(shifted)), "frac_shifted": frac_shift}
            )
            all_results.append(sub_row)

        # Also score *global* on the same shifted/stable subsets for fair comparison
        for label, mask in [("shifted", shifted), ("stable", ~shifted)]:
            if np.sum(mask) < 20:
                continue
            sub_row = score_predictions(
                y_te[mask], pred_g[mask], coords_te[mask],
                {"strategy": f"global_{label}", "period": period,
                 "n_shifted": int(np.sum(shifted)), "frac_shifted": frac_shift}
            )
            all_results.append(sub_row)

        gc.collect()

    # 9. Save
    results_df = pd.DataFrame(all_results)
    out_csv    = os.path.join(OUTPUT_DIR, "ood_cluster_shift_results.csv")
    results_df.to_csv(out_csv, index=False)
    log(f"\nResults saved → {out_csv}")

    # 10. Summary
    print_summary(results_df)
    log("Done.")


# ---------------------------
# Summary
# ---------------------------
def print_summary(df):
    print("\n" + "=" * 75)
    print("OOD CLUSTER-SHIFT RESULTS")
    print("=" * 75)

    # Top-level: global vs static vs shift (mean across periods for shift)
    print("\n--- Overall: global vs cluster_static vs cluster_shift (mean across periods) ---")
    top = df[df["strategy"].isin(["global", "cluster_static", "cluster_shift"])].copy()
    agg = (top.groupby("strategy")[["R2", "MAE", "RMSE", "pBIAS", "MoranI"]]
           .mean().reset_index())
    print(agg.to_string(index=False, float_format="{:.4f}".format))

    # Shift strategy per period (sorted chronologically)
    print("\n--- cluster_shift by period (with fraction of pixels that shifted cluster) ---")
    shift = df[df["strategy"] == "cluster_shift"].copy()
    cols  = ["period", "frac_shifted", "R2", "RMSE", "pBIAS", "MoranI"]
    print(shift[cols].to_string(index=False, float_format="{:.4f}".format))

    # Stratified: shifted pixels vs stable pixels
    print("\n--- Shifted vs stable pixels: cluster_shift vs global (mean across periods) ---")
    strat = df[df["strategy"].isin([
        "cluster_shift_shifted", "cluster_shift_stable",
        "global_shifted",        "global_stable"
    ])].copy()
    if not strat.empty:
        agg2 = (strat.groupby("strategy")[["R2", "RMSE", "pBIAS", "MoranI"]]
                .mean().reset_index())
        print(agg2.to_string(index=False, float_format="{:.4f}".format))

    print("=" * 75)


if __name__ == "__main__":
    main()
