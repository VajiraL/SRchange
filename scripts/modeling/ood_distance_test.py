"""
Fair OOD Test: Performance vs Pixel-wise Climate Shift Distance
===============================================================
For each target climate period (past + future), computes the per-pixel
Mahalanobis distance of the climate feature shift from the training period
(1991-2020), then evaluates four model configurations stratified by shift
magnitude:

    Configs:
      A. global   + poisson
      B. global   + absoluteerror
      C. clustered + poisson        (cluster labels from TARGET period)
      D. clustered + absoluteerror  (cluster labels from TARGET period)

Stratification: shift distances are binned into quartiles (Q1-Q4) across
ALL periods combined, giving a common distance axis. Results are reported
per stratum × config × period, and aggregated across periods within strata.

Distance metric: Mahalanobis on the full CLIMATE_FEATURES vector
(covariance estimated from the training-period pixel distribution).
This accounts for feature correlations and scale differences.

Ground truth: the SR target (1991-2020) is used for ALL evaluations.
This is the only available ground truth. The interpretation is explicitly
framed as: "given that a pixel's climate has shifted by distance D from
the training distribution, how well can each config predict its
present-day SR from its new climate inputs?" This is a clean test of
extrapolation skill — not a test of whether future SR predictions are
correct (which is unknowable).
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
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.covariance import LedoitWolf  # robust covariance for high-dim data

warnings.filterwarnings("ignore")

# ---------------------------
# Configuration
# ---------------------------
TRAIN_SAMPLE   = 800_000   # pixels drawn from training period for model fitting
VAL_FRAC       = 0.05      # fraction of train sample held out for early stopping
EVAL_SAMPLE    = 500_000   # pixels drawn per period for evaluation (if period has more)
MORAN_SAMPLES  = 50_000
CORES          = 128
RANDOM_STATE   = 42

DIR_ROOT                 = "/home/vajira/SRchange"
SR_TARGET_PATH           = os.path.join(DIR_ROOT, "data/SR-target/merged_SR.nc")
CLIMATE_ZARR             = os.path.join(DIR_ROOT, "data/climate_features.zarr")
CLIMATE_CLUSTER_ZARR     = os.path.join(DIR_ROOT, "data/climate_clusters.zarr")
TERRAIN_FILE             = os.path.join(DIR_ROOT, "data/terrain_hand_0p05.nc")
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, "data/LU_features_2")
OUTPUT_DIR               = os.path.join(DIR_ROOT, "output_SR/ood_distance_test")
LOG_FILE                 = os.path.join(OUTPUT_DIR, "ood_distance.log")

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
ALL_FEATURES     = STATIC_FEATURES + CLIMATE_FEATURES + SCENARIO_LU_FEATURES
TRAINING_PERIOD  = "1991_2020"

# All periods to evaluate (past + future)
# Format mirrors get_clim_group() logic from production script
TARGET_PERIODS = [
    "1901_1930", "1931_1960", "1961_1990",           # past
    "2041_2070_ssp126", "2041_2070_ssp245",           # mid-century
    "2041_2070_ssp370", "2041_2070_ssp460",
    "2071_2099_ssp126", "2071_2099_ssp245",           # end-century
    "2071_2099_ssp370", "2071_2099_ssp460",
]

OBJECTIVES = [
    ("poisson",      "count:poisson",    {}),
    ("absoluteerror","reg:absoluteerror",{}),
]

N_ROUNDS       = 2000
EARLY_STOPPING = 100
N_STRATA       = 4          # quartile bins on shift distance
MIN_STRATUM_N  = 200        # minimum pixels per stratum to report metrics

BASE_XGB = {
    "max_depth":        5,
    "eta":              0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "nthread":          CORES,
    "device":           "cpu",
    "eval_metric":      "rmse",
}

# ---------------------------
# Helpers
# ---------------------------
def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")


def get_clim_group(name):
    if "ssp" in name:
        parts = name.split("_")
        return f"{parts[0]}_{parts[1]}/{parts[2]}"
    return name


def compute_metrics(y_true, y_pred):
    if len(y_true) < 5:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "pBIAS": np.nan}
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "pBIAS": pbias}


def calculate_morans_i(coords, residuals, n_samples=MORAN_SAMPLES):
    if len(residuals) < 10:
        return np.nan
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


def score(y_true, y_pred, coords, extra):
    m = compute_metrics(y_true, y_pred)
    m["MoranI"] = calculate_morans_i(coords, y_true - y_pred)
    m["n"] = len(y_true)
    m.update(extra)
    return m


# ---------------------------
# Data loading
# ---------------------------
def load_base(lat_vals, lon_vals):
    """Load terrain + training-period climate + LU + SR target + cluster labels."""
    log("  terrain...")
    ds  = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    ter = ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values.squeeze()
    if ter.ndim == 4: ter = ter[:, 0, :, :]

    log(f"  climate ({TRAINING_PERIOD})...")
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group=TRAINING_PERIOD)[CLIMATE_FEATURES]
    clim    = clim_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]

    log(f"  LU ({TRAINING_PERIOD})...")
    lu_ds = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{TRAINING_PERIOD}.nc"))
    lu    = lu_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]

    log(f"  cluster labels ({TRAINING_PERIOD})...")
    cl_da    = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=TRAINING_PERIOD)["cluster"]
    clusters = cl_da.sel(lat=lat_vals, lon=lon_vals, method="nearest").values.flatten()

    return ter, clim, lu, clusters


def load_period_climate(period, lat_vals, lon_vals):
    group   = get_clim_group(period)
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group=group)[CLIMATE_FEATURES]
    clim    = clim_ds.sel(lat=lat_vals, lon=lon_vals, method="nearest").to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    return clim   # shape: (n_clim_features, n_lat, n_lon)


def load_period_clusters(period, lat_vals, lon_vals):
    group    = get_clim_group(period)
    cl_da    = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=group)["cluster"]
    clusters = cl_da.sel(lat=lat_vals, lon=lon_vals, method="nearest").values.flatten()
    return clusters


# ---------------------------
# Climate distance computation
# ---------------------------
def fit_covariance(clim_train_flat):
    """
    Fit a robust covariance matrix (Ledoit-Wolf shrinkage) on the
    training-period climate features. Used for Mahalanobis distance.
    clim_train_flat: (n_pixels, n_clim_features)
    """
    log("  Fitting Ledoit-Wolf covariance on training climate...")
    lw = LedoitWolf(assume_centered=False)
    lw.fit(clim_train_flat)
    # Precision matrix (inverse covariance)
    VI = lw.precision_
    mu = lw.location_
    log(f"  Covariance fitted (shape {VI.shape})")
    return mu, VI


def compute_mahalanobis_shift(clim_train_flat, clim_target_flat, mu_train, VI):
    """
    For each pixel, compute Mahalanobis distance of the SHIFT vector
    (clim_target - clim_train) from zero, using the training covariance.

    shift_i = clim_target_i - clim_train_i
    dist_i  = sqrt(shift_i @ VI @ shift_i)

    This measures how large the climate change is relative to the
    natural variability structure of the training distribution.

    Returns: (n_pixels,) array of distances
    """
    shift = clim_target_flat - clim_train_flat   # (n_pixels, n_clim_feat)
    # Vectorised: dist_i = sqrt(shift_i @ VI @ shift_i.T)
    # For large n, do this in chunks to avoid memory issues
    n      = shift.shape[0]
    chunk  = 50_000
    dist   = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        s       = shift[start:start+chunk]           # (c, p)
        tmp     = s @ VI                              # (c, p)
        dist[start:start+chunk] = np.sqrt(np.einsum("ij,ij->i", tmp, s))
    return dist


# ---------------------------
# Model training
# ---------------------------
def train_models(df_tr, df_va, objective, extra_params):
    """Train global + cluster models. Returns (global_model, cluster_models_dict)."""
    params = {**BASE_XGB, "objective": objective, **extra_params}

    dtr_g = xgb.DMatrix(df_tr[ALL_FEATURES].values, label=df_tr["sr"].values, feature_names=ALL_FEATURES)
    dva_g = xgb.DMatrix(df_va[ALL_FEATURES].values, label=df_va["sr"].values, feature_names=ALL_FEATURES)
    m_g   = xgb.train(params, dtr_g, N_ROUNDS, evals=[(dva_g, "val")],
                      early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
    log(f"    global best_iter={m_g.best_iteration}")

    cluster_models = {}
    cl_tr = df_tr["cluster_train"].values
    cl_va = df_va["cluster_train"].values
    for cid in range(7):
        m_tr = cl_tr == cid
        if np.sum(m_tr) < 100:
            continue
        dtr_c = xgb.DMatrix(df_tr[ALL_FEATURES].values[m_tr],
                             label=df_tr["sr"].values[m_tr], feature_names=ALL_FEATURES)
        m_va  = cl_va == cid
        dva_c = (xgb.DMatrix(df_va[ALL_FEATURES].values[m_va],
                              label=df_va["sr"].values[m_va], feature_names=ALL_FEATURES)
                 if np.sum(m_va) >= 10 else dva_g)
        m_c   = xgb.train(params, dtr_c, N_ROUNDS, evals=[(dva_c, "val")],
                           early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
        cluster_models[cid] = m_c
    log(f"    cluster models trained: {sorted(cluster_models.keys())}")
    return m_g, cluster_models


# ---------------------------
# Prediction
# ---------------------------
def predict_global(m_g, X):
    d = xgb.DMatrix(X, feature_names=ALL_FEATURES)
    return np.clip(m_g.predict(d, iteration_range=(0, m_g.best_iteration + 1)), 0, None)


def predict_clustered(m_g, cluster_models, X, cluster_labels):
    pred = predict_global(m_g, X)   # global as fallback
    for cid, mod in cluster_models.items():
        mask = cluster_labels == cid
        if not np.any(mask):
            continue
        d_c  = xgb.DMatrix(X[mask], feature_names=ALL_FEATURES)
        pred[mask] = np.clip(
            mod.predict(d_c, iteration_range=(0, mod.best_iteration + 1)), 0, None)
    return pred


# ---------------------------
# Stratum scoring
# ---------------------------
def score_by_stratum(y_true, preds_dict, coords, distances, strata_edges, period, obj_label):
    """
    Score each prediction config within each distance stratum.
    preds_dict: {config_label: pred_array}
    strata_edges: (N_STRATA+1,) quantile boundaries
    Returns list of metric dicts.
    """
    rows = []
    strata_labels = [f"Q{i+1}" for i in range(N_STRATA)]

    for si in range(N_STRATA):
        lo = strata_edges[si]
        hi = strata_edges[si + 1]
        # inclusive upper bound for last stratum
        if si < N_STRATA - 1:
            mask = (distances >= lo) & (distances < hi)
        else:
            mask = (distances >= lo) & (distances <= hi)

        n = np.sum(mask)
        if n < MIN_STRATUM_N:
            continue

        for cfg_label, pred in preds_dict.items():
            row = score(
                y_true[mask], pred[mask], coords[mask],
                {"period": period, "objective": obj_label,
                 "config": cfg_label, "stratum": strata_labels[si],
                 "dist_lo": lo, "dist_hi": hi,
                 "dist_mean": float(distances[mask].mean())}
            )
            rows.append(row)

    # Also score the full period (all strata combined)
    for cfg_label, pred in preds_dict.items():
        row = score(
            y_true, pred, coords,
            {"period": period, "objective": obj_label,
             "config": cfg_label, "stratum": "ALL",
             "dist_lo": strata_edges[0], "dist_hi": strata_edges[-1],
             "dist_mean": float(distances.mean())}
        )
        rows.append(row)

    return rows


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("=" * 70)
    log("Fair OOD Test: Climate-Shift Distance Stratified Evaluation")
    log("=" * 70)

    # 1. Load SR target grid
    log("Loading SR target...")
    sr_target          = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    y_full             = sr_target.values.flatten()
    lat_flat           = np.repeat(lat_vals, len(lon_vals))
    lon_flat           = np.tile(lon_vals, len(lat_vals))

    # 2. Load base period data
    log("Loading base period features...")
    terrain, clim_train, lu_train, clusters_train = load_base(lat_vals, lon_vals)

    # Flat arrays for all pixels
    X_full_train = np.concatenate([terrain, clim_train, lu_train], axis=0)\
                     .reshape(len(ALL_FEATURES), -1).T   # (n_pix, n_feat)
    clim_train_flat = clim_train.reshape(len(CLIMATE_FEATURES), -1).T  # (n_pix, n_clim)

    # Valid pixel mask (must have SR > 0 and no NaN features)
    valid = (~np.isnan(y_full)) & (y_full > 0) & (~np.isnan(X_full_train).any(axis=1))
    log(f"Valid pixels: {valid.sum():,} / {len(valid):,}")

    # 3. Build training DataFrame
    df_all = pd.DataFrame(X_full_train[valid], columns=ALL_FEATURES)
    df_all["sr"]            = y_full[valid]
    df_all["lat"]           = lat_flat[valid]
    df_all["lon"]           = lon_flat[valid]
    df_all["cluster_train"] = clusters_train[valid].astype(float)
    df_all["clim_flat_idx"] = np.where(valid)[0]   # original flat index

    # Sample training set from valid pixels
    n_tr_raw = min(TRAIN_SAMPLE, len(df_all))
    rng      = np.random.default_rng(RANDOM_STATE)
    tr_idx   = rng.choice(len(df_all), n_tr_raw, replace=False)
    va_n     = int(n_tr_raw * VAL_FRAC)
    df_tr    = df_all.iloc[tr_idx[va_n:]].reset_index(drop=True)
    df_va    = df_all.iloc[tr_idx[:va_n]].reset_index(drop=True)
    log(f"Train: {len(df_tr):,}  Val: {len(df_va):,}")

    # 4. Fit covariance on training-period climate (ALL valid pixels, not just sample)
    mu_train, VI = fit_covariance(clim_train_flat[valid])

    # 5. Train all model configs
    trained_models = {}   # (obj_label) -> (m_glob, cluster_models)
    for obj_label, objective, extra in OBJECTIVES:
        log(f"\nTraining [{obj_label}]...")
        m_g, cm = train_models(df_tr, df_va, objective, extra)
        trained_models[obj_label] = (m_g, cm)
        gc.collect()

    # 6. Pre-compute shift distances for ALL target periods
    #    so we can set global quantile boundaries across all periods
    log("\nComputing shift distances for all periods...")
    period_distances = {}   # period -> (n_valid_pix,) distance array
    period_clim      = {}   # period -> clim flat array (n_valid_pix, n_clim)
    period_clusters  = {}   # period -> cluster labels (n_valid_pix,)

    for period in TARGET_PERIODS:
        log(f"  {period}")
        clim_tgt = load_period_climate(period, lat_vals, lon_vals)
        clim_tgt_flat = clim_tgt.reshape(len(CLIMATE_FEATURES), -1).T   # (n_pix, n_clim)

        # Keep only valid pixels
        clim_tgt_valid = clim_tgt_flat[valid]
        # Replace NaN in target climate with training value (conservative: zero shift for missing)
        nan_mask = np.isnan(clim_tgt_valid).any(axis=1)
        clim_tgt_valid[nan_mask] = clim_train_flat[valid][nan_mask]

        dists = compute_mahalanobis_shift(clim_train_flat[valid], clim_tgt_valid, mu_train, VI)
        period_distances[period] = dists
        period_clim[period]      = clim_tgt_valid

        cl_tgt = load_period_clusters(period, lat_vals, lon_vals)
        period_clusters[period] = cl_tgt[valid]
        gc.collect()

    # 7. Compute global quantile boundaries across all periods combined
    all_dists_combined = np.concatenate(list(period_distances.values()))
    strata_edges = np.quantile(all_dists_combined,
                               np.linspace(0, 1, N_STRATA + 1))
    log(f"\nGlobal stratum boundaries (Mahalanobis):")
    for i in range(N_STRATA):
        log(f"  Q{i+1}: [{strata_edges[i]:.3f}, {strata_edges[i+1]:.3f}]")

    # Save distance distribution summary
    dist_summary = pd.DataFrame({
        "period": TARGET_PERIODS,
        "dist_mean":   [period_distances[p].mean()   for p in TARGET_PERIODS],
        "dist_median": [np.median(period_distances[p]) for p in TARGET_PERIODS],
        "dist_p75":    [np.percentile(period_distances[p], 75) for p in TARGET_PERIODS],
        "dist_p90":    [np.percentile(period_distances[p], 90) for p in TARGET_PERIODS],
        "dist_max":    [period_distances[p].max()    for p in TARGET_PERIODS],
    })
    dist_summary.to_csv(os.path.join(OUTPUT_DIR, "distance_summary.csv"), index=False)
    log("\nDistance summary:")
    log(dist_summary.to_string(index=False, float_format="{:.3f}".format))

    # 8. Evaluate all configs × periods × strata
    log("\nEvaluating all configs across periods and strata...")
    all_results = []

    # We evaluate on ALL valid pixels for each period
    # (or a large random sample if too many for memory)
    y_eval   = df_all["sr"].values
    coords_eval = df_all[["lon", "lat"]].values

    for period in TARGET_PERIODS:
        log(f"\n  Period: {period}")
        dists       = period_distances[period]
        clim_target = period_clim[period]       # (n_valid, n_clim)
        cl_target   = period_clusters[period]   # (n_valid,)

        # Subsample for evaluation if needed
        n_avail = len(y_eval)
        if n_avail > EVAL_SAMPLE:
            eval_idx    = rng.choice(n_avail, EVAL_SAMPLE, replace=False)
        else:
            eval_idx    = np.arange(n_avail)

        y_e      = y_eval[eval_idx]
        coords_e = coords_eval[eval_idx]
        dists_e  = dists[eval_idx]
        cl_e     = cl_target[eval_idx]

        # Build feature matrix for this period:
        # terrain + LU stay as training period (same as production for past periods
        # where no alternative LU is used; for future, this is a deliberate choice
        # to isolate the effect of climate shift alone)
        X_base_e = df_all[ALL_FEATURES].values[eval_idx].copy()
        # Replace only the climate feature columns with target-period values
        clim_idx_start = len(STATIC_FEATURES)
        clim_idx_end   = clim_idx_start + len(CLIMATE_FEATURES)
        X_base_e[:, clim_idx_start:clim_idx_end] = clim_target[eval_idx]

        preds_dict = {}
        for obj_label, (m_g, cm) in trained_models.items():
            # Global
            preds_dict[f"global_{obj_label}"] = predict_global(m_g, X_base_e)
            # Clustered (target-period cluster labels = shift design)
            preds_dict[f"clustered_{obj_label}"] = predict_clustered(
                m_g, cm, X_base_e, cl_e)

        rows = score_by_stratum(
            y_e, preds_dict, coords_e, dists_e, strata_edges, period, "all"
        )
        # Also tag individual objectives for cleaner downstream analysis
        for obj_label, _, _ in OBJECTIVES:
            preds_obj = {
                "global":    preds_dict[f"global_{obj_label}"],
                "clustered": preds_dict[f"clustered_{obj_label}"],
            }
            rows_obj = score_by_stratum(
                y_e, preds_obj, coords_e, dists_e, strata_edges, period, obj_label
            )
            all_results.extend(rows_obj)

        log(f"    scored {len(rows_obj)} rows")
        gc.collect()

    # 9. Save full results
    results_df = pd.DataFrame(all_results)
    out_csv    = os.path.join(OUTPUT_DIR, "ood_distance_results.csv")
    results_df.to_csv(out_csv, index=False)
    log(f"\nResults saved → {out_csv}")

    # 10. Summary tables
    print_summary(results_df, strata_edges)
    log("Done.")


# ---------------------------
# Summary
# ---------------------------
def print_summary(df, strata_edges):
    print("\n" + "=" * 80)
    print("OOD DISTANCE-STRATIFIED RESULTS")
    print("=" * 80)

    strata_labels = [f"Q{i+1}" for i in range(N_STRATA)] + ["ALL"]

    for obj_label, _, _ in OBJECTIVES:
        print(f"\n{'─'*70}")
        print(f"Objective: {obj_label}")
        sub = df[df["objective"] == obj_label]

        print("\n  -- Mean R² across periods, by config and stratum --")
        pivot = (sub[sub["stratum"].isin(strata_labels)]
                 .groupby(["config", "stratum"])["R2"]
                 .mean()
                 .unstack("stratum")
                 .reindex(columns=strata_labels))
        print(pivot.round(4).to_string())

        print("\n  -- Mean pBIAS across periods, by config and stratum --")
        pivot_b = (sub[sub["stratum"].isin(strata_labels)]
                   .groupby(["config", "stratum"])["pBIAS"]
                   .mean()
                   .unstack("stratum")
                   .reindex(columns=strata_labels))
        print(pivot_b.round(2).to_string())

    # Cross-objective global model comparison by stratum
    print(f"\n{'─'*70}")
    print("Global model: poisson vs absoluteerror — R² by stratum (mean across periods)")
    gl = df[(df["config"] == "global") & (df["stratum"].isin(strata_labels))]
    piv = (gl.groupby(["objective", "stratum"])["R2"]
             .mean()
             .unstack("stratum")
             .reindex(columns=strata_labels))
    print(piv.round(4).to_string())

    # Per-period distance summary with best config
    print(f"\n{'─'*70}")
    print("Per-period ALL-stratum results (R²) — which config wins?")
    best = (df[df["stratum"] == "ALL"]
            .sort_values("R2", ascending=False)
            .groupby(["period", "objective"])
            .first()[["config", "R2", "RMSE", "pBIAS", "dist_mean"]]
            .reset_index())
    print(best.to_string(index=False, float_format="{:.4f}".format))

    print("=" * 80)


if __name__ == "__main__":
    main()
