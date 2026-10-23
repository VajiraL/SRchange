"""
Historical land-use predictors from HYDE 3.5 change applied to present observations (v3).

Replaces the class x population rank-transfer (v1/v2) and the separate anthrome
scheme for the idealized states. Every state -- intact (10000 BC), pre-industrial
(1700 AD) and the four 30-year periods -- is built the same way: present-day
observations are the anchor, and HYDE 3.5 supplies how each cell differed from
the observation epoch (2015).

  irr_frac, rain_frac   observed x HYDE ratio (t / 2015) of irrigated (rainfed)
                        area, computed on a 0.5 deg support so that observed
                        irrigation is scaled even where HYDE's own 5' map has
                        none; where the 0.5 deg cell has no HYDE irrigated
                        (rainfed) area in 2015, the HYDE cropland ratio is used.
                        No cropland at t (e.g. intact) -> 0. Ratios capped at
                        MAX_RATIO; clipped to [0, 1]; irr + rain <= 1.1.
                        (A first version used the 5' ratio with an additive
                        fallback, which left most observed irrigation untouched
                        back to 10 ka because LGRIP maps irrigation over ~3x the
                        area HYDE does.)
  gHM, forest_age       observed + f(X_t, env) - f(X_2015, env), where f is a
                        gradient-boosted model fitted on present-day cells mapping
                        HYDE drivers X (crop, irrigated, pasture, rangeland, urban
                        fractions, log population density) and the present climate
                        cluster (env) to the observed variable. Holding env at the
                        present isolates the land-use part of the change.
                        Clipped to [0, 1] (gHM) and [0, 300] years (forest age).
                        --fa-mode forest (default) applies the forest-age change
                        only where forest exists today (observed age > 0); 'all'
                        applies it everywhere (first version: invents ages in
                        drylands and steppe).

Absent irrigated/rainfed cropland and absent forest are 0 on land (as in v2).
Period values are means over the HYDE snapshots in the period, end points
half-weighted.

Outputs: data/LU_features_v3/<scenario>.nc (same variables and grid as LU_features)
         data/LU_features_v3/diagnostics.json (model fits, global means)

    python scripts/data-preparation/build_lu_features_v3.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import xgboost as xgb

ROOT = Path("/home/vajira/SRchange")
HYDE = ROOT / "data" / "HYDE35"
PRESENT = ROOT / "data" / "presentLU"
LU_CLASSES = ROOT / "data" / "harmonized_LU_classes.zarr"
CLUSTERS = ROOT / "data" / "climate_clusters.zarr"
OUT = ROOT / "data" / "LU_features_v3"
FA_MODE = "forest"
MONOTONE = False

REF = "2015AD"
SCENARIOS = {   # scenario -> [(HYDE year, weight)]
    "intact": [("10000BC", 1.0)],
    "pre-industrial": [("1700AD", 1.0)],
    "1901_1930": [("1900AD", .5), ("1910AD", 1), ("1920AD", 1), ("1930AD", .5)],
    "1931_1960": [("1930AD", .5), ("1940AD", 1), ("1950AD", 1), ("1960AD", .5)],
    "1961_1990": [("1960AD", .5), ("1970AD", 1), ("1980AD", 1), ("1990AD", .5)],
    "1991_2020": [("1990AD", .5), ("2000AD", 1), ("2010AD", 1), ("2020AD", .5)],
}
DRIVERS = ["crop_frac", "irr_frac", "pasture_frac", "rangeland_frac", "urban_frac", "lpopd"]
MIN_REF = 1e-4        # minimum 0.5 deg HYDE fraction for a ratio
COARSE = 6            # 6 x 5' = 0.5 deg
MAX_RATIO = 10.0
N_FIT = 3_000_000
SEED = 0


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def target_grid():
    ref = xr.open_dataset(PRESENT / "gHM.nc")
    return ref.lat.values.astype("float64"), ref.lon.values.astype("float64")


def hyde_on_grid(year, lat, lon):
    """Nearest HYDE 5' cell for every 0.05 deg cell (a lookup, no interpolation)."""
    ds = xr.open_dataset(HYDE / f"hyde35_{year}.nc")
    hl, ho = ds.lat.values, ds.lon.values
    il = np.clip(np.searchsorted(hl, lat), 1, len(hl) - 1)
    il = np.where(np.abs(hl[il - 1] - lat) < np.abs(hl[il] - lat), il - 1, il)
    io = np.clip(np.searchsorted(ho, lon), 1, len(ho) - 1)
    io = np.where(np.abs(ho[io - 1] - lon) < np.abs(ho[io] - lon), io - 1, io)
    out = {}
    for v in ["crop_frac", "irr_frac", "rainfed_frac", "pasture_frac", "rangeland_frac",
              "urban_frac", "popd"]:
        a = ds[v].values[np.ix_(il, io)]
        out[v] = np.nan_to_num(a, nan=0.0).astype(np.float32)
    out["lpopd"] = np.log1p(out.pop("popd"))
    # 0.5 deg area-weighted fractions (by HYDE cell area), mapped back to the target grid
    w = np.cos(np.deg2rad(hl))[:, None] * np.ones((1, len(ho)))
    ny, nx = len(hl) // COARSE, len(ho) // COARSE
    for v in ("crop_frac", "irr_frac", "rainfed_frac"):
        a = np.nan_to_num(ds[v].values, nan=0.0) * w
        num = a.reshape(ny, COARSE, nx, COARSE).sum((1, 3))
        den = w.reshape(ny, COARSE, nx, COARSE).sum((1, 3))
        c = (num / den).astype(np.float32)
        out[v + "_05"] = c[np.ix_(il // COARSE, io // COARSE)]
    return out


def observations(land):
    obs = {"gHM": xr.open_dataset(PRESENT / "gHM.nc")["ghm"].values.astype(np.float32)}
    for v in ("irr_frac", "rain_frac"):
        obs[v] = xr.open_dataset(PRESENT / f"{v}.nc")[v].values.astype(np.float32)
    obs["forest_age"] = xr.open_dataset(PRESENT / "forest_age.nc")["forest_age"].values.astype(np.float32)
    for v in ("irr_frac", "rain_frac", "forest_age"):          # NaN = absent -> 0 on land
        obs[v] = np.where(land & np.isnan(obs[v]), 0.0, obs[v]).astype(np.float32)
    return obs


def scale_by_ratio(obs, X, Xref, var):
    """Observed fraction x HYDE ratio at 0.5 deg; cropland ratio where `var` is absent."""
    def ratio(t, r):
        return np.where(r >= MIN_REF, np.minimum(t / np.maximum(r, 1e-12), MAX_RATIO), np.nan)
    r = ratio(X[var + "_05"], Xref[var + "_05"])
    rc = ratio(X["crop_frac_05"], Xref["crop_frac_05"])
    r = np.where(np.isfinite(r), r, rc)
    r = np.where(np.isfinite(r), r, 1.0)      # no HYDE cropland in 2015 either: keep observed
    return np.clip(obs * r, 0.0, 1.0)


def fit_delta_model(y, X, env, land, name, diag):
    ok = land & np.isfinite(y) & (env >= 0)
    idx = np.flatnonzero(ok.ravel())
    rng = np.random.default_rng(SEED)
    idx = rng.choice(idx, size=min(N_FIT, idx.size), replace=False)
    feats = np.column_stack([X[d].ravel()[idx] for d in DRIVERS] + [env.ravel()[idx]])
    yy = y.ravel()[idx]
    n = len(idx); cut = int(0.8 * n)
    dtr = xgb.DMatrix(feats[:cut], label=yy[:cut], feature_names=DRIVERS + ["cluster"])
    dva = xgb.DMatrix(feats[cut:], label=yy[cut:], feature_names=DRIVERS + ["cluster"])
    params = {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.9, "tree_method": "hist", "nthread": 64, "seed": SEED}
    if MONOTONE:
        # more human pressure can only raise gHM and only lower forest age;
        # the climate cluster (last feature) is unconstrained
        sign = 1 if name == "gHM" else -1
        params["monotone_constraints"] = "(" + ",".join([str(sign)] * len(DRIVERS) + ["0"]) + ")"
    m = xgb.train(params, dtr, 1500, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
    pred = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    r2 = 1 - np.sum((pred - yy[cut:]) ** 2) / np.sum((yy[cut:] - yy[cut:].mean()) ** 2)
    diag[f"{name}_fit_r2"] = float(r2)
    log(f"  {name}: delta model fitted on {n:,} cells, held-out R2 {r2:.3f}")
    return m


def predict_full(model, X, env, land):
    out = np.full(env.shape, np.nan, dtype=np.float32)
    idx = np.flatnonzero((land & (env >= 0)).ravel())
    for s in range(0, idx.size, 5_000_000):
        j = idx[s:s + 5_000_000]
        feats = np.column_stack([X[d].ravel()[j] for d in DRIVERS] + [env.ravel()[j]])
        out.ravel()[j] = model.predict(xgb.DMatrix(feats, feature_names=DRIVERS + ["cluster"]),
                                       iteration_range=(0, model.best_iteration + 1))
    return out


def main():
    global OUT, FA_MODE, MONOTONE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa-mode", choices=["forest", "all"], default="forest")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--monotone", action="store_true",
                    help="constrain gHM to increase and forest age to decrease with every HYDE driver")
    a = ap.parse_args()
    FA_MODE, OUT, MONOTONE = a.fa_mode, Path(a.out), a.monotone
    log(f"forest-age mode: {FA_MODE}; monotone: {MONOTONE}; output {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    lat, lon = target_grid()
    lu = xr.open_zarr(LU_CLASSES, group="1991_2020")["LU_class"].values
    env = xr.open_zarr(CLUSTERS, group="1991_2020")["cluster"].values.astype(np.float32)
    land = lu > 0
    obs = observations(land)
    diag = {}

    log(f"loading HYDE reference {REF}")
    Xref = hyde_on_grid(REF, lat, lon)
    log("fitting delta models (present-day observations ~ HYDE drivers + climate cluster)")
    models = {"gHM": fit_delta_model(obs["gHM"], Xref, env, land, "gHM", diag),
              "forest_age": fit_delta_model(obs["forest_age"], Xref, env, land, "forest_age", diag)}
    fref = {k: predict_full(m, Xref, env, land) for k, m in models.items()}

    cache = {}
    for scen, years in SCENARIOS.items():
        log(f"scenario {scen}: HYDE {', '.join(y for y, _ in years)}")
        acc = {v: np.zeros(lu.shape, np.float64) for v in ("gHM", "irr_frac", "rain_frac", "forest_age")}
        wsum = 0.0
        for y, w in years:
            if y not in cache:
                cache[y] = hyde_on_grid(y, lat, lon)
            X = cache[y]
            irr = scale_by_ratio(obs["irr_frac"], X, Xref, "irr_frac")
            rain = scale_by_ratio(obs["rain_frac"], X, Xref, "rainfed_frac")
            tot = irr + rain
            over = tot > 1.1
            with np.errstate(invalid="ignore", divide="ignore"):
                irr = np.where(over, irr * 1.1 / tot, irr); rain = np.where(over, rain * 1.1 / tot, rain)
            g = np.clip(obs["gHM"] + predict_full(models["gHM"], X, env, land) - fref["gHM"], 0, 1)
            dfa = predict_full(models["forest_age"], X, env, land) - fref["forest_age"]
            if FA_MODE == "forest":
                dfa = np.where(obs["forest_age"] > 0, dfa, 0.0)
            fa = np.clip(obs["forest_age"] + dfa, 0, 300)
            # cells without a climate cluster keep observed values (no model applies)
            nocl = env < 0
            g = np.where(nocl, obs["gHM"], g); fa = np.where(nocl, obs["forest_age"], fa)
            for v, a in (("gHM", g), ("irr_frac", irr), ("rain_frac", rain), ("forest_age", fa)):
                acc[v] += w * a
            wsum += w
        res = {v: (a / wsum).astype(np.float32) for v, a in acc.items()}
        for v in res:                                        # keep the observation's domain
            res[v] = np.where(np.isfinite(obs[v]) & land, res[v], np.nan).astype(np.float32)
        ds = xr.Dataset({v: (("lat", "lon"), res[v]) for v in ("gHM", "irr_frac", "rain_frac", "forest_age")},
                        coords={"lat": lat, "lon": lon},
                        attrs={"method": "HYDE 3.5 change applied to present observations (v3)",
                               "hyde_years": ",".join(y for y, _ in years), "reference": REF,
                               "script": "scripts/data-preparation/build_lu_features_v3.py"})
        ds.to_netcdf(OUT / f"{scen}.nc", encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
        wcos = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
        m = land & np.isfinite(res["gHM"])
        diag[scen] = {v: float(np.average(res[v][m & np.isfinite(res[v])], weights=wcos[m & np.isfinite(res[v])]))
                      for v in res}
        log(f"  wrote {scen}.nc  " + ", ".join(f"{v} {x:.4f}" for v, x in diag[scen].items()))
    json.dump(diag, open(OUT / "diagnostics.json", "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
