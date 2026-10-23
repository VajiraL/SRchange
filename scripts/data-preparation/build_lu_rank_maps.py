"""
Build per-cell rank maps consistent with the fitted land-use distributions.

Each cell's rank is the CDF of its observed present-day value under the mixture
model of the bin it occupies in the present day. Sampling a past bin's mixture at
that rank (estimate_lu_scenario_features.py) then transfers the cell's relative
position within its environmental class through time.

Replaces data/statistical_models/{feat}_rank_map.nc (5 May 2026), which were
computed under an earlier anthrome x k5-cluster binning and were NaN on 29 % of
land for gHM and 47 % for irrigated/rainfed fraction; the sampler turned NaN
ranks into zeros in every scenario. See RootZoneChange notes
(lu_synth_bias) for the diagnosis.

Two rank maps per feature, one for each model family:
  standard  : HILDA+ LU class x climate cluster x population bin (2014), fitted
              in beta_mixture_models.json -> historical and future scenarios
  anthrome  : anthrome (2015) x climate cluster, fitted in
              anthrome_only_mixture_models.json -> intact and pre-industrial

Point masses get the midpoint of their probability mass (zero: w0/2; one, for
forest age: 1 - w1/2), so the sampler maps them back to 0 or 1 in the present bin.
Cells whose bin has no fitted model, or whose observation is missing, get NaN;
the sampler keeps the present-day value there.
"""

import json
import os
from datetime import datetime

import numpy as np
import xarray as xr
from scipy import stats

ROOT_DIR = "/home/vajira/SRchange"
DATA_PRESENT = f"{ROOT_DIR}/data/presentLU"
DATA_LU = f"{ROOT_DIR}/data/harmonized_LU_classes.zarr"
DATA_CLUSTER = f"{ROOT_DIR}/data/climate_clusters.zarr"
DATA_POP = f"{ROOT_DIR}/data/population"
DATA_ANTHROME = f"{ROOT_DIR}/data/anthromes/anthrome_2015.nc"
DATA_MODELS = f"{ROOT_DIR}/data/statistical_models_v2"   # refit by train_lu_distributions.py (v2)
OUTPUT_DIR = f"{ROOT_DIR}/data/statistical_models_v2"

FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
NAN_IS_ZERO = ['irr_frac', 'rain_frac', 'forest_age']   # NaN = absent; 0 on land, as in fitting


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_obs(feat):
    """Present-day observation, scaled exactly as in train_lu_distributions.py."""
    if feat == 'gHM':
        return xr.open_dataset(f"{DATA_PRESENT}/gHM.nc")['ghm'].values
    if feat == 'forest_age':
        v = xr.open_dataset(f"{DATA_PRESENT}/forest_age.nc")['forest_age'].values
        return np.clip(v / 300.0, 0, 1)
    return xr.open_dataset(f"{DATA_PRESENT}/{feat}.nc")[feat].values


def mixture_cdf(model, x):
    """CDF of the zero/one point-mass + beta mixture used by the sampler."""
    w0 = model['weight_zero']
    w1 = model.get('weight_one', 0.0)
    r = np.full(x.shape, np.nan, dtype=np.float64)
    zero = x <= 1e-6
    one = (x >= 1.0 - 1e-6) & (w1 > 0)
    mid = ~(zero | one)
    r[zero] = w0 / 2.0
    r[one] = 1.0 - w1 / 2.0
    if np.any(mid):
        if model['beta_params'] is not None:
            a, b = model['beta_params']
            xc = np.clip(x[mid], 1e-6, 1 - 1e-6)
            r[mid] = w0 + (1.0 - w0 - w1) * stats.beta.cdf(xc, a, b)
        else:
            # no beta fitted (fewer than 10 non-zero values): middle of the
            # continuous mass; the sampler returns 0.5 there
            r[mid] = w0 + 0.5 * (1.0 - w0 - w1)
    return r


def rank_map(obs, bin_ids, models):
    """Rank of every cell under the model of its own bin; NaN where unavailable."""
    rank = np.full(obs.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(obs)
    for key, mask in bin_ids:
        if key not in models:
            continue
        m = mask & valid
        if np.any(m):
            rank[m] = mixture_cdf(models[key], obs[m])
    return rank


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("Loading present-day conditioning layers...")
    ref = xr.open_dataset(f"{DATA_PRESENT}/gHM.nc")
    lat, lon = ref.lat, ref.lon
    lu = xr.open_zarr(DATA_LU, group='1991_2020')['LU_class'].values
    cl = xr.open_zarr(DATA_CLUSTER, group='1991_2020')['cluster'].values
    pop = (xr.open_zarr(f"{DATA_POP}/population_histsoc_30arcmin_annual_1901_2014_binned.zarr")
           ['pop_bin'].sel(time='2014-01-01', method='nearest').values)
    ant = xr.open_dataset(DATA_ANTHROME)['anthrome'].values

    # bin masks, built once (same keys as train_lu_distributions.py)
    std_bins = []
    for l in np.unique(lu[lu > 0]):
        for c in range(6):
            for p in range(10):
                m = (lu == l) & (cl == c) & (pop == p)
                if np.any(m):
                    std_bins.append((f"{int(l)}_{c}_{p}", m))
    ant_bins = []
    for c in range(6):
        for a in np.unique(ant[ant > 0]):
            m = (cl == c) & (ant == a)
            if np.any(m):
                ant_bins.append((f"{c}_{int(a)}", m))
    log(f"{len(std_bins)} standard bins, {len(ant_bins)} anthrome bins")

    std_models = json.load(open(f"{DATA_MODELS}/beta_mixture_models.json"))
    ant_models = json.load(open(f"{DATA_MODELS}/anthrome_only_mixture_models.json"))

    for feat in FEATURES:
        obs = load_obs(feat)
        if feat in NAN_IS_ZERO:
            obs = np.where((lu > 0) & np.isnan(obs), 0.0, obs)
        for family, bins, models in (("standard", std_bins, std_models[feat]),
                                     ("anthrome", ant_bins, ant_models[feat])):
            r = rank_map(obs, bins, models)
            land = np.isfinite(obs) & (lu > 0)
            log(f"{feat:10} {family:9}: rank defined on {100*np.mean(np.isfinite(r[land])):.1f} % "
                f"of land cells with an observation")
            xr.Dataset({f"{feat}_rank": (("lat", "lon"), r)},
                       coords={"lat": lat, "lon": lon},
                       attrs={"description": f"CDF of observed present-day {feat} under the "
                                             f"{family} mixture model of the cell's present-day bin",
                              "source_script": "scripts/data-preparation/build_lu_rank_maps.py"}
                       ).to_netcdf(f"{OUTPUT_DIR}/{feat}_rank_map_{family}.nc")
    log(f"Saved rank maps to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
