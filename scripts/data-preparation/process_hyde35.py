"""
Convert HYDE 3.5 (Klein Goldewijk et al.) yearly archives into NetCDF.

Input : data-raw/HYDE3.5/<year>_lu.zip and <year>_pop.zip (ESRI ASCII grids,
        5 arcmin, land-use layers in km2 per cell, popd in persons/km2)
Output: data/HYDE35/hyde35_<year>.nc on the native 5 arcmin grid, with
          crop_frac, irr_frac, rainfed_frac, pasture_frac, rangeland_frac,
          grazing_frac, urban_frac  -- fractions of the cell area
          popd                                      -- persons per km2
Cell area is computed on the sphere (authalic radius); fractions are clipped
to [0, 1] to absorb rounding in the source.

    python scripts/data-preparation/process_hyde35.py [YEAR ...]
(default: every <year>_lu.zip found)
"""

import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/home/vajira/SRchange")
RAW = ROOT / "data-raw" / "HYDE3.5"
OUT = ROOT / "data" / "HYDE35"
R_EARTH = 6371.0072          # km, authalic

LU_LAYERS = {                 # output name -> HYDE file stem (area, km2)
    "crop_frac": "cropland",
    "irr_frac": "tot_irri",
    "rainfed_frac": "tot_rainfed",
    "pasture_frac": "pasture",
    "rangeland_frac": "rangeland",
    "grazing_frac": "grazing",
}
POP_LAYERS = {"popd": "popd_", "urban_area": "uopp_"}   # uopp: urban area, km2


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def read_asc(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        header = {}
        for _ in range(6):
            k, v = f.readline().decode().split()
            header[k.lower()] = float(v)
        a = np.loadtxt(f, dtype=np.float32)
    a[a == header["nodata_value"]] = np.nan
    return a, header


def cell_area_km2(nrows, ncols, cellsize, yll):
    lat_edges = yll + cellsize * np.arange(nrows + 1)
    band = (R_EARTH ** 2) * np.deg2rad(cellsize) * np.abs(
        np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1])))
    return np.repeat(band[::-1, None], ncols, axis=1)        # rows N -> S


def process(year: str):
    lu_zip, pop_zip = RAW / f"{year}_lu.zip", RAW / f"{year}_pop.zip"
    if not lu_zip.exists() or not pop_zip.exists():
        log(f"{year}: missing archive(s), skipped")
        return
    data, hdr = {}, None
    with zipfile.ZipFile(lu_zip) as z:
        names = z.namelist()
        for out, stem in LU_LAYERS.items():
            n = next(x for x in names if x.startswith(stem) and x.endswith(".asc"))
            data[out], hdr = read_asc(z, n)
    with zipfile.ZipFile(pop_zip) as z:
        names = z.namelist()
        for out, stem in POP_LAYERS.items():
            n = next(x for x in names if x.startswith(stem) and x.endswith(".asc"))
            data[out], _ = read_asc(z, n)
    nr, nc, cs = int(hdr["nrows"]), int(hdr["ncols"]), hdr["cellsize"]
    area = cell_area_km2(nr, nc, cs, hdr["yllcorner"])
    for k in list(LU_LAYERS) + ["urban_area"]:
        frac = np.clip(data.pop(k) / area, 0.0, 1.0)
        data["urban_frac" if k == "urban_area" else k] = frac.astype(np.float32)
    lat = hdr["yllcorner"] + cs * (np.arange(nr)[::-1] + 0.5)
    lon = hdr["xllcorner"] + cs * (np.arange(nc) + 0.5)
    ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in data.items()},
                    coords={"lat": lat, "lon": lon},
                    attrs={"source": "HYDE 3.5 (gbc2025_7apr_base)", "year": year,
                           "script": "scripts/data-preparation/process_hyde35.py"})
    ds = ds.sortby("lat")
    OUT.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(OUT / f"hyde35_{year}.nc", encoding=enc)
    land = np.isfinite(ds.crop_frac)
    w = np.cos(np.deg2rad(ds.lat))
    s = {k: float(ds[k].where(land).weighted(w).mean()) for k in ("crop_frac", "irr_frac", "grazing_frac", "urban_frac")}
    log(f"{year}: wrote hyde35_{year}.nc; land means " + ", ".join(f"{k} {v:.4f}" for k, v in s.items()))


def main():
    years = sys.argv[1:] or sorted(p.name.split("_")[0] for p in RAW.glob("*_lu.zip"))
    for y in years:
        process(y)


if __name__ == "__main__":
    main()
