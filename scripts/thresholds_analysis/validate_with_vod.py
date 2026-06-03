import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
VOD_ZARR = os.path.join(DIR_ROOT, 'data/VODCA_CXKu_monthly.zarr')
PROB_DIR = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/scenarios_probabilistic')
OUTPUT_DIR = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/validation_vod')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg: str):
    print(f"[LOG] {msg}", flush=True)

# ---------------------------
# 2. Loading Data
# ---------------------------
def main():
    log("Loading VODCA data...")
    ds_vod = xr.open_zarr(VOD_ZARR)
    
    # Calculate Interannual Variability (CV) of VOD
    # We use monthly data, so let's calculate CV of annual means to focus on interannual scale
    log("Calculating VOD Interannual Variability (CV)...")
    vod_annual = ds_vod.vod.resample(time='YE').mean()
    vod_mean = vod_annual.mean('time')
    vod_std = vod_annual.std('time')
    # Avoid divide-by-zero runtime warnings
    vod_cv = xr.where(vod_mean != 0, vod_std / vod_mean, np.nan)
    
    # Filter out extreme outliers and non-land
    vod_cv = vod_cv.where((vod_cv > 0) & (vod_cv < 1))

    # Make VOD CV eager so grid alignment inside the loop is fast.
    vod_cv = vod_cv.compute()

    # Load Psat maps for 1991-2020 (this should be a small number of files)
    log("Loading Psat maps...")
    prob_files = glob.glob(os.path.join(PROB_DIR, "*_1991_2020.nc"))
    if not prob_files:
        raise FileNotFoundError(f"No probability maps found in {PROB_DIR} for 1991-2020")

    # ---------------------------
    # 3. Binned analysis (per interaction surface)
    # ---------------------------
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    results = []

    for pf in sorted(prob_files):
        fname = os.path.basename(pf)
        # Extract metadata from filename (same scheme as validate_with_sif.py)
        # prob_sat_c{cid}_{f1}_{f2}_1991_2020.nc
        parts = fname.replace(".nc", "").split("_")
        cid = int(parts[2][1:]) if parts[2].startswith('c') else int(parts[2])
        f1 = parts[3] + "_" + parts[4]
        f2 = parts[5] + "_" + parts[6] if len(parts) > 8 else parts[5]

        log(f"Validating Cluster {cid}: {f1} x {f2}")
        prob_ds = xr.open_dataset(pf)
        var_name = list(prob_ds.data_vars)[0]
        prob = prob_ds[var_name].load()

        # Align VOD CV to this probability map grid
        vod_cv_aligned = vod_cv.reindex_like(prob, method='nearest')

        prob_vals = prob.values
        vod_vals = vod_cv_aligned.values
        mask = (~np.isnan(prob_vals)) & (~np.isnan(vod_vals))
        if mask.sum() < 1000:
            log(f"  Skipping {fname}: too few valid pixels ({int(mask.sum())})")
            continue

        p = prob_vals[mask]
        v = vod_vals[mask]

        # Bin using numpy (avoids pushing millions of rows through pandas)
        # digitize returns bin index in 1..len(bins)-1
        bin_idx = np.digitize(p, bins, right=False) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(bin_centers))
        bin_idx = bin_idx[valid]
        v = v[valid]

        # Aggregate stats per bin
        stats_rows = []
        for bi, bc in enumerate(bin_centers):
            vv = v[bin_idx == bi]
            if vv.size == 0:
                stats_rows.append((bc, np.nan, np.nan, 0))
            else:
                stats_rows.append((bc, float(vv.mean()), float(vv.std(ddof=1)) if vv.size > 1 else 0.0, int(vv.size)))

        summary = pd.DataFrame(stats_rows, columns=['psat_bin_mid', 'vod_cv_mean', 'vod_cv_std', 'count'])
        summary['cluster'] = cid
        summary['interaction'] = f"{f1}_{f2}"
        results.append(summary)

        # Plot per interaction
        plt.figure(figsize=(8, 5))
        plt.errorbar(summary['psat_bin_mid'], summary['vod_cv_mean'], yerr=summary['vod_cv_std'], fmt='o-', capsize=4)
        plt.xlabel("Probability of Adaptation Saturation (Model Inferred)")
        plt.ylabel("VOD Interannual CV")
        plt.title(f"VOD Validation: Cluster {cid} ({f1} x {f2})")
        plt.grid(True, alpha=0.3)

        # Secondary axis: pixel count per bin
        ax2 = plt.twinx()
        ax2.bar(summary['psat_bin_mid'], summary['count'], width=(bins[1]-bins[0]) * 0.8, alpha=0.15, color='gray')
        ax2.set_ylabel('Pixel Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"vod_val_c{cid}_{f1}_{f2}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        log(f"  Plot saved to {plot_path}")

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(os.path.join(OUTPUT_DIR, "vod_validation_summary.csv"), index=False)
        log("Validation complete.")
    else:
        log("No validation results generated.")

if __name__ == "__main__":
    main()
