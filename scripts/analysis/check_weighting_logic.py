import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr

def check_weighting_math():
    # Load a sample file
    sample_path = '/home/vajira/SRchange/output_SR/scenario_predictions/CLIM_1901_1930_LU_1901_1930/ensemble_1/fold_1/sr_k6.nc'
    ds = xr.open_dataset(sample_path)
    da = ds.sr
    
    # 1. Simple weighting (from your notebook logic: da * cos(lat))
    weights = np.cos(np.deg2rad(da.lat))
    simple_weighted_da = da * weights
    simple_mean = float(simple_weighted_da.mean())
    
    # 2. Robust weighting (Xarray weighted mean)
    w_obj = da.weighted(weights)
    robust_mean = float(w_obj.mean())
    
    # 3. Ratio comparison
    sum_weights = float(weights.sum())
    n_lon = len(da.lon)
    # The simple mean is effectively sum(da * w) / (n_lat * n_lon)
    # The robust mean is sum(da * w * 1) / (sum(w) * n_lon)
    
    print(f"File: {sample_path}")
    print(f"Simple Mean (notebook style): {simple_mean:.4f}")
    print(f"Robust Mean (xarray style):   {robust_mean:.4f}")
    print(f"Ratio (Robust/Simple):        {robust_mean/simple_mean:.4f}")

if __name__ == "__main__":
    check_weighting_math()
