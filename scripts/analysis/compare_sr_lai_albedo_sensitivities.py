import xarray as xr
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr

# Paths
DIR_ROOT = "/home/vajira/SRchange"
OUTPUT_SR = os.path.join(DIR_ROOT, 'output_SR')
OUTPUT_LAI = os.path.join(DIR_ROOT, 'output_lai')
OUTPUT_ALB = os.path.join(DIR_ROOT, 'output_albedo')

def load_and_flatten(path, var_name):
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path)
    # The var name in the NC might vary. Usually it's the same as the file suffix or explicitly defined.
    # In generate_sensitivity.py it's 'lai_95th' or 'albedo_10th'
    # In calculate_sr_sensitivity_maps.py it's 'sr_k6_mean'
    if 'sr_k6_mean' in ds:
        v = 'sr_k6_mean'
    elif 'lai_95th' in ds:
        v = 'lai_95th'
    elif 'albedo_10th' in ds:
        v = 'albedo_10th'
    else:
        # Fallback to the first data variable
        v = list(ds.data_vars)[0]
    
    return ds[v].values.flatten()

def compare_sensitivities():
    types = ['climate', 'lu', 'total_change']
    results = []
    
    for t in types:
        print(f"\nComparing {t} effects...")
        
        f_sr = os.path.join(OUTPUT_SR, f"sr_sensitivity_{t}.nc" if t != 'total_change' else "sr_total_change.nc")
        f_lai = os.path.join(OUTPUT_LAI, f"lai_sensitivity_{t}.nc" if t != 'total_change' else "lai_total_change.nc")
        f_alb = os.path.join(OUTPUT_ALB, f"albedo_sensitivity_{t}.nc" if t != 'total_change' else "albedo_total_change.nc")
        
        sr = load_and_flatten(f_sr, 'sr')
        lai = load_and_flatten(f_lai, 'lai')
        alb = load_and_flatten(f_alb, 'albedo')
        
        if sr is None or lai is None or alb is None:
            print(f"Skipping {t} due to missing files.")
            continue
            
        # Mask NaNs
        mask = ~np.isnan(sr) & ~np.isnan(lai) & ~np.isnan(alb)
        sr_m = sr[mask]
        lai_m = lai[mask]
        alb_m = alb[mask]
        
        if len(sr_m) == 0:
            print(f"No common valid pixels for {t}.")
            continue
            
        c_sr_lai, _ = pearsonr(sr_m, lai_m)
        c_sr_alb, _ = pearsonr(sr_m, alb_m)
        c_lai_alb, _ = pearsonr(lai_m, alb_m)
        
        print(f"Pearson Corr (SR vs LAI):    {c_sr_lai:.4f}")
        print(f"Pearson Corr (SR vs Albedo): {c_sr_alb:.4f}")
        print(f"Pearson Corr (LAI vs Albedo): {c_lai_alb:.4f}")
        
        results.append({
            'type': t,
            'corr_sr_lai': c_sr_lai,
            'corr_sr_alb': c_sr_alb,
            'corr_lai_alb': c_lai_alb,
            'n_pixels': len(sr_m)
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DIR_ROOT, 'output_SR/sensitivity_cross_correlation.csv'), index=False)
    print(f"\nResults saved to {os.path.join(DIR_ROOT, 'output_SR/sensitivity_cross_correlation.csv')}")

if __name__ == "__main__":
    compare_sensitivities()
