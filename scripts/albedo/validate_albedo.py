import os
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score, mean_squared_error

# Paths
DIR_ROOT = "/home/vajira/SRchange"
OUTPUT_ALBEDO = os.path.join(DIR_ROOT, 'output_albedo')
TARGET_PRESENT = os.path.join(DIR_ROOT, 'data/albedo_mean_10th_2011_2020.nc')
TARGET_HIST = os.path.join(DIR_ROOT, 'data/albedo_mean_10th_1981_1990.nc')

PRED_PRESENT_OOS = os.path.join(OUTPUT_ALBEDO, 'albedo_present_oos_mosaic.nc')
PRED_HIST = os.path.join(OUTPUT_ALBEDO, 'albedo_historical_ensemble_mean.nc')

def calculate_metrics(target_path, pred_path, label):
    print(f"\n--- Validation: {label} ---")
    
    ds_tar = xr.open_dataset(target_path)
    ds_pre = xr.open_dataset(pred_path)
    
    # Identify variables
    v_tar = 'albedo_10th'
    v_pre = 'albedo_10th'
    
    tar = ds_tar[v_tar].values.flatten()
    pre = ds_pre[v_pre].values.flatten()
    
    mask = ~np.isnan(tar) & ~np.isnan(pre)
    tar = tar[mask]
    pre = pre[mask]
    
    if len(tar) == 0:
        print("Error: No common valid pixels found.")
        return
        
    r2 = r2_score(tar, pre)
    rmse = np.sqrt(mean_squared_error(tar, pre))
    bias = np.mean(pre - tar)
    
    print(f"Pixels: {len(tar)}")
    print(f"R2:     {r2:.4f}")
    print(f"RMSE:   {rmse:.4f}")
    print(f"Bias:   {bias:.4f}")

    # Difference Map
    diff = ds_pre[v_pre] - ds_tar[v_tar]
    diff.to_netcdf(os.path.join(OUTPUT_ALBEDO, f"albedo_diff_{label.lower().replace(' ', '_')}.nc"))
    print(f"Difference map saved to output_albedo/albedo_diff_{label.lower().replace(' ', '_')}.nc")

if __name__ == "__main__":
    if os.path.exists(PRED_PRESENT_OOS):
        calculate_metrics(TARGET_PRESENT, PRED_PRESENT_OOS, "Present OOS")
    else:
        print("Present OOS prediction not found yet.")
        
    if os.path.exists(PRED_HIST):
        calculate_metrics(TARGET_HIST, PRED_HIST, "Historical Validation")
    else:
        print("Historical prediction not found yet.")
