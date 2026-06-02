import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
DATA_PATH = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/extracted_shap_interactions.parquet')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/thresholds')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Piecewise Linear Model
# ---------------------------
def piecewise_linear(x, x0, y0, k1, k2):
    return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))

def fit_threshold(x, y):
    # Initial guesses
    x_min, x_max = np.min(x), np.max(x)
    x0_guess = (x_min + x_max) / 2
    y0_guess = np.mean(y)
    k1_guess = (y[len(y)//2] - y[0]) / (x[len(x)//2] - x[0]) if len(x) > 1 else 0
    k2_guess = (y[-1] - y[len(y)//2]) / (x[-1] - x[len(x)//2]) if len(x) > 1 else 0
    
    try:
        popt, pcov = curve_fit(piecewise_linear, x, y, 
                               p0=[x0_guess, y0_guess, k1_guess, k2_guess],
                               bounds=([x_min, -np.inf, -np.inf, -np.inf], 
                                       [x_max, np.inf, np.inf, np.inf]))
        
        # Calculate R-squared
        residuals = y - piecewise_linear(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return popt, r_squared
    except Exception as e:
        # log(f"Fit failed: {e}")
        return None, 0

# ---------------------------
# 3. Processing
# ---------------------------
def main():
    log(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    features = ['aridity_index', 'gHM', 'pet_annual']
    results = []
    
    cluster_ids = sorted(df['cluster'].unique())
    
    for cid in cluster_ids:
        log(f"Processing Cluster {cid}...")
        df_c = df[df['cluster'] == cid]
        
        for feat in features:
            shap_feat = f'shap_{feat}'
            if shap_feat not in df_c.columns: continue
            
            # Prepare data: binning to handle large N and noise
            # We want to fit the trend, not the individual points
            data_sub = df_c[[feat, shap_feat]].dropna()
            if len(data_sub) < 100: continue
            
            # Binning
            data_sub['bin'] = pd.qcut(data_sub[feat], q=100, duplicates='drop')
            binned = data_sub.groupby('bin', observed=True).agg({feat: 'mean', shap_feat: 'mean'}).reset_index()
            
            x = binned[feat].values
            y = binned[shap_feat].values
            
            popt, r2 = fit_threshold(x, y)
            
            if popt is not None:
                x0, y0, k1, k2 = popt
                results.append({
                    'cluster': cid,
                    'feature': feat,
                    'threshold': x0,
                    'y_at_threshold': y0,
                    'slope_before': k1,
                    'slope_after': k2,
                    'slope_change': k2 - k1,
                    'r_squared': r2
                })
                
                # Plotting
                plt.figure(figsize=(10, 6))
                plt.scatter(x, y, label='Binned Data', color='gray', alpha=0.5)
                x_fine = np.linspace(np.min(x), np.max(x), 500)
                plt.plot(x_fine, piecewise_linear(x_fine, *popt), 'r-', label=f'Piecewise Fit (R²={r2:.3f})')
                plt.axvline(x0, color='blue', linestyle='--', label=f'Threshold: {x0:.4f}')
                plt.xlabel(feat)
                plt.ylabel(f"SHAP value for {feat}")
                plt.title(f"Cluster {cid}: Threshold Detection for {feat}")
                plt.legend()
                
                save_path = os.path.join(PLOT_DIR, f"threshold_cluster_{cid}_{feat}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                # log(f"  Detected threshold for {feat}: {x0:.4f} (R²={r2:.3f})")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_BASE, "detected_thresholds.csv"), index=False)
    log(f"Saved {len(results_df)} detected thresholds to CSV.")

if __name__ == "__main__":
    main()
