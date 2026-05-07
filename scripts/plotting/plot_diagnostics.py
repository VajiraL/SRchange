import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
import os

# Config
ROOT_DIR = "/home/vajira/SRchange"
DATA_MODELS = f"{ROOT_DIR}/data/statistical_models"
DATA_PRESENT = f"{ROOT_DIR}/data/presentLU"
DATA_LU = f"{ROOT_DIR}/data/harmonized_LU_classes.zarr"
DATA_CLUSTER = f"{ROOT_DIR}/data/climate_clusters.zarr"
DATA_POP = f"{ROOT_DIR}/data/population"
OUT_DIR = f"{ROOT_DIR}/notes/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_distribution_diagnostic(feat='forest_age', bin_key='44_1_1'):
    print(f"Plotting diagnostic for {feat} in bin {bin_key}...")
    
    # 1. Load Data
    with open(f"{DATA_MODELS}/beta_mixture_models.json", 'r') as f:
        models = json.load(f)
    
    model = models[feat][bin_key]
    weight_one = model.get('weight_one', 0.0)
    
    # Load actual data for this bin to compare
    ds_lu = xr.open_zarr(DATA_LU, group='1991_2020')
    ds_cluster = xr.open_zarr(DATA_CLUSTER, group='1991_2020')
    ds_pop = xr.open_zarr(f"{DATA_POP}/population_histsoc_30arcmin_annual_1901_2014_binned.zarr")
    pop_2015 = ds_pop['pop_bin'].sel(time='2014-01-01', method='nearest').values
    
    lu_classes = ds_lu['LU_class'].values
    clusters = ds_cluster['cluster'].values
    
    lu, cluster, pop = map(int, bin_key.split('_'))
    bin_mask = (lu_classes == lu) & (clusters == cluster) & (pop_2015 == pop)
    
    if feat == 'forest_age':
        ds_obs = xr.open_dataset(f"{DATA_PRESENT}/forest_age.nc")
        actual_data = ds_obs['forest_age'].values[bin_mask]
        actual_data = actual_data[~np.isnan(actual_data)]
        actual_norm = np.clip(actual_data / 300.0, 0, 1)
    elif feat == 'gHM':
        ds_obs = xr.open_dataset(f"{DATA_PRESENT}/gHM.nc")
        actual_data = ds_obs['ghm'].values[bin_mask]
        actual_norm = actual_data[~np.isnan(actual_data)]
    else:
        ds_obs = xr.open_dataset(f"{DATA_PRESENT}/{feat}.nc")
        actual_data = ds_obs[feat].values[bin_mask]
        actual_norm = actual_data[~np.isnan(actual_data)]

    # 2. Sample from model
    x = np.linspace(0, 1, 1000)
    wz = model['weight_zero']
    a, b = model['beta_params']
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Histogram of actual
    plt.hist(actual_norm, bins=50, density=True, alpha=0.3, color='gray', label='Actual Observations')
    
    # fitted Beta PDF
    y = stats.beta.pdf(x, a, b)
    plt.plot(x, y * (1-wz-weight_one), 'r-', lw=2, label=f'Fitted Beta Mixture (wz={wz:.2f}, wo={weight_one:.2f})')
    
    # Add point masses as vertical bars
    if wz > 0:
        plt.vlines(0, 0, plt.gca().get_ylim()[1], colors='blue', linestyles='dashed', label='Zero Point Mass')
    if weight_one > 0:
        plt.vlines(1, 0, plt.gca().get_ylim()[1], colors='green', linestyles='dashed', label='One Point Mass')
    
    plt.title(f"Distribution Diagnostic: {feat} (Bin {bin_key})")
    plt.xlabel("Normalized Value [0-1]")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(f"{OUT_DIR}/diagnostic_{feat}_{bin_key}.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    # Diagnostic for Forest Age in a specific bin (Forest, Cluster 1, Pop 1)
    plot_distribution_diagnostic('forest_age', '44_1_1')
    # Diagnostic for gHM in an Urban bin
    plot_distribution_diagnostic('gHM', '11_0_8')
