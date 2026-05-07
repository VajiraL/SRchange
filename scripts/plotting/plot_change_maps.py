import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os

ROOT_DIR = "/home/vajira/SRchange"
DATA_DIR = f"{ROOT_DIR}/data/scenario_features"
OUT_DIR = f"{ROOT_DIR}/notes/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_change_map(feat='forest_age', vmin=0, vmax=300, cmap='viridis'):
    print(f"Plotting change map for {feat}...")
    
    # 1. Load Scenarios
    ds_hist = xr.open_dataset(f"{DATA_DIR}/standard_1991_2020_none.nc")
    ds_future = xr.open_dataset(f"{DATA_DIR}/standard_2071_2099_ssp370.nc")
    ds_intact = xr.open_dataset(f"{DATA_DIR}/intact_1991_2020_none.nc")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), subplot_kw={'projection': ccrs.PlateCarree()})
    
    scenarios = [
        (ds_hist[feat], "Historical (1991-2020)"),
        (ds_future[feat], "Future SSP3-7.0 (2085)"),
        (ds_intact[feat], "Intact Scenario")
    ]
    
    for i, (da, title) in enumerate(scenarios):
        ax = axes[i]
        im = da.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap, 
                     add_colorbar=False)
        ax.coastlines()
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=feat, orientation='vertical', shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/change_map_{feat}.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    plot_change_map('forest_age', 0, 300, 'viridis')
    plot_change_map('gHM', 0, 1, 'magma')
