#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting script for Cluster Sensitivity Analysis.
Generates:
1. Heatmap of the Cluster-Model Performance Matrix.
2. Global map of the 'Winning Model' ID.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy import crs as ccrs

# Config
OUTDIR = "/home/vajira/SRchange/output_SR/cluster_sensitivity_analysis"
MATRIX_FILE = os.path.join(OUTDIR, "mae_sensitivity_matrix.csv")
MAP_FILE = os.path.join(OUTDIR, "sensitivity_analysis_maps.nc")

def plot_sensitivity_heatmap():
    df = pd.read_csv(MATRIX_FILE, index_index=0)
    plt.figure(figsize=(10, 8))
    # Log scale for better visualization of the huge Boreal errors
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'MAE (mm)'})
    plt.title("Cluster-Model Sensitivity Matrix (MAE)")
    plt.xlabel("Model ID (Trained on Cluster X)")
    plt.ylabel("Test Cluster ID (Evaluated on Cluster Y)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "sensitivity_heatmap.png"), dpi=300)
    plt.close()

def plot_winning_model_map():
    ds = xr.open_dataset(MAP_FILE)
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Custom discrete cmap for 6 clusters
    cmap = plt.get_cmap('tab10', 6)
    p = ds.best_model_id.plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap=cmap, add_colorbar=False, vmin=-0.5, vmax=5.5
    )
    
    cb = plt.colorbar(p, orientation='horizontal', pad=0.05, shrink=0.6)
    cb.set_label('Best Performing Model ID')
    ax.coastlines(resolution='110m', linewidth=0.5)
    plt.title("Spatial Distribution of the Mathematically Optimal Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "winning_model_map.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_sensitivity_heatmap()
    plot_winning_model_map()
