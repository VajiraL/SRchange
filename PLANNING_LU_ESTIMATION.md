# Planning: Land Use Feature Estimation

## Goal
Estimate four continuous land use features (`gHM`, `irr_frac`, `rain_frac`, `forest_age`) for all climate and land use scenarios (Historical, SSP1-4, Pre-Industrial, Intact) at 0.05° resolution.

## Strategy: Rank-Based Statistical Transfer
- **Model:** Beta-Point-Mass mixture model to handle skewed fractional data and zero inflation.
- **Conditioning Bins:**
    - **Standard (1901–2100):** LU Class $\times$ Climate Cluster ($k=6$) $\times$ Population Level (10 bins).
    - **Intact/Pre-Industrial:** LU Class $\times$ Climate Cluster ($k=6$) $\times$ Anthrome Class.
- **Spatial Structure:** Preserve local patterns via "intrinsic rank" sampling (Pixel rank within its bin in present-day is transferred to scenario-specific bins).
- **Soft Constraints:** Allow a margin of error (e.g., 5-10%) when validating if `irr_frac + rain_frac` matches Cropland extent, avoiding hard artifacts.

## Phase 1: Dynamic Climate Clustering (Current Focus)
- **Objective:** Create a unified `climate_clusters.zarr` containing cluster IDs ($k=6$) for all target scenarios.
- **Method:**
    1. Standardize features (Annual Precip, Temperature, Aridity, etc. + Elevation).
    2. Dimensionality reduction via PCA.
    3. Global K-Means clustering (using present-day as the training baseline).
    4. Predict cluster IDs for all scenario-specific climate states.
- **Scenarios (12 total):**
    - **Historical:** 1901-1930, 1931-1960, 1961-1990, 1991-2020.
    - **Future:** (2041-2070, 2071-2099) $\times$ (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP4-6.0).

## Phase 2: Feature Learning & Estimation
1. **Learn:** Fit Beta distributions to present-day observational data within (LU, Cluster, Pop) bins.
2. **Transfer:** Use scenario-specific LU, Population, and Cluster maps to lookup distributions and sample using the rank-order method.
3. **Refined Forest Age:** Tracing forest history back to 1900, adding a pre-observation offset sampled from forest-specific bins.

## Hardware Optimization
- Use `dask` and `multiprocessing` to leverage 364 cores.
- Keep high-resolution rank maps and distribution tables in memory (2.3 TB available).
