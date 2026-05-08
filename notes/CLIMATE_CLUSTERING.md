# Climate Clustering Methodology (k=6)

## Objective
Generate spatially and temporally consistent climate clusters ($k=6$) across all historical and future scenarios to serve as conditioning bins for land use feature estimation.

## Technical Approach
The clustering follows a "Fit-on-Present, Project-to-Scenario" workflow to ensure that the cluster definitions remain stable while their spatial extents shift with climate change.

1.  **Feature Selection:**
    - **Climate:** 40 annual climate variables (Precipitation, Temperature, Aridity indices, etc.) from `data/climate_features.zarr`.
    - **Terrain:** Elevation data from `data/terrain_0p05.nc`, resampled to the 0.05° global grid.
2.  **Preprocessing:**
    - **Standardization:** Features were standardized using `StandardScaler` fitted on the 1991–2020 (Present Day) baseline.
    - **Dimensionality Reduction:** PCA was applied to reduce the 41-feature space to 5 principal components, capturing the majority of global variance.
3.  **Clustering Method:**
    - **Training:** A K-Means model ($k=6$) was fitted on a representative subsample of 20,000 pixels from the baseline period.
    - **Proxy Assignment:** To ensure computational efficiency and avoid thread-lock issues on the high-core system, a Random Forest Classifier was trained as a proxy to replicate the K-Means assignments.
4.  **Projection:**
    - The trained pipeline (Scaler -> PCA -> RF Proxy) was applied to all 12 target scenarios.

## Data & Scripts
- **Script:** `scripts/data-preparation/compute_climate_clusters.py`
- **Reference Logic:** Adapted from `scripts/ref/precompute_clusters.py`.
- **Input Data:**
    - `data/climate_features.zarr` (Groups: `1901_1930` through `2071_2099/ssp460`)
    - `data/terrain_0p05.nc` (Variable: `elev`)

## Outputs
- **Zarr Store:** `data/climate_clusters.zarr`
- **Structure:** Hierarchical groups matching the climate input structure (e.g., `1991_2020`, `2041_2070/ssp370`).
- **Variable:** `cluster` (int8), where values range from 0–5, and -1 represents invalid/ocean pixels.

## Summary of Results
Clusters correctly capture major global biomes (e.g., Arid, Tropical, Boreal) and exhibit realistic spatial shifts under SSP scenarios as temperature and precipitation patterns evolve.
