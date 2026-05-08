# Estimation of Scenario Land Use (LU) Variables

This note describes the methodology used to generate continuous high-resolution (0.05°) predictor variables for scenarios where only categorical land use (LU) classes are available.

## 1. Concept: Statistical Transfer of Local Characteristics
The core challenge is estimating continuous variables (`gHM`, `irr_frac`, `rain_frac`, `forest_age`) for past or future scenarios. We use a **Beta-Point-Mass Mixture Model** to learn the statistical distribution of these variables conditioned on local environments, and then apply a **Rank-Based Transfer** (Copula-like approach) to preserve spatial details.

### Conditioning Variables
Distributions are conditioned on:
1.  **LU Class:** Harmonized into 7 classes (Water, Urban, Rainfed, Rangeland, Forest, Pasture, Natural).
2.  **Climate Cluster:** Global pixels grouped into $k=6$ clusters based on PCA of temperature and precipitation.
3.  **Human Influence:**
    *   **Standard Scenarios:** Binned Population Density (10 levels).
    *   **Special Scenarios (Pre-Industrial/Intact):** Anthrome classes (1700 or 2015).

## 2. Methodology & Logic

### Step A: Learning (Baseline 1991–2020)
For each unique combination of (LU $\times$ Cluster $\times$ Population/Anthrome), we fit:
-   A **Zero-Point-Mass**: Captures the probability that the variable is exactly zero.
-   A **Beta Distribution**: Captures the shape and spread of non-zero values (scaled to [0, 1]).

### Step B: Rank Mapping
We generate "Rank Maps" from present-day observations. For every pixel, we determine its percentile rank *within its own conditioning bin*. These ranks represent local spatial anomalies (e.g., "This pixel is the 90th percentile of forest age for a Tropical Forest in a low-pop area").

### Step C: Synthesis (Scenario Mapping)
For a target scenario (e.g., SSP3 2085):
1.  Determine the target conditioning bin for each pixel using scenario LU, scenario Climate Clusters, and scenario Population.
2.  Look up the learned Beta parameters for that bin.
3.  Map the present-day **Rank** back into the new distribution using the Inverse CDF (PPF).
4.  Apply physical constraints (e.g., `irr_frac + rain_frac` limited to 110% of cropland area).

## 3. Scripts and Workflow

1.  `scripts/data-preparation/compute_climate_clusters.py`: Generates the $k=6$ spatial clusters for all time periods.
2.  `scripts/data-preparation/train_lu_distributions.py`: Performs the Beta-mixture fitting and generates the `_rank_map.nc` files.
3.  `scripts/data-preparation/estimate_scenario_features.py`: The main engine that synthesizes the 14 scenarios using multiprocessing.
4.  `scripts/data-preparation/validate_synthetic_features.py`: Compares 1991–2020 synthetic output against original observations (achieving $r > 0.83$ for all variables).

## 4. Input Data
-   **LU Classes:** `data/harmonized_LU_classes.zarr`
-   **Climate Clusters:** `data/climate_clusters.zarr`
-   **Population:** `data/population/*.nc`
-   **Observations (Present):** `data/presentLU/*.nc` (gHM, Forest Age, etc.)

## 5. Outputs
Final synthesized predictors are stored in `data/scenario_features/`:
-   `standard_{period}_{ssp}.nc` (Historical and SSPs)
-   `pre-industrial_1901_1930_none.nc`
-   `intact_1991_2020_none.nc`

Each file contains the 4 continuous variables required for the final Root Zone Storage Capacity ($S_R$) modeling.
