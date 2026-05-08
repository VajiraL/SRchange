# Supplementary Note: Regional Model Specialization and Sensitivity

## Overview
This section quantifies the regional specialization of the $S_R$ models. By applying every cluster-specific model to every grid cell globally, we assess the "hydrological distance" between climate clusters and the necessity of the climate-gating approach.

## 1. Regional Specialization Matrix
The table below shows the Mean Absolute Error (MAE in mm) of each regional model when evaluated against ground truth across all clusters.

| Test Cluster \ Trained Model | M0 (Arid) | M1 (Trop) | M2 (Temp) | M3 (Bor) | M4 (Medit) | M5 (Subtrop) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0 (Arid)** | **8.5** | 116.0 | 33.8 | 671.6 | 38.5 | 30.8 |
| **C1 (Tropical)** | 309.5 | **77.9** | 199.6 | 774.6 | 470.6 | 501.1 |
| **C2 (Temperate)** | 189.7 | 157.8 | **32.4** | 2567.3 | 121.8 | 157.1 |
| **C3 (Boreal)** | 143.7 | 258.6 | 165.6 | **65.2** | 328.0 | 312.2 |
| **C4 (Mediterranean)** | 91.8 | 169.4 | 80.4 | 1970.8 | **33.8** | 75.6 |
| **C5 (Subtropical)** | 83.1 | 96.9 | 76.3 | 574.3 | 92.4 | **37.0** |

### Key Findings:
- **Diagonal Dominance:** In all 6 regions, the model trained on the local climate (the diagonal) outperformed all foreign models. This confirms that regional $S_R$ sensitivities are distinct and that a "Global-Only" model inevitably averages out critical regional signals.
- **Boreal Divergence:** Model 3 (Boreal) exhibits the highest degree of specialization. While it is highly accurate in high latitudes (65.2 mm), its errors exceed 2000 mm in temperate zones, indicating that the logic governing storage in permafrost or snow-dominated regions is fundamentally incompatible with lower-latitude hydrology.
- **Arid-Subtropical Continuity:** Models 0 and 5 show the highest cross-compatibility, reflecting a shared reliance on similar precipitation-PET thresholds.

## 2. Gating Efficiency
We define "Gating Efficiency" as the percentage of pixels where the climate-assigned model is the mathematically optimal choice among all available models.

| Cluster | Gating Efficiency (%) |
| :--- | :---: |
| Arid | 55.0% |
| Tropical | 52.0% |
| Temperate | 57.7% |
| Boreal | 22.5% |
| Mediterranean | 33.9% |
| Subtropical | 38.8% |

*Interpretation: In the Arid, Tropical, and Temperate zones, the climate-clustering successfully identifies the optimal model for the majority of the landscape. In the Boreal and Mediterranean zones, the lower efficiency suggests significant sub-regional heterogeneity, where transition-zone cells might be better served by a neighboring cluster's logic.*

## 3. Conclusion
The sensitivity analysis demonstrates that $S_R$ behavior is not globally uniform. The climate-gating approach ($k=6$) effectively captures these regional nuances, particularly for extreme climates (Tropical and Boreal) where "off-cluster" predictions lead to catastrophic error inflation.
