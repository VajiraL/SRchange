# Phase 5: Advanced Threshold Detection & Taxonomy

## Overview
In this phase, we upgraded our threshold detection methodology by integrating Generalized Additive Models (GAMs) and a multi-criteria taxonomy. We expanded the analysis beyond Aridity Index and gHM to include `forest_age`, `pet_annual`, `annual_precip`, `monthly_precip_std`, and `mean_annual_tmp`.

## Methodology
- **Statistical Framework:** Used `pygam` to fit smooth response curves to binned SHAP values.
- **Taxonomy Classification:**
    - **Hard:** High-confidence ($R^2 > 0.85$) piecewise linear breakpoints.
    - **Soft:** Smooth saturation detected via first-derivative decay in GAMs.
    - **Collapse:** Sign reversal in the first derivative (marginal effect changes from positive to negative).
    - **None:** No significant non-linearity detected.
- **Implementation:** Leveraged the 364-core environment to process all features across all clusters in parallel.

## Key Findings

### 1. New Threshold Variables
- **Forest Age:** Detected "Hard" thresholds in Clusters 0, 1, 2, 4, and 5. For example, in Cluster 0 (Boreal), $S_R$ adaptation potential shifts abruptly at **~71 years**, suggesting a transition in rooting strategy as forests mature.
- **Annual Precipitation:** Strong "Hard" thresholds detected in Cluster 5 at **~578 mm/yr** and Cluster 4 at **~715 mm/yr**.
- **Temperature (mean_annual_tmp):** Detected a "Collapse" threshold at **~20.6°C** in Cluster 3 (Semi-arid), where higher temperatures begin to negatively impact $S_R$ adaptation.

### 2. Diverse Response Types
- **Aridity Index:** Mostly "Hard" transitions, confirming our initial findings.
- **PET Annual:** Frequently shows "Collapse" behavior, indicating that beyond certain evaporative demand levels, the ecosystem's ability to maintain a large $S_R$ buffer is compromised.
- **Human Modification (gHM):** Transitioned from "Hard" to "Collapse" in some clusters (e.g., Cluster 2), suggesting that high-intensity modification not only saturates but actively suppresses $S_R$ adaptation.

## Significance
Moving beyond a binary "Adaptive vs. Saturated" view allows us to pinpoint where ecosystems are not just reaching a limit, but are potentially undergoing a functional collapse in their water storage strategies.

## Next Steps
- **Spatial Block Bootstrap (Phase 5b):** Quantify the probability distributions for these newly identified thresholds.
- **Multivariate Surface Selection:** Select the most robust "Hard" and "Collapse" variables for interaction modeling (e.g., Aridity vs. Forest Age).
