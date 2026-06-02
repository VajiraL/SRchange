# Phase 6: Multivariate Threshold Interaction Surfaces

## Overview
Ecosystem thresholds are rarely univariate. We used tensor-product Generalized Additive Models (GAMs) to model the interaction between key climatic and land-use drivers. This allowed us to extract **Threshold Contours**—showing how the critical limit for one driver (e.g., Aridity) shifts under different levels of another constraint (e.g., Forest Age or Human Modification).

## Key Interactions and Findings

### 1. Aridity Index vs. Human Modification (gHM)
- **Clusters 0, 3, 5:** In all three clusters, high human modification (gHM) significantly **lowers the aridity threshold**. 
- Ecosystems in more modified landscapes reach their $S_R$ adaptation limit much earlier (at lower aridity levels) than pristine ecosystems. 
- This suggests a synergistic effect where human-driven degradation reduces the hydroclimatic "buffer capacity" of the land surface.

### 2. Aridity Index vs. Forest Age
- **Clusters 0, 2, 4:** There is a clear interaction where **older forests possess higher aridity thresholds**.
- As forest stands mature (particularly beyond the 40-70 year "Hard" thresholds identified in Phase 5), they appear capable of expanding their root zone storage capacity to withstand greater climatic stress.
- This highlights the role of vegetation maturity in enhancing ecosystem resilience.

## Statistical Method
- **Tensor-Product GAMs:** Modeled $SHAP(S_R)_{feat1} = f(feat1, feat2)$.
- **Uncertainty:** Stability was verified via spatial block bootstrap in Phase 5b, showing that these surfaces are robust features of the learned model response.
- **Extraction:** Threshold contours were defined where the marginal sensitivity ($\partial SHAP / \partial f_1$) dropped below 10% of its maximum value.

## Significance
The discovery of these "interaction surfaces" provides a more nuanced view of vulnerability. For example, a future increase in aridity might be "safe" for an old-growth forest but "critical" for a young or human-impacted stand in the same climatic zone.

## Next Steps
- **Phase 7: Ecological Validation:** Test these surfaces against independent VOD and GPP sensitivity data.
- **Phase 8: Dynamic Scenario Mapping:** Use these 2D surfaces to map future vulnerability, allowing both climate and land-use to move independently.
