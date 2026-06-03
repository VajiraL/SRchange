# Phase 7 Scientific Note: Ecological Validation via SIF and VOD

## Objective
Verify if model-inferred "Saturation Probabilities" ($P_{sat}$) correspond to observable ecological state changes.

## Methodology
- **Proxy:** SIF Coefficient of Variation (CV) from LHSIF (1995–2023). 
  - *Rationale:* Systems near their adaptation limit ($P_{sat} \to 0$) are expected to show higher interannual variability (higher CV) due to reduced water-storage buffering. Saturated systems ($P_{sat} \to 1$) should show more stable function.
- **Analysis:** Binned SIF CV against $P_{sat}$ for key interaction surfaces.

## Key Results
- **Boreal (C0):** Strong negative relationship between $P_{sat}$ and SIF CV. As the probability of being "Saturated" increases, the interannual variability of SIF drops significantly. This confirms that the model's "Saturated" regime corresponds to hydrologically buffered ecosystem function.
- **Arid/Semi-Arid (C3):** Similar trend, though noise increases at low $P_{sat}$. 
- **Temperate/Tropical (C4/C5):** The relationship is robust across all clusters analyzed. 

## VODCA Validation (Hydraulic Resilience Proxy)
- **Proxy:** VOD interannual coefficient of variation (CV) computed from CXKu monthly VODCA, used as an independent vegetation resilience/stress proxy.
- **Overall pattern:** Across most tested interaction surfaces (1991–2020), higher $P_{sat}$ bins often correspond to lower VOD CV (i.e., buffered behavior), consistent with the “Saturated = safe operating space” interpretation.
- **Notable nuance:** The gHM-based interactions show stronger non-monotonicity/noisiness (e.g., clusters C3 and C5), indicating that VOD-based hydraulic/vegetation coupling is more heterogeneous than the SIF CV proxy.
- **Artifacts generated:** binned validation plots and `vod_validation_summary.csv` are written to `output_SR/thresholds_analysis/validation_vod/`.

## Conclusion
The SHAP-derived thresholds are not merely model artifacts. The transition from $P_{sat} \approx 0$ to $P_{sat} \approx 1$ aligns with a physical stabilization of vegetation function, validating our "Safe Operating Space" framing for Paper 1.
