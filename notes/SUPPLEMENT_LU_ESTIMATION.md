# Supplementary Method: Estimation of Continuous LU-Related Predictors

## 1. Objective and Rational
To model root zone storage capacity ($S_R$) across diverse historical and future scenarios, high-resolution (0.05°) continuous predictors are required. While categorical land use (LU) classes are available from the LUH2/LUH2-GCB datasets, associated continuous variables—specifically the Global Human Modification index (`gHM`), irrigation/rainfed fractions (`irr_frac`, `rain_frac`), and forest stand age (`forest_age`)—are often missing or spatially inconsistent across time periods.

We employ a statistical distribution transfer method to synthesize these variables for 14 scenarios, ensuring that local spatial structure and realistic pixel-level variability are preserved.

## 2. Statistical Framework

### 2.1 Beta-Mixture with Dual Point Masses
We characterize each variable $x \in [0, 1]$ using a mixture of point masses at the boundaries and a Beta distribution for the interior:

$$ f(x) = \pi_0 \delta(0) + \pi_1 \delta(1) + (1 - \pi_0 - \pi_1) \text{Beta}(x; \alpha, \beta) $$

where:
- $\pi_0$ is the probability of the variable being zero (e.g., non-irrigated or non-forested).
- $\pi_1$ is the probability of the variable being at its maximum value (specifically relevant for `forest_age`, where 15-20% of pixels are saturated at the 300-year cap).
- $\alpha, \beta$ are the shape parameters for the continuous interior.

### 2.2 Conditioning
Distributions are conditioned on local environmental and socio-economic drivers to capture global patterns of variability:
- **Bioclimatic Constraints:** $k=6$ global clusters derived from PCA and K-means clustering of temperature and precipitation.
- **Land Management:** Categorical LU classes (Urban, Cropland, Rangeland, Forest, Pasture, Natural).
- **Human Pressure:** Binned population density (Standard scenarios) or Anthrome classes (Pre-Industrial/Intact scenarios).

### 2.3 Rank-Based Transfer (Copula Approach)
...
In scenario synthesis, we apply this rank to the target distribution's Inverse Cumulative Distribution Function (ICDF or PPF):

$$ \hat{x}_{scenario} = 
\begin{cases} 
0 & \text{if } R \le \pi_0 \\
1 & \text{if } R > (1 - \pi_1) \\
F_{Beta}^{-1}\left(\frac{R - \pi_0}{1 - \pi_0 - \pi_1}; \alpha, \beta\right) & \text{otherwise}
\end{cases}
$$


## 3. Implementation and Results

### 3.1 Distribution Diagnostics
Diagnostic plots confirm that the Beta-mixture models effectively capture the zero-inflation and skewness of the observational data.

*Figure 1: Distribution diagnostics for Forest Age (LU 44) and gHM (LU 11).*
![Distribution Diagnostic](figures/diagnostic_forest_age_44_1_1.png)

### 3.2 Scenario Comparisons
The synthesized maps reflect the interplay between changing land use patterns and local baseline characteristics.

*Figure 2: Global maps of estimated Forest Age across Historical, Future (SSP3-7.0), and Intact scenarios.*
![Forest Age Change Map](figures/change_map_forest_age.png)

*Figure 3: Global maps of estimated gHM across scenarios.*
![gHM Change Map](figures/change_map_gHM.png)

## 4. Validation and Performance

Synthetic variables for the 1991–2020 baseline period were validated against the original observational datasets at the global scale. The performance was evaluated using three key metrics: Mean Bias, Root Mean Square Error (RMSE), and the Pearson Correlation Coefficient ($r$).

### 4.1 Summary of Agreement
The table below summarizes the statistical agreement between observations (Obs) and synthesized (Syn) features.

| Variable | Mean (Obs) | Mean (Syn) | Bias | RMSE | Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **gHM** [0-1] | 0.168 | 0.127 | -0.041 | 0.125 | 0.831 |
| **Irrigated Fraction** [0-1] | 0.059 | 0.055 | -0.004 | 0.055 | 0.958 |
| **Rainfed Fraction** [0-1] | 0.099 | 0.074 | -0.025 | 0.128 | 0.838 |
| **Forest Age** [years] | 97.85 | 98.18 | +0.333 | 26.74 | 0.946 |

### 4.2 Discussion of Results
The high correlation coefficients ($r \ge 0.83$) across all variables confirm that the rank-based statistical transfer successfully preserves the primary spatial gradients and local anomalies present in the baseline data. 

Specifically:
- **Irrigated Fraction** shows near-perfect agreement ($r=0.96$), largely due to its strong dependence on categorical land use classes (Cropland) and climate clusters.
- **Forest Age** ($r=0.95$) significantly benefited from the inclusion of the dual point-mass mixture model, which correctly handled the 300-year saturation cap present in the input data. The negligible bias (+0.33 years) indicates that the Beta distributions accurately captured the central tendencies of forest stand ages.
- **gHM** and **Rainfed Fraction** exhibit strong spatial consistency ($r \approx 0.83-0.84$), with slight negative biases reflecting the conservative nature of the Beta-mixture fitting in high-variance agricultural regions.

Overall, the synthesized predictors provide a robust and spatially coherent basis for global-scale $S_R$ modeling across all historical and future scenarios.
