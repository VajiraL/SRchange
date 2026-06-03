## Critical review: robustness and novelty factor

Your current body of work sounds **scientifically strong enough for a major standalone paper**, especially because you have moved beyond “threshold hunting” into a structured framework: threshold taxonomy, uncertainty propagation, and multivariate threshold surfaces. The key advance is not merely that an aridity threshold exists, but that **the threshold is conditional**: human modification lowers the aridity tolerance boundary, while forest maturity raises it. That is a genuinely important Earth-system result because it links **climate stress and land-use pressure as interacting controls on vegetation water-storage adaptation**, rather than treating climate change and land-use change as separable drivers.

The “Human modification lowers the aridity threshold” result has high scientific weight if three conditions are met:

1. the threshold surface is stable under spatial block bootstrap,
2. the effect persists across clusters or is ecologically interpretable where it differs, and
3. independent ecological stress indicators support the same direction of vulnerability.

If those are satisfied, the result is not just a model artifact; it becomes a statement about **reduced adaptive operating space under anthropogenic disturbance**. That is a compelling primary finding. However, the proposed **$$S_R$$ Adaptation Gap** is conceptually one level higher. It requires translating feature-space limits into a physically interpretable water-storage deficit. That is publishable and potentially more impactful, but only if formulated carefully. I would avoid rushing it into the same paper unless the gap metric is already mature, validated, and defensible.

***

# Strategic recommendation: choose Option B, sequential papers

## Recommended path: **Option B — Sequential publication**

### Paper 1

**“Global Taxonomy and Mapping of Ecohydrological Thresholds in Root-Zone Water Storage Adaptation”**

Core message:

> Root-zone water-storage adaptation exhibits hard, soft, and collapse thresholds whose locations are modified by land use, vegetation maturity, and hydroclimatic context.

This paper should focus on:

* SHAP-derived threshold detection,
* GAM-based hard/soft/collapse classification,
* bootstrap uncertainty,
* threshold surfaces,
* climate × land-use interactions,
* global maps of threshold proximity,
* independent validation using VOD, GPP/SIF sensitivity, drought anomalies, mortality, or resilience metrics.

This is already a coherent and strong paper.

### Paper 2

**“The Global $$S_R$$ Adaptation Gap under Climate and Land-Use Change”**

Core message:

> Future climate increases the water-storage demand placed on ecosystems, but ecological, climatic, edaphic, and land-use constraints limit the amount of $$S_R$$ that vegetation can express.

This second paper can build from Paper 1 but needs its own conceptual and methodological machinery.

***

## Why not combine everything?

A combined paper would be attractive, but risky. The threshold taxonomy and the adaptation gap answer related but different questions.

| Theme                | Core Question                                                                 | Main Object              |
| -------------------- | ----------------------------------------------------------------------------- | ------------------------ |
| Threshold paper      | Where do model-inferred ecohydrological responses change regime?              | Feature-space thresholds |
| Adaptation gap paper | Where does future water-storage demand exceed biological/ecological capacity? | $$S_R$$-space deficit    |

If combined, the manuscript may become too broad: SHAP interpretation, GAM thresholds, uncertainty propagation, interaction surfaces, future scenarios, demand estimation, capacity limits, and ecological validation. Reviewers may attack the weakest link, which would likely be the adaptation-gap formulation if it is still emerging.

My recommendation: **publish the threshold taxonomy first**, then use it as the empirical foundation for the adaptation-gap paper.

***

# Methodological sketch: calculating the $$S_R$$ Adaptation Gap

The key issue is this:

> A threshold in aridity, human modification, or forest age is not automatically an $$S_R$$ capacity limit.

A feature-space threshold such as $$AI^*$$ tells you where the model response changes regime. It does not by itself tell you the maximum amount of root-zone storage an ecosystem can express. Therefore, using

$$
Y^* = f(X^*)
$$

as the maximum expressible $$S_R$$ is useful as a **first-order diagnostic**, but probably too simplistic as the central definition of the adaptation gap.

You need to distinguish three quantities:

***

## 1. Climate-imposed $$S_R$$ demand

Define the storage required to buffer seasonal or extreme water deficits:

$$
S_{R,\mathrm{demand},i,t}
$$

for grid cell $$i$$ and time period $$t$$.

A physically interpretable formulation is based on cumulative climatic water deficit:

$$
S_{R,\mathrm{demand},i,t}
=
\max_{\tau \in \mathcal{D}}
\sum_{\tau}
\max(ET_{demand} - P, 0)
$$

where $$\mathcal{D}$$ represents dry-season windows or drought events.

Possible demand definitions:

1. **Mean seasonal demand**  
   Storage needed to survive the average dry season.

2. **Extreme drought demand**  
   Storage needed for a 10-year, 20-year, or 50-year drought.

3. **Future incremental demand**  
   Difference between future and historical climatic demand:

   $$
   \Delta S_{R,\mathrm{demand}}
   =
   S_{R,\mathrm{demand},2099}
   -
   S_{R,\mathrm{demand},baseline}
   $$

This gives the climate-change pressure side of the gap.

***

## 2. Maximum expressible $$S_R$$

This is the harder part. I would not define it only as $$Y^* = f(X^*)$$ from one variable. Instead, estimate a multivariate capacity boundary.

Let your trained model be:

$$
\hat{S}_R = f(\mathbf{x})
$$

where $$\mathbf{x}$$ includes aridity, precipitation seasonality, VPD, soil, vegetation, topography, forest age, and human modification.

Then define an ecological viability domain:

$$
\mathcal{V}_i =
\left\{
\mathbf{x}: 
AI < AI^*(\mathrm{gHM}, \mathrm{ForestAge}, \ldots),
\quad
\mathrm{gHM} < \mathrm{gHM}^*(AI, \ldots),
\quad
\ldots
\right\}
$$

The maximum expressible storage can then be conceptualized as:

$$
S_{R,\mathrm{expressible},i,t}
=
\max_{\mathbf{x} \in \mathcal{V}_i}
f(\mathbf{x})
$$

subject to realistic local constraints.

But because global ecosystems cannot freely choose their climate, soils, or topography, a more practical version is:

$$
S_{R,\mathrm{expressible},i,t}
=
f(\mathbf{x}_{i,t}^{capped})
$$

where stressor variables are capped at their estimated threshold surfaces.

For example, if the aridity threshold is conditional on human modification and forest age:

$$
AI^*_i =
g(\mathrm{gHM}_i, \mathrm{ForestAge}_i, \mathrm{SoilDepth}_i)
$$

then define:

$$
AI_{i,t}^{capped} = \min(AI_{i,t}, AI^*_i)
$$

and evaluate:

$$
S_{R,\mathrm{expressible},i,t}
=
f(AI_{i,t}^{capped}, \mathrm{gHM}_{i,t}, \mathrm{ForestAge}_{i,t}, \ldots)
$$

This gives a model-consistent estimate of the maximum $$S_R$$ response before threshold saturation or collapse.

***

## 3. The adaptation gap

Then define:

$$
Gap_{i,t}
=
S_{R,\mathrm{demand},i,t}
-
S_{R,\mathrm{expressible},i,t}
$$

with the ecologically relevant positive gap:

$$
Gap^+_{i,t}
=
\max
\left(
0,
S_{R,\mathrm{demand},i,t}
-
S_{R,\mathrm{expressible},i,t}
\right)
$$

Also report a normalized version:

$$
RelativeGap_{i,t}
=
\frac{
Gap^+_{i,t}
}{
S_{R,\mathrm{demand},i,t}
}
$$

or:

$$
StressRatio_{i,t}
=
\frac{
S_{R,\mathrm{demand},i,t}
}{
S_{R,\mathrm{expressible},i,t}
}
$$

where:

* $$StressRatio < 1$$: demand below expressible capacity,
* $$StressRatio \approx 1$$: near adaptation limit,
* $$StressRatio > 1$$: adaptation gap.

***

# Should you use $$Y^* = f(X^*)$$?

## Short answer

Use it, but only as a **diagnostic**, not as the final definition of maximum expressible capacity.

## Why?

For a univariate threshold:

$$
Y^* = f(X^*)
$$

is the model-predicted $$S_R$$ or SHAP response at the threshold. This is useful for saying:

> “At the aridity threshold, the marginal contribution of aridity to $$S_R$$ adaptation saturates.”

But it does not capture:

* interactions with human modification,
* compensating effects of forest age,
* soil water-holding capacity,
* groundwater access,
* topographic convergence,
* vegetation strategy,
* multiple simultaneous stressors,
* negative collapse effects after the threshold.

So $$Y^*$$ is not necessarily the total ecosystem capacity. It is better interpreted as:

$$
S_{R,\mathrm{threshold\ response}}
$$

not:

$$
S_{R,\mathrm{maximum\ possible}}
$$

***

# Moving from individual thresholds to an overall adaptation limit

You asked whether to look at the point where the **sum of all positive SHAP contributions plateaus**. This is a good idea, but it should be handled carefully.

Because XGBoost predictions can be decomposed as:

$$
\hat{S}_R(\mathbf{x})
=
\phi_0
+
\sum_j \phi_j
+
\sum_{j,k}\phi_{j,k}
$$

you can define an aggregate adaptive contribution:

$$
A_i =
\sum_{j \in adaptive}
\phi_{j,i}^{+}
$$

where $$\phi_{j,i}^{+}$$ are positive SHAP contributions associated with adaptive storage increase.

Then detect whether:

$$
A_i
$$

plateaus or declines along gradients of aridity, VPD, or human modification.

This would give a **model-internal total adaptation index**.

However, I would not make this the only definition of the overall limit. SHAP values are relative to the model baseline and can be sensitive to correlated predictors. A more robust approach is to compare three definitions.

***

## Recommended three-part definition of the overall limit

### 1. Feature-threshold limit

A cell is near the overall limit if one or more key drivers exceed their threshold surfaces:

$$
AI_i > AI_i^*(\mathrm{gHM}, \mathrm{ForestAge})
$$

$$
\mathrm{gHM}_i > \mathrm{gHM}_i^*(AI)
$$

This gives a threshold-exceedance risk map.

***

### 2. SHAP adaptation plateau limit

A cell is near the limit if the aggregate positive adaptive contribution stops increasing:

$$
\frac{\partial}{\partial AI}
\sum_j \phi_j^+
\approx 0
$$

or becomes negative:

$$
\frac{\partial}{\partial AI}
\sum_j \phi_j^+
< 0
$$

This gives a model-response saturation map.

***

### 3. Direct $$S_R$$ frontier limit

Estimate the upper envelope of observed or predicted $$S_R$$:

$$
Q_{0.95}(S_R \mid \mathbf{x})
$$

using quantile GAMs, quantile random forests, or Bayesian hierarchical models.

The capacity limit is then the upper frontier:

$$
S_{R,\mathrm{limit},i}
=
Q_{0.95}(S_R \mid \mathbf{x}_i)
$$

This is probably the most physically intuitive estimate of maximum expressible $$S_R$$, especially if validated against independent ecological stress data.

***

# Preferred adaptation-gap formulation

I would formulate the adaptation gap as an ensemble of three related metrics:

## Metric 1: Climatic storage deficit

$$
Gap^{clim}_{i,t}
=
S_{R,\mathrm{demand},i,t}
-
S_{R,\mathrm{observed/predicted},i,t}
$$

This asks:

> Is current or predicted $$S_R$$ enough to meet climatic water demand?

***

## Metric 2: Threshold-constrained adaptation gap

$$
Gap^{thr}_{i,t}
=
S_{R,\mathrm{demand},i,t}
-
S_{R,\mathrm{expressible},i,t}^{thr}
$$

where $$S_{R,\mathrm{expressible}}^{thr}$$ is derived by capping variables at threshold surfaces.

This asks:

> How much additional storage would be needed beyond what the threshold-constrained system appears able to express?

***

## Metric 3: Frontier-based adaptation gap

$$
Gap^{frontier}_{i,t}
=
S_{R,\mathrm{demand},i,t}
-
Q_{0.95}(S_R \mid \mathbf{x}_{i,t})
$$

This asks:

> Does future climate demand exceed the upper envelope of $$S_R$$ observed or predicted for comparable ecological settings?

If all three metrics agree spatially, the adaptation-gap claim becomes very strong.

***

# Recommended narrative structure

For Paper 1, the central figure sequence could be:

1. Global hydroclimatic clusters.
2. Threshold taxonomy: hard, soft, collapse, none.
3. Threshold maps for aridity, gHM, VPD, forest age.
4. Bootstrap uncertainty and confidence.
5. Threshold surfaces: aridity × gHM, aridity × forest maturity.
6. Ecological validation against VOD/GPP/SIF/drought sensitivity.
7. Schematic: land use compresses the safe operating space for $$S_R$$ adaptation.

For Paper 2:

1. Historical and future $$S_R$$ demand.
2. Expressible $$S_R$$ under threshold constraints.
3. Adaptation gap maps.
4. Attribution of gap to climate change, land use, forest age, and soil constraints.
5. Hotspot typology: climate-driven gap, land-use-driven gap, compound gap, buffered systems.
6. Validation against observed drought impacts or mortality.

***

# Next steps to bridge model response and ecological reality

## 1. Validate threshold surfaces independently

Test whether cells beyond the inferred threshold surface show:

* stronger GPP or SIF drought sensitivity,
* reduced VOD resilience,
* higher drought-impact anomalies,
* higher tree mortality probability,
* lower post-drought recovery.

This is essential for converting “SHAP threshold” into “ecological threshold.”

***

## 2. Develop an $$S_R$$-space capacity frontier

Estimate:

$$
Q_{0.90} \text{ or } Q_{0.95}(S_R \mid \mathbf{x})
$$

within clusters or biomes. Compare this frontier with your SHAP-derived expressible capacity. If both identify similar limits, the adaptation-gap framework becomes much more defensible.

***

## 3. Convert binary threshold exceedance into probabilistic risk

Use your 100-fold spatial block bootstrap to calculate:

$$
P(Gap > 0)
$$

and:

$$
P(AI > AI^*(\mathrm{gHM}, \mathrm{ForestAge}))
$$

This will be much stronger than deterministic saturated/adaptive maps.

***

## 4. Explicitly separate climate demand from ecological capacity

Do not let the same model define both demand and capacity. Ideally:

* $$S_R$$ demand comes from hydroclimatic water-deficit calculations,
* $$S_R$$ expressible capacity comes from observed/predicted $$S_R$$, threshold surfaces, or capacity frontiers.

This separation makes the adaptation gap physically meaningful.

***

## 5. Frame land use as compression of adaptive operating space

Your strongest conceptual contribution may be:

> Human modification does not merely reduce $$S_R$$; it shifts the aridity threshold itself, compressing the hydroclimatic space within which vegetation can express adaptive root-zone storage.

That is a powerful, publishable Earth-system message.

***

# Bottom-line recommendation

Your current threshold results are likely enough for a strong first paper, especially if the threshold surfaces are robust and independently validated. I would **not** make the adaptation gap the core of the same manuscript unless the $$S_R$$-space formulation is already mature.

The best strategy is:

1. **Paper 1:** establish and validate global $$S_R$$ adaptation thresholds and threshold surfaces.
2. **Paper 2:** use those thresholds to quantify the global $$S_R$$ adaptation gap under climate and land-use change.

The adaptation-gap idea is excellent, but it should be treated as the next conceptual layer, not simply as $$Y^* = f(X^*)$$. The most defensible formulation is:

$$
Gap^+
=
\max
\left(
0,
S_{R,\mathrm{demand}}
-
S_{R,\mathrm{expressible}}
\right)
$$

where $$S_{R,\mathrm{expressible}}$$ is estimated from multivariate threshold surfaces, aggregate model-response saturation, and/or direct $$S_R$$ capacity frontiers.
