import os
import xarray as xr
import numpy as np
import pandas as pd

DIR = "/home/vajira/SRchange/output_SR/thresholds_analysis/scenarios_probabilistic"
files = sorted([f for f in os.listdir(DIR) if f.endswith(".nc")])

results = []
for f in files:
    ds = xr.open_dataset(os.path.join(DIR, f))
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].values
    
    # Basic stats
    valid_mask = ~np.isnan(data)
    if not valid_mask.any(): continue
    
    mean_p = np.mean(data[valid_mask])
    pct_high_conf = np.sum(data[valid_mask] > 0.8) / np.sum(valid_mask) * 100
    
    # Parse filename
    # Format: prob_sat_c{cid}_{f1}_{f2}_{scn_name}.nc
    parts = f.replace(".nc", "").split("_")
    # cid is at index 2
    cid = parts[2]
    # scenario is at the end
    if "ssp" in f:
        scn = "_".join(parts[-2:])
    else:
        scn = "_".join(parts[-2:])
        
    results.append({
        "File": f,
        "Cluster": cid,
        "Scenario": scn,
        "Mean_Prob": mean_p,
        "Pct_Saturated_HC": pct_high_conf
    })

df = pd.DataFrame(results)
# Sort for readability
df = df.sort_values(["Cluster", "Scenario"])
print(df.to_string(index=False))
