import pandas as pd
import numpy as np
from pathlib import Path

def calculate_scenario_uncertainty():
    # Load per-model stats
    input_csv = '/home/vajira/SRchange/output_SR/sr_scn_per_model_stats_k6_masked.csv'
    output_csv = '/home/vajira/SRchange/output_SR/sr_scn_stats_with_uncertainty_k6_masked.csv'
    
    if not Path(input_csv).exists():
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Group by scenario/climate/LU and calculate stats across the 25 models
    # We want the mean of the means, and the uncertainty (CI) of that mean.
    # Note: df['mean_weighted'] already contains the area-weighted mean for each model.
    
    results = []
    
    # Grouping to ensure we keep climate and LU labels
    grouped = df.groupby(['scenario', 'climate', 'LU'])
    
    for (scn, clim, lu), group in grouped:
        vals = group['mean_weighted'].values
        n = len(vals)
        
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        sem_val = std_val / np.sqrt(n)
        ci95_val = 1.96 * sem_val
        
        results.append({
            'scenario': scn,
            'climate': clim,
            'LU': lu,
            'mean_weighted_ensemble': mean_val,
            'std_weighted_ensemble': std_val,
            'sem_weighted_ensemble': sem_val,
            'ci95_weighted_ensemble': ci95_val,
            'n_members': n
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"Scenario stats with ensemble uncertainty saved to {output_csv}")

    # Comparison with old stats (if file exists)
    old_csv = '/home/vajira/SRchange/output_SR/sr_scn_stats_k6.csv'
    if Path(old_csv).exists():
        old_df = pd.read_csv(old_csv)
        # The old CSV used labels like '1901_1930_1901_1930' instead of 'CLIM_...'
        # Let's standardize the lookup
        print("\n--- Comparison with old stats (mean_weighted) ---")
        for idx, row in res_df.iterrows():
            # Construct old-style scenario label
            # Note: The old labels seem to be clim_lu
            old_label = f"{row['climate']}_{row['LU']}"
            if old_label in old_df['scenario'].values:
                new_v = row['mean_weighted_ensemble']
                old_v = old_df[old_df['scenario'] == old_label]['mean_weighted'].values[0]
                diff = new_v - old_v
                # Just show the first 10 for comparison
                if idx < 10:
                    print(f"Scenario: {old_label}")
                    print(f"  New (Robust Weighted): {new_v:.4f}")
                    print(f"  Old (Simple Weighted): {old_v:.4f}")
                    print(f"  Difference:            {diff:.4f}")

if __name__ == "__main__":
    calculate_scenario_uncertainty()
