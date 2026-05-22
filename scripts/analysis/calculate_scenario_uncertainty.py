import pandas as pd
import numpy as np
from pathlib import Path

def calculate_scenario_uncertainty():
    # Load per-model stats (using static mask)
    input_csv = '/home/vajira/SRchange/output_SR/sr_scn_per_model_stats_k6.csv'
    output_csv = '/home/vajira/SRchange/output_SR/sr_scn_stats_with_uncertainty_k6.csv'
    
    if not Path(input_csv).exists():
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Group by scenario/climate/LU and calculate stats across the 25 models
    results = []
    
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
    print(f"Scenario stats with ensemble uncertainty (static mask) saved to {output_csv}")

if __name__ == "__main__":
    calculate_scenario_uncertainty()
