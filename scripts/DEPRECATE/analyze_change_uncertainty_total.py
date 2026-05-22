# Calculate the change in SR and its attribution to climate and LU changes, along with uncertainty estimates, using the robust area-weighted per-model stats.
# This is similar to the incremental version but for getting change factors between two any given time periods including intact, pre-industrial, 4 past periods, and 2 future periods under 4 SSPs.
# take two time periods as parameters and calculate the change in SR and its attribution to climate and LU changes, along with uncertainty estimates, using the robust area-weighted per-model stats.

import numpy as np
import pandas as pd
from pathlib import Path

def calculate_change_uncertainty(period_base=None, period_target=None):
    # Load per-model stats (already computed with robust area-weighting)
    input_csv = '/home/vajira/SRchange/output_SR/sr_scn_per_model_stats_k6_masked.csv'
    output_csv = f'/home/vajira/SRchange/output_SR/sr_change_uncertainty_{period_base}_to_{period_target}.csv'
    
    if not Path(input_csv).exists():
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Periods for decomposition
    all_periods = ["intact", "pre-industrial",
                   "1901_1930", "1931_1960", "1961_1990", "1991_2020",
                   "2041_2070_ssp126", "2071_2099_ssp126",
                   "2041_2070_ssp245", "2071_2100_ssp245",
                   "2041_2070_ssp370", "2071_2100_ssp370",
                   "2041_2070_ssp460", "2071_2100_ssp460"]
    
    # If no periods specified, process all successive transitions
    if period_base is None or period_target is None:
        periods_to_process = [(all_periods[i], all_periods[i+1]) for i in range(len(all_periods) - 1)]
    else:
        periods_to_process = [(period_base, period_target)]
    
    def get_val(clim, lu, ens, fold):
        # Climate for intact/PI is assumed to be 1901_1930 baseline
        c = "1901_1930" if clim in ["intact", "pre-industrial"] else clim
        mask = (df['climate'] == c) & (df['LU'] == lu) & (df['ensemble'] == ens) & (df['fold'] == fold)
        val = df[mask]['mean_weighted']
        return val.values[0] if len(val) > 0 else np.nan

    results = []
    
    for p_curr, p_next in periods_to_process:
        
        deltas = {
            'net': [],
            'clim': [],
            'lu': [],
            'inter': []
        }
        
        # Calculate for each individual model (25 members)
        for ens in range(1, 6):
            for fold in range(1, 6):
                s0 = get_val(p_curr, p_curr, ens, fold)
                s1 = get_val(p_next, p_next, ens, fold)
                sc = get_val(p_next, p_curr, ens, fold) # Effect of Climate change: (Clim_next, LU_curr)
                sl = get_val(p_curr, p_next, ens, fold) # Effect of LU change: (Clim_curr, LU_next)
                
                if np.isnan([s0, s1, sc, sl]).any():
                    continue
                
                d_net = s1 - s0
                d_clim = sc - s0
                d_lu = sl - s0
                d_inter = d_net - (d_clim + d_lu)
                
                deltas['net'].append(d_net)
                deltas['clim'].append(d_clim)
                deltas['lu'].append(d_lu)
                deltas['inter'].append(d_inter)
        
        # Aggregate across the available members (ideally 25)
        for component, vals in deltas.items():
            if not vals: continue
            
            n = len(vals)
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            sem = std / np.sqrt(n)
            ci95 = 1.96 * sem
            
            results.append({
                'transition': f"{p_curr} -> {p_next}",
                'component': component,
                'mean_change': mean,
                'std': std,
                'sem': sem,
                'ci95': ci95,
                'min': np.min(vals),
                'max': np.max(vals),
                'n_members': n
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    
    print("\n--- SR CHANGE ATTRIBUTION WITH UNCERTAINTY (ROBUST WEIGHTING) ---")
    print(f"Results saved to {output_csv}")
    
    summary = res_df.pivot(index='transition', columns='component', values='mean_change')
    print("\nMean Changes (mm):")
    print(summary[['clim', 'lu', 'inter', 'net']])
    
    summary_ci = res_df.pivot(index='transition', columns='component', values='ci95')
    print("\n95% Confidence Intervals (± mm):")
    print(summary_ci[['clim', 'lu', 'inter', 'net']])

if __name__ == "__main__":
    calculate_change_uncertainty(period_base="pre-industrial", period_target="1991_2020")
