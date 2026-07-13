import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Setup paths
csv_path = "/home/vajira/SRchange/output_SR/sr_scn_stats_with_uncertainty_k6.csv"
output_dir = Path("/home/vajira/SRchange/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Toggle for error bars
SHOW_ERROR_BARS = False

# Load data
df = pd.read_csv(csv_path)

# Mapping for display
period_labels = {
    "intact": "Intact",
    "pre-industrial": "Pre-industrial",
    "1901_1930": "1901-1930",
    "1931_1960": "1931-1960",
    "1961_1990": "1961-1990",
    "1991_2020": "1991-2020"
}

historical_periods = ["intact", "pre-industrial", "1901_1930", "1931_1960", "1961_1990", "1991_2020"]

def get_sr(clim, lu):
    # Map 'intact' and 'pre-industrial' LU to 1901_1930 climate if not explicit
    c = "1901_1930" if clim in ["intact", "pre-industrial"] else clim
    l = lu
    row = df[(df['climate'] == c) & (df['LU'] == l)]
    if len(row) > 0:
        return row['mean_weighted_ensemble'].values[0], row['ci95_weighted_ensemble'].values[0]
    return None, None

# Calculate transitions
# Transition i -> i+1
# Global Mean (i)
# Climate effect: (Clim_i+1, LU_i) - (Clim_i, LU_i)
# LU effect: (Clim_i, LU_i+1) - (Clim_i, LU_i)
# Interaction/Residual: Total_Change - (Clim_effect + LU_effect)
# where Total_Change = (Clim_i+1, LU_i+1) - (Clim_i, LU_i)

data = []
for i in range(len(historical_periods)):
    p_curr = historical_periods[i]
    sr_curr, std_curr = get_sr(p_curr, p_curr)
    
    entry = {
        "period": p_curr,
        "sr_mean": sr_curr,
        "sr_std": std_curr,
        "clim_delta": 0,
        "lu_delta": 0,
        "inter_delta": 0
    }
    
    if i < len(historical_periods) - 1:
        p_next = historical_periods[i+1]
        sr_next, _ = get_sr(p_next, p_next)
        
        # Isolate effects
        # Delta Climate = (Clim_next, LU_curr) - (Clim_curr, LU_curr)
        sr_clim_next_lu_curr, _ = get_sr(p_next, p_curr)
        d_clim = sr_clim_next_lu_curr - sr_curr
        
        # Delta LU = (Clim_curr, LU_next) - (Clim_curr, LU_curr)
        sr_clim_curr_lu_next, _ = get_sr(p_curr, p_next)
        d_lu = sr_clim_curr_lu_next - sr_curr
        
        # Interaction = Total Change - (d_clim + d_lu)
        total_delta = sr_next - sr_curr
        d_inter = total_delta - (d_clim + d_lu)
        
        entry.update({
            "clim_delta": d_clim,
            "lu_delta": d_lu,
            "inter_delta": d_inter
        })
    
    data.append(entry)

plot_df = pd.DataFrame(data)

# Visualizing as a Step-wise Attribution Plot
plt.figure(figsize=(14, 8))
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.6)

x = np.arange(len(plot_df))
width = 0.2

color_mean = "#525252"
color_clim = "#e31a1c"
color_lu = "#1f78b4"
color_inter = "#feb24c"

# Plotting each period
for i in range(len(plot_df)):
    curr_sr = plot_df.loc[i, 'sr_mean']
    curr_std = plot_df.loc[i, 'sr_std']
    
    # Position for the main bar
    x_base = i
    
    # 1. Main SR bar
    plt.bar(x_base, curr_sr, width*1.5, color=color_mean, edgecolor='black', alpha=0.9, label=r'Mean $S_R$' if i==0 else "", zorder=3)
    
    # Add error bar for global mean (optional)
    if SHOW_ERROR_BARS:
        plt.errorbar(x_base, curr_sr, yerr=curr_std, fmt='none', ecolor='black', capsize=5, zorder=6)
        label_text = f"{curr_sr:.1f} ± {curr_std:.1f}"
    else:
        label_text = f"{curr_sr:.1f}"
        
    plt.text(x_base, curr_sr + 0.4, label_text, ha='center', va='bottom', fontweight='bold', fontsize=14, zorder=4)
    
    # 2. Change bars for the transition to NEXT period (if not last)
    if i < len(plot_df) - 1:
        d_clim = plot_df.loc[i, 'clim_delta']
        d_lu = plot_df.loc[i, 'lu_delta']
        d_inter = plot_df.loc[i, 'inter_delta']
        
        # Positions for change bars - placed between period i and i+1
        x_step_start = i + 0.35
        step_w = 0.18
        
        x_clim = x_step_start
        x_lu = x_step_start + step_w
        x_inter = x_step_start + 2 * step_w
        
        # All bars start from current mean SR (not stacked on each other)
        # Climate bar
        plt.bar(x_clim, d_clim, step_w, bottom=curr_sr, color=color_clim, edgecolor='black', 
                label=r'$\Delta S_{R-Climate}$' if i==0 else "", zorder=3)
        
        # LU bar
        plt.bar(x_lu, d_lu, step_w, bottom=curr_sr, color=color_lu, edgecolor='black', 
                label=r'$\Delta S_{R-LU}$' if i==0 else "", zorder=3)
        
        # Interaction bar
        plt.bar(x_inter, d_inter, step_w, bottom=curr_sr, color=color_inter, edgecolor='black', 
                label=r'$\Delta S_{R-Interaction}$' if i==0 else "", zorder=3)
        
        # Add labels for deltas - positioned above/below the bar for visibility
        for val, x_pos in zip([d_clim, d_lu, d_inter], [x_clim, x_lu, x_inter]):
            if abs(val) > 0.01:
                # Calculate vertical offset
                y_offset = 0.7 if val >= 0 else -0.7
                va = 'bottom' if val >= 0 else 'top'
                
                plt.text(x_pos, curr_sr + val + y_offset, f"{val:+.1f}", 
                         ha='center', va=va, fontsize=14, 
                         color='black', rotation=0, zorder=5) # fontweight='bold' 8

# Aesthetics
plt.xticks(x, [period_labels[p] for p in plot_df['period']], fontsize=18) # fontweight='bold'
plt.ylabel("$S_R$ (mm)", fontsize=22) # fontweight='bold'
plt.yticks(fontsize=18)
# plt.title("Change of global mean $S_R$ over time", fontsize=26, pad=25) # fontweight='bold' 20

plt.legend(frameon=False, loc='upper center', ncols=4, fontsize=18) #, bbox_to_anchor=(1, 1))
sns.despine(offset=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Adjust y-axis to focus on the top part
ymin = plot_df['sr_mean'].min() - 10
ymax = (plot_df['sr_mean'].max() + 5)
plt.ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(output_dir / "sr_change_step_attribution.png", dpi=150, bbox_inches='tight')
plt.savefig(output_dir / "sr_change_step_attribution.svg", bbox_inches='tight')

print(f"Plot saved to {output_dir}")
