import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Setup paths
csv_path = "/home/vajira/SRchange/output_SR/sr_scn_stats_k6.csv"
output_dir = Path("/home/vajira/SRchange/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Define periods and their sequence
periods = [
    "intact", 
    "pre-industrial", 
    "1901_1930", 
    "1931_1960", 
    "1961_1990", 
    "1991_2020"
]

# Mapping for display
period_labels = {
    "intact": "Intact",
    "pre-industrial": "Pre-industrial",
    "1901_1930": "1901-1930",
    "1931_1960": "1931-1960",
    "1961_1990": "1961-1990",
    "1991_2020": "1991-2020"
}

# We need to handle 'intact' and 'pre-industrial' as baseline states.
# In the CSV, climate for these is usually 1901_1930 (checked in head).
# For the purpose of the plot:
# 1. Intact: climate=1901_1930, LU=intact
# 2. PI: climate=1901_1930, LU=pre-industrial
# 3. 1901-1930: climate=1901_1930, LU=1901_1930
# ... and so on.

def get_sr(clim, lu):
    val = df[(df['climate'] == clim) & (df['LU'] == lu)]['mean_weighted']
    if len(val) == 0:
        # Fallback for intact/PI if 1901_1930 isn't the climate label used
        # (Though head showed 1901_1930_intact)
        return None
    return val.values[0]

# Calculate components for the bar plot
# We want to show:
# Baseline SR for period i
# Change due to Climate (from i to i+1)
# Change due to LU (from i to i+1)

results = []
current_clim = "1901_1930" # Baseline climate for Intact/PI

# Step 1: Intact
sr_intact = get_sr("1901_1930", "intact")
results.append({
    "period": "intact",
    "base_sr": sr_intact,
    "clim_change": 0,
    "lu_change": 0,
    "total_sr": sr_intact
})

# Step 2: Pre-industrial
# Change from Intact to PI is purely LU
sr_pi = get_sr("1901_1930", "pre-industrial")
results.append({
    "period": "pre-industrial",
    "base_sr": sr_intact,
    "clim_change": 0,
    "lu_change": sr_pi - sr_intact,
    "total_sr": sr_pi
})

# Step 3: 1901-1930
# Change from PI to 1901_1930 is purely LU (Climate still 1901_1930)
sr_1901 = get_sr("1901_1930", "1901_1930")
results.append({
    "period": "1901_1930",
    "base_sr": sr_pi,
    "clim_change": 0,
    "lu_change": sr_1901 - sr_pi,
    "total_sr": sr_1901
})

# Steps 4-6: Historical transitions
historical_periods = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
for i in range(len(historical_periods) - 1):
    p_prev = historical_periods[i]
    p_curr = historical_periods[i+1]
    
    sr_prev_all = get_sr(p_prev, p_prev)
    sr_curr_all = get_sr(p_curr, p_curr)
    
    # Climate change: (Clim_curr, LU_prev) - (Clim_prev, LU_prev)
    sr_clim_effect = get_sr(p_curr, p_prev) - get_sr(p_prev, p_prev)
    
    # LU change: (Clim_curr, LU_curr) - (Clim_curr, LU_prev)
    sr_lu_effect = get_sr(p_curr, p_curr) - get_sr(p_curr, p_prev)
    
    results.append({
        "period": p_curr,
        "base_sr": sr_prev_all,
        "clim_change": sr_clim_effect,
        "lu_change": sr_lu_effect,
        "total_sr": sr_curr_all
    })

plot_df = pd.DataFrame(results)

# Plotting
fig, ax = plt.subplots(figsize=(11, 7))
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5)

x = np.arange(len(plot_df))
width = 0.65

# Colors - Professional palette
color_base = '#f0f0f0'
color_clim = '#e31a1c' # Red for climate
color_lu = '#1f78b4'   # Blue for LU

# Plot bars
# Base bars
ax.bar(x, plot_df['base_sr'], width, label='Initial $S_R$', color=color_base, edgecolor='dimgray', linewidth=0.8)

# Climate change contribution
clim_bottoms = plot_df['base_sr'].values
ax.bar(x, plot_df['clim_change'], width, bottom=clim_bottoms, 
        label='$\Delta S_R$ (Climate)', color=color_clim, edgecolor='black', linewidth=0.5)

# LU change contribution
lu_bottoms = clim_bottoms + plot_df['clim_change'].values
ax.bar(x, plot_df['lu_change'], width, bottom=lu_bottoms, 
        label='$\Delta S_R$ (Land Use)', color=color_lu, edgecolor='black', linewidth=0.5)

# Add error bars or indicators if available? (CSV has std_weighted but it's per scenario, not delta)
# For now, stick to the clear attribution.

# Final SR markers
ax.scatter(x, plot_df['total_sr'], color='black', marker='_', s=1000, linewidths=2.5, zorder=5, label='Net $S_R$')

# Aesthetics
ax.set_xticks(x)
ax.set_xticklabels([period_labels[p] for p in plot_df['period']], rotation=0)
ax.set_ylabel("$S_R$ (mm)", fontweight='bold')
ax.set_title("Attribution of Global Mean $S_R$ Change", fontsize=20, pad=25, fontweight='bold')

# Legend
handles, labels = ax.get_legend_handles_labels()
# Reorder legend to match visual stack
order = [3, 2, 1, 0]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
          frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
sns.despine(offset=10)

# Add net change labels
for i in range(1, len(plot_df)):
    diff = plot_df.loc[i, 'total_sr'] - plot_df.loc[i-1, 'total_sr']
    color = 'darkgreen' if diff > 0 else 'darkred'
    prefix = '+' if diff > 0 else ''
    ax.text(i, max(plot_df.loc[i, 'total_sr'], plot_df.loc[i, 'base_sr']) + 5, 
            f"{prefix}{diff:.1f}", ha='center', color=color, fontweight='bold', fontsize=12)

# Set Y axis limits to focus on change
ymin = plot_df[['base_sr', 'total_sr']].min().min() - 15
ymax = plot_df[['base_sr', 'total_sr']].max().max() + 20
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(output_dir / "sr_change_attribution_global.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "sr_change_attribution_global.pdf", bbox_inches='tight')

print(f"Plot saved to {output_dir}")
