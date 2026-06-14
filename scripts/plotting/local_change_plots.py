"""
plot_change_decomposition.py
----------------------------
Visualizes the results of explain_change.py.
Supports four plot types:
  1. group_waterfall : Steps from Pred A -> Groups -> Pred B
  2. feature_bar     : Horizontal bar chart of top N feature effects on change
  3. comparison      : Horizontal bar chart annotated with Scenario A -> B feature values
  4. spatial_drivers : Regional map of coordinates colored by the dominant driver
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default Output Directory
DEFAULT_PLOT_DIR = "/home/vajira/SRchange/output_SR/local_change/plots"

# Feature categories to match explain_change.py
STATIC_FEATURES = ['elev', 'slope', 'roughness', 'hand']
CLIMATE_FEATURES = [
    'annual_precip', 'aridity_index', 'frost_months', 'gdm_0', 'gdm_10', 'gdm_5', 
    'longest_dry_spell_months', 'max_monthly_precip', 'max_monthly_tmean', 
    'mean_annual_tmp', 'min_monthly_precip', 'min_monthly_tmean', 
    'monthly_precip_std', 'monthly_tmean_range', 'monthly_tmean_std', 
    'months_gt_25C', 'months_gt_30C', 'pet_annual', 'precip_coldest_quarter', 
    'precip_cv', 'precip_deficit_annual', 'precip_iqr', 'precip_p10', 'precip_p25', 
    'precip_p5', 'precip_p50', 'precip_p75', 'precip_p90', 'precip_p95', 
    'precip_warmest_quarter', 'temp_driest_quarter', 'temp_wettest_quarter', 
    'tmean_iqr', 'tmean_p10', 'tmean_p25', 'tmean_p5', 'tmean_p50', 'tmean_p75', 
    'tmean_p90', 'tmean_p95'
]
SCENARIO_LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + SCENARIO_LU_FEATURES


def find_nearest_pixel(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    """Finds the row in the dataframe closest to the target lat/lon."""
    dist = np.sqrt((df['input_lat'] - lat)**2 + (df['input_lon'] - lon)**2)
    idx = np.argmin(dist)
    row = df.iloc[idx]
    print(f"Target coordinate ({lat}, {lon}) matched closest to row coordinate ({row['input_lat']:.4f}, {row['input_lon']:.4f})")
    return row


def plot_group_waterfall(row: pd.Series, out_path: str):
    """
    Renders a custom Waterfall plot showing the transition from Pred A to Pred B
    via grouped contributions.
    """
    pred_A = np.log(row['pred_A'])
    pred_B = np.log(row['pred_B'])

    # Define steps
    steps = [
        ('Static Group', row['group_static_effect']),
        ('Climate Group', row['group_climate_effect']),
        ('Land Use Group', row['group_lu_effect']),
        ('Cluster Shift', row['cluster_shift_effect'])
    ]
    
    # Filter out steps with virtually 0 contribution to keep plot clean
    steps = [s for s in steps if abs(s[1]) > 1e-5]
    
    labels = ['Scenario A Pred'] + [s[0] for s in steps] + ['Scenario B Pred']
    
    # Calculate cumulative positions
    values = [pred_A]
    current = pred_A
    for _, val in steps:
        current += val
        values.append(current)
    values.append(pred_B)  # The last value is just pred_B again (for the final bar)
    
    # Plot construction
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bars = len(labels)
    bar_width = 0.6
    
    # Plot first and last (neutral anchors)
    ax.bar(0, pred_A, width=bar_width, color='#7f7f7f', edgecolor='black')
    ax.bar(n_bars - 1, pred_B, width=bar_width, color='#555555', edgecolor='black')
    
    # Plot intermediate transition steps
    for i in range(1, n_bars - 1):
        start_val = values[i-1]
        change = steps[i-1][1]
        color = '#d62728' if change > 0 else '#1f77b4'  # Red for positive, Blue for negative
        
        ax.bar(i, change, bottom=start_val, width=bar_width, color=color, edgecolor='black')
        
        # Add a text label above/below the transition bar
        text_y = start_val + change / 2.0
        ax.text(i, text_y, f"{change:+.3f}", ha='center', va='center', color='white', fontweight='bold')
        
        # Draw dashed connector lines
        ax.plot([i - 1 + bar_width/2, i - bar_width/2], [start_val, start_val], color='black', linestyle='--', linewidth=0.8)
        ax.plot([i - bar_width/2, i + bar_width/2], [start_val + change, start_val + change], color='black', linestyle='--', linewidth=0.8)
        
    # Draw final connector
    ax.plot([n_bars - 2 + bar_width/2, n_bars - 1 - bar_width/2], [pred_B, pred_B], color='black', linestyle='--', linewidth=0.8)
    
    # Add values on top of anchor bars
    ax.text(0, pred_A + 0.02 * pred_A, f"{pred_A:.3f}", ha='center', va='bottom', fontweight='bold')
    ax.text(n_bars - 1, pred_B + 0.02 * pred_B, f"{pred_B:.3f}", ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel("Predicted SR Value")
    ax.set_title(f"SR Prediction Change Decomposition Waterfall\nCoord: ({row['snapped_lat']:.4f}, {row['snapped_lon']:.4f})  |  Total Change: {pred_B - pred_A:+.4f}")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Group waterfall plot successfully saved to: {out_path}")


def plot_feature_bar(row: pd.Series, top_n: int, out_path: str):
    """Horizontal bar chart showing the individual contribution of top N features."""
    feature_effects = []
    for feat in ALL_FEATURES:
        effect_col = f"{feat}_effect"
        if effect_col in row:
            feature_effects.append((feat, row[effect_col]))
            
    # Sort by absolute impact
    feature_effects = sorted(feature_effects, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    feature_effects.reverse()  # Reverse to place largest at the top of the horizontal bar
    
    feats = [x[0] for x in feature_effects]
    effects = [x[1] for x in feature_effects]
    colors = ['#d62728' if e > 0 else '#1f77b4' for e in effects]
    
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1.5))
    bars = ax.barh(range(len(feats)), effects, color=colors, edgecolor='black', height=0.6)
    
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Marginal Feature Contribution to Change ($\Delta \phi_i$)")
    ax.set_title(f"Top {top_n} Individual Feature Contributions to SR Change\nCoord: ({row['snapped_lat']:.4f}, {row['snapped_lon']:.4f})  |  Total Change: {row['pred_B'] - row['pred_A']:+.4f}")
    
    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        align = 'left' if width < 0 else 'right'
        offset = -5 if width < 0 else 5
        ax.annotate(f"{width:+.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=align, va='center', fontsize=9, color='black', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Feature bar plot successfully saved to: {out_path}")


def plot_value_comparison(row: pd.Series, top_n: int, out_path: str):
    """
    Feature contribution bar chart that explicitly states the transition 
    of the feature's physical values from Scenario A to Scenario B.
    """
    feature_effects = []
    for feat in ALL_FEATURES:
        effect_col = f"{feat}_effect"
        if effect_col in row:
            val_A = row[f"{feat}_val_A"]
            val_B = row[f"{feat}_val_B"]
            feature_effects.append((feat, row[effect_col], val_A, val_B))
            
    # Sort by absolute impact
    feature_effects = sorted(feature_effects, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    feature_effects.reverse()
    
    labels = []
    for feat, _, v_A, v_B in feature_effects:
        # Create dynamic label showing value transition
        labels.append(f"{feat}\n({v_A:.2f} → {v_B:.2f})")
        
    effects = [x[1] for x in feature_effects]
    colors = ['#d62728' if e > 0 else '#1f77b4' for e in effects]
    
    fig, ax = plt.subplots(figsize=(11, top_n * 0.55 + 1.5))
    bars = ax.barh(range(len(labels)), effects, color=colors, edgecolor='black', height=0.6)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Marginal Feature Contribution to Change ($\Delta \phi_i$)")
    ax.set_title(f"Top {top_n} Feature Contributions with Values (Scenario A → Scenario B)\nCoord: ({row['snapped_lat']:.4f}, {row['snapped_lon']:.4f})")
    
    for bar in bars:
        width = bar.get_width()
        align = 'left' if width < 0 else 'right'
        offset = -5 if width < 0 else 5
        ax.annotate(f"{width:+.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=align, va='center', fontsize=9, color='black', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Dual-scenario feature value comparison plot successfully saved to: {out_path}")


def plot_spatial_drivers(df: pd.DataFrame, out_path: str):
    """
    Identifies the dominant driver of change at each spatial pixel in the CSV 
    and renders a regional map scatter.
    """
    if len(df) < 2:
        print("Warning: Spatial driver map requires multiple coordinates in the input CSV. Skipping.")
        return
        
    # Find dominant driver for each coordinate
    dominant_drivers = []
    colors = []
    
    for idx, row in df.iterrows():
        # Compare absolute effects
        eff_clim = abs(row['group_climate_effect'])
        eff_lu = abs(row['group_lu_effect'])
        eff_shift = abs(row['cluster_shift_effect'])
        
        mx = max(eff_clim, eff_lu, eff_shift)
        if mx == eff_clim:
            dominant_drivers.append("Climate")
            colors.append('#1f77b4')  # Blue
        elif mx == eff_lu:
            dominant_drivers.append("Land Use")
            colors.append('#ff7f0e')  # Orange
        else:
            dominant_drivers.append("Cluster Shift")
            colors.append('#9467bd')  # Purple

    temp_df = df.copy()
    temp_df['dominant'] = dominant_drivers
    temp_df['color'] = colors
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot by group to construct a clean legend
    for grp, grp_df in temp_df.groupby('dominant'):
        ax.scatter(grp_df['snapped_lon'], grp_df['snapped_lat'], 
                   c=grp_df['color'], label=grp, s=70, edgecolor='black', alpha=0.8)
        
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Symmetric Decomposition: Dominant Driver of SR Change")
    ax.legend(title="Dominant Driver")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Spatial dominant driver map successfully saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plotting suite for SR Change Decomposition output.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV produced by explain_change.py")
    parser.add_argument("--plot_type", type=str, default="group_waterfall",
                        choices=["group_waterfall", "feature_bar", "comparison", "spatial_drivers"],
                        help="Type of plot to generate")
    parser.add_argument("--lat", type=float, default=None, help="Target latitude for single-pixel plots")
    parser.add_argument("--lon", type=float, default=None, help="Target longitude for single-pixel plots")
    parser.add_argument("--top_n", type=int, default=10, help="Number of features to display in feature plots")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PLOT_DIR, help="Directory to save generated plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input decomposition CSV file not found: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # 2. Dispatch Plot
    if args.plot_type == "spatial_drivers":
        out_file = os.path.join(args.output_dir, "spatial_dominant_drivers.png")
        plot_spatial_drivers(df, out_file)
    else:
        # Resolve target pixel for single-pixel plot types
        if args.lat is not None and args.lon is not None:
            row = find_nearest_pixel(df, args.lat, args.lon)
        else:
            print("No coordinate specified via --lat and --lon. Defaulting to the first row in the CSV.")
            row = df.iloc[0]
            
        coord_tag = f"lat{row['snapped_lat']:.3f}_lon{row['snapped_lon']:.3f}"
        
        if args.plot_type == "group_waterfall":
            out_file = os.path.join(args.output_dir, f"waterfall_{coord_tag}.png")
            plot_group_waterfall(row, out_file)
            
        elif args.plot_type == "feature_bar":
            out_file = os.path.join(args.output_dir, f"feature_bar_{coord_tag}.png")
            plot_feature_bar(row, args.top_n, out_file)
            
        elif args.plot_type == "comparison":
            out_file = os.path.join(args.output_dir, f"comparison_bar_{coord_tag}.png")
            plot_value_comparison(row, args.top_n, out_file)


if __name__ == "__main__":
    main()