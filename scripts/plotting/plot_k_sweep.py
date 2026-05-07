import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT = "/home/vajira/SRchange"
CSV_PATH = f"{ROOT}/output_SR/k_sweep/k_sweep_metrics.csv"
OUT_PLOT = f"{ROOT}/output_SR/k_sweep/k_sweep_plot.png"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
        
    df = pd.read_csv(CSV_PATH).sort_values('k')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # R2 Plot
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('R²', color=color)
    ax1.plot(df['k'], df['r2'], marker='o', color=color, linewidth=2, label='R²')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # MAE Plot
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MAE (mm)', color=color)
    ax2.plot(df['k'], df['mae'], marker='s', color=color, linewidth=2, label='MAE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight k=6
    if 6 in df['k'].values:
        row6 = df[df['k'] == 6]
        ax1.annotate(f'Main Analysis (k=6)\nR²={row6.r2.values[0]:.3f}', 
                    xy=(6, row6.r2.values[0]), xytext=(8, row6.r2.values[0]-0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.title('S_R Prediction Accuracy vs. Number of Clusters (k)')
    fig.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    print(f"Plot saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
