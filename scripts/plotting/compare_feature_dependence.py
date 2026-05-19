import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/explanation')
DATA_DIR = os.path.join(OUTPUT_BASE, 'data')
PLOT_DIR = os.path.join(OUTPUT_BASE, 'plots/comparison')

os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_TO_COMPARE = 'tmean_p25'

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Main
# ---------------------------
def main():
    # Identify models from saved data files
    data_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("shap_") and f.endswith(".npz")])
    model_names = [f[5:-4] for f in data_files]
    
    log(f"Comparing {FEATURE_TO_COMPARE} across {len(model_names)} models.")

    # We'll create a multi-panel plot
    n_models = len(model_names)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        data_path = os.path.join(DATA_DIR, f"shap_{name}.npz")
        with np.load(data_path, allow_pickle=True) as data:
            shap_values = data['shap_values']
            X = data['X']
            feature_names = data['feature_names'].tolist()
        
        df_X = pd.DataFrame(X, columns=feature_names)
        
        if FEATURE_TO_COMPARE not in feature_names:
            log(f"Feature {FEATURE_TO_COMPARE} not in model {name}, skipping panel.")
            axes[i].set_title(f"{name}: Feature Not Found")
            continue

        feat_idx = feature_names.index(FEATURE_TO_COMPARE)
        
        # Plot on specific axis
        # shap.dependence_plot doesn't natively take an 'ax' argument for the scatter part easily 
        # while keeping its color logic without some effort, but we can use shap.dependence_plot 
        # and specify show=False, then manage the figure.
        # Actually, for multi-panel, it's easier to manually plot or use the 'ax' parameter if supported.
        # SHAP dependence_plot supports 'ax' in newer versions.
        
        shap.dependence_plot(FEATURE_TO_COMPARE, shap_values, df_X, ax=axes[i], show=False)
        axes[i].set_title(f"Model: {name}")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Cross-Model Comparison of {FEATURE_TO_COMPARE} Dependence", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(PLOT_DIR, f"comparison_{FEATURE_TO_COMPARE}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    main()
