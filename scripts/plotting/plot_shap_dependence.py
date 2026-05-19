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
PLOT_DIR = os.path.join(OUTPUT_BASE, 'plots/dependence')

os.makedirs(PLOT_DIR, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Plotting Logic
# ---------------------------
def plot_dependence_for_model(model_name):
    data_path = os.path.join(DATA_DIR, f"shap_{model_name}.npz")
    if not os.path.exists(data_path):
        log(f"Data for {model_name} not found, skipping.")
        return

    log(f"Loading SHAP data for {model_name}...")
    with np.load(data_path, allow_pickle=True) as data:
        shap_values = data['shap_values']
        X = data['X']
        feature_names = data['feature_names'].tolist()

    # Create a DataFrame for plotting convenience
    df_X = pd.DataFrame(X, columns=feature_names)

    # Determine top 5 features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    top_features = [feature_names[i] for i in top_indices]

    log(f"Top 5 features for {model_name}: {top_features}")

    # Create sub-directory for this model
    model_plot_dir = os.path.join(PLOT_DIR, model_name)
    os.makedirs(model_plot_dir, exist_ok=True)

    for feature in top_features:
        plt.figure(figsize=(10, 6))
        # shap.dependence_plot automatically finds the best interaction feature if interaction_index='auto'
        shap.dependence_plot(feature, shap_values, df_X, show=False)
        plt.title(f"Dependence Plot: {feature} ({model_name})")
        plt.tight_layout()
        
        save_path = os.path.join(model_plot_dir, f"dependence_{feature}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        log(f"Saved dependence plot for {feature} in {model_name}")

# ---------------------------
# 3. Main
# ---------------------------
def main():
    # Identify models from saved data files
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith("shap_") and f.endswith(".npz")]
    model_names = [f[5:-4] for f in data_files] # Strip "shap_" and ".npz"
    
    log(f"Found {len(model_names)} models to process: {model_names}")

    for name in model_names:
        plot_dependence_for_model(name)

    log("Dependence plotting complete.")

if __name__ == "__main__":
    main()
