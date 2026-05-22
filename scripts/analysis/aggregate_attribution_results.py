# aggreagte attribution of individual transitions
import pandas as pd
from pathlib import Path

def aggregate_results():
    output_dir = Path('/home/vajira/SRchange/output_SR/sr_change_attribution/')
    files = list(output_dir.glob('sr_change_uncertainty_*.csv'))
    
    all_data = []
    for f in files:
        df = pd.read_csv(f)
        all_data.append(df)
    
    if not all_data:
        print("No files found.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    
    # Sort transitions in a logical order
    # (Simplified sorting for now)
    combined.to_csv(output_dir / 'sr_change_uncertainty_all.csv', index=False)
    print(f"Aggregated {len(files)} files into sr_change_uncertainty_all.csv")

    # Display summary
    summary = combined.pivot(index='transition', columns='component', values='mean_change')
    # Filter for interesting transitions or sort them
    print("\nSummary of Mean Changes (mm):")
    print(summary)

if __name__ == "__main__":
    aggregate_results()