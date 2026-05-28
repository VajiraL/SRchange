import os
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
from datetime import datetime
import multiprocessing

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
LAI_DIR = os.path.join(ROOT_DIR, "data-raw/CDR_LAI_FAPAR")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/LAI_processed")
REF_GRID_PATH = os.path.join(ROOT_DIR, "data/SR-target/refGrid.nc")

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def process_period(start_year, end_year):
    """
    Computes 95th percentile of LAI for each year, then averages them over the period.
    Saves the final result at 0.05 resolution.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"Starting processing for period {start_year}-{end_year}")
    
    # Load reference grid for final alignment
    ref_grid = xr.open_dataset(REF_GRID_PATH)
    
    yearly_95ths = []
    
    for year in range(start_year, end_year + 1):
        year_path = os.path.join(LAI_DIR, str(year))
        if not os.path.exists(year_path):
            log(f"Year {year} directory missing. Skipping.")
            continue
            
        files = sorted([os.path.join(year_path, f) for f in os.listdir(year_path) if f.endswith('.nc')])
        if not files:
            log(f"No files found for year {year}. Skipping.")
            continue
            
        log(f"Processing {len(files)} files for year {year}...")
        
        # Load and concat all daily files for the year
        try:
            # Open files one by one to find the corrupted one
            valid_files = []
            for f in files:
                try:
                    xr.open_dataset(f).close()
                    valid_files.append(f)
                except Exception as e:
                    log(f"  Corrupted file detected and skipped: {f}")

            if not valid_files:
                log(f"  No valid files found for year {year}. Skipping.")
                continue

            ds_year = xr.open_mfdataset(valid_files, combine='nested', concat_dim='time', 
                                       parallel=True, chunks={'latitude': 600, 'longitude': 600})
            
            lai_year = ds_year['LAI']
            
            # Mask out invalid values
            lai_year = lai_year.where((lai_year >= 0) & (lai_year <= 10))
            
            # Since we have 2.3TB RAM, we can afford large chunks.
            # 3600x7200 total size. 600x600 is a good compromise for Dask parallelism.
            # Percentile/Quantile requires full access along the time dimension.
            lai_year = lai_year.chunk({'time': -1})
            
            log(f"  Computing 95th percentile for {year}...")
            # Quantile over the time dimension for the whole year
            lai_95 = lai_year.quantile(0.95, dim='time').compute()
            
            yearly_95ths.append(lai_95.compute())
            log(f"  Finished year {year}.")
            
        except Exception as e:
            log(f"  Error processing year {year}: {e}")

    if not yearly_95ths:
        log("No yearly data processed. Exiting.")
        return

    # Average the yearly 95th percentiles
    log("Averaging yearly 95th percentiles...")
    final_lai = xr.concat(yearly_95ths, dim='year').mean(dim='year')
    
    # Standardize names and attributes
    final_lai = final_lai.rename('lai_95_avg')
    final_lai.attrs['long_name'] = f"Average 95th percentile LAI ({start_year}-{end_year})"
    final_lai.attrs['units'] = 'm2/m2'
    
    # Align to reference grid
    log("Aligning to reference grid (0.05 deg)...")
    final_lai = final_lai.rename({'latitude': 'lat', 'longitude': 'lon'})
    final_lai = final_lai.interp_like(ref_grid, method='nearest')
    
    output_filename = os.path.join(OUTPUT_DIR, f"lai_95th_{start_year}_{end_year}.nc")
    log(f"Saving to {output_filename}...")
    final_lai.to_netcdf(output_filename)
    log("Processing complete.")

if __name__ == "__main__":
    # Test on year 1983
    process_period(1983, 1983)