import os
import requests
from bs4 import BeautifulSoup
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Source for filenames (fast for listing)
LISTING_BASE_URL = "https://www.ncei.noaa.gov/data/land-leaf-area-index-and-fapar/access/"
# Source for data (fast for downloading)
S3_BASE_URL = "https://noaa-cdr-leaf-area-index-fapar-pds.s3.amazonaws.com/data/"
TARGET_DIR = "/home/vajira/SRchange/data-raw/CDR_LAI_FAPAR"

def download_file(s3_url, year_dir):
    filename = s3_url.split('/')[-1]
    filepath = os.path.join(year_dir, filename)
    if os.path.exists(filepath):
        # Optional: check if file is complete (not zero size)
        if os.path.getsize(filepath) > 0:
            return
    
    # Using wget with multiple connections and backgrounding
    # S3 is very fast, we can use higher concurrency
    cmd = ["wget", "-q", "-c", "-O", filepath, s3_url]
    subprocess.run(cmd)

def process_year(year):
    print(f"[{year}] Fetching file list from NCEI...")
    listing_url = f"{LISTING_BASE_URL}{year}/"
    year_dir = os.path.join(TARGET_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    try:
        response = requests.get(listing_url)
        if response.status_code != 200:
            print(f"[{year}] Failed to access {listing_url}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract .nc filenames
        filenames = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]
        
        print(f"[{year}] Found {len(filenames)} files. Starting download from S3...")
        
        # Construct S3 URLs
        s3_links = [f"{S3_BASE_URL}{year}/{fn}" for fn in filenames]
        
        # Increased workers for S3
        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(lambda url: download_file(url, year_dir), s3_links)
        
        print(f"[{year}] Finished year.")
            
    except Exception as e:
        print(f"[{year}] Error: {e}")

if __name__ == "__main__":
    # Process years 1981 to 2020
    # We can also parallelize across years, but 20 threads per year is already a lot.
    for year in range(1981, 2021):
        process_year(year)
