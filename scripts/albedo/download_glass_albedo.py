import os
import requests
from bs4 import BeautifulSoup
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
BASE_URL = "https://glass.hku.hk/archive/Albedo/MIX/0.05D/"
TARGET_DIR = "/home/vajira/SRchange/data-raw/albedo"
YEARS = list(range(1981, 1991)) + list(range(2011, 2021))

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def download_file(url, year_dir):
    filename = url.split('/')[-1]
    filepath = os.path.join(year_dir, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return
    
    # Use wget for robust downloads
    cmd = ["wget", "-q", "-c", "-O", filepath, url]
    subprocess.run(cmd)

def process_year(year):
    log(f"Fetching file list for year {year}...")
    year_url = f"{BASE_URL}{year}/"
    year_dir = os.path.join(TARGET_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    try:
        # Note: glass.hku.hk might need verify=False if there are SSL issues, 
        # but let's try standard first.
        response = requests.get(year_url, timeout=30)
        if response.status_code != 200:
            log(f"  Failed to access {year_url} (Status: {response.status_code})")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all .hdf links (exclude .xml, .jpg, etc.)
        links = [a['href'] for a in soup.find_all('a', href=True) 
                 if a['href'].endswith('.hdf')]
        
        if not links:
            log(f"  No HDF files found for year {year}.")
            return
            
        log(f"  Found {len(links)} HDF files. Starting download...")
        
        # Construct full URLs (handle relative links)
        full_urls = []
        for link in links:
            if link.startswith('http'):
                full_urls.append(link)
            else:
                full_urls.append(year_url + link)
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda url: download_file(url, year_dir), full_urls)
            
        log(f"  Finished year {year}.")
            
    except Exception as e:
        log(f"  Error processing year {year}: {e}")

if __name__ == "__main__":
    for year in YEARS:
        process_year(year)
