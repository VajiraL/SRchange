import os
import requests
from bs4 import BeautifulSoup
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pyhdf.SD import SD, SDC

# ---------------------------
# Config
# ---------------------------
BASE_URL = "https://glass.hku.hk/archive/Albedo/MIX/0.05D/"
TARGET_DIR = "/home/vajira/SRchange/data-raw/albedo"
YEARS = list(range(1981, 1991)) + list(range(2011, 2021))

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def verify_hdf(filepath):
    try:
        hdf = SD(filepath, SDC.READ)
        hdf.end()
        return True
    except:
        return False

def download_and_verify(url, year_dir):
    filename = url.split('/')[-1]
    filepath = os.path.join(year_dir, filename)
    
    # Try up to 3 times
    for attempt in range(3):
        if os.path.exists(filepath):
            if verify_hdf(filepath):
                return # Already exists and valid
            else:
                log(f"  Verification failed for {filename}. Re-downloading (Attempt {attempt+1})...")
                os.remove(filepath)
        
        # Robust wget: no-clobber and no-parent are less relevant here than just clean overwrite for retry
        # -t 5: 5 retries, -w 2: 2 sec wait
        cmd = ["wget", "-q", "-t", "5", "-w", "2", "-O", filepath, url]
        subprocess.run(cmd)
        
        if verify_hdf(filepath):
            return
            
    log(f"  FAILED to download/verify {filename} after 3 attempts.")

def process_year(year):
    log(f"Fetching file list for year {year}...")
    year_url = f"{BASE_URL}{year}/"
    year_dir = os.path.join(TARGET_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    try:
        response = requests.get(year_url, timeout=30)
        if response.status_code != 200:
            log(f"  Failed to access {year_url}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.hdf')]
        
        if not links:
            log(f"  No HDF files found for year {year}.")
            return
            
        full_urls = [year_url + l if not l.startswith('http') else l for l in links]
        
        # Using fewer workers for better stability during HDF verification IO
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(lambda url: download_and_verify(url, year_dir), full_urls)
            
        log(f"  Finished year {year}.")
            
    except Exception as e:
        log(f"  Error processing year {year}: {e}")

if __name__ == "__main__":
    for year in YEARS:
        process_year(year)
