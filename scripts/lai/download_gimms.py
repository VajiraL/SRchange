import os
import requests
from concurrent.futures import ThreadPoolExecutor

files = {
    "GIMMS_LAI4g_AVHRR_solely_1982_1990.zip": "https://zenodo.org/records/8281930/files/GIMMS_LAI4g_AVHRR_solely_1982_1990.zip?download=1",
    "GIMMS_LAI4g_AVHRR_solely_1991_2000.zip": "https://zenodo.org/records/8281930/files/GIMMS_LAI4g_AVHRR_solely_1991_2000.zip?download=1",
    "GIMMS_LAI4g_AVHRR_solely_2011_2015.zip": "https://zenodo.org/records/8281930/files/GIMMS_LAI4g_AVHRR_solely_2011_2015.zip?download=1"
}

target_dir = "/home/vajira/SRchange/data-raw/GIMMS_LAI4g_AVHRR/"

def download_file(filename, url):
    filepath = os.path.join(target_dir, filename)
    print(f"Downloading {filename}...")
    
    # Check if already downloaded
    # Note: Zenodo download links might redirect to different URLs, so we handle it.
    resume_header = {}
    if os.path.exists(filepath):
        first_byte = os.path.getsize(filepath)
        resume_header = {'Range': f'bytes={first_byte}-'}
    else:
        first_byte = 0

    response = requests.get(url, headers=resume_header, stream=True, allow_redirects=True)
    
    mode = 'ab' if first_byte > 0 else 'wb'
    
    if response.status_code in [200, 206]:
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
        print(f"Finished downloading {filename}")
    elif response.status_code == 416:
        print(f"{filename} already fully downloaded.")
    else:
        print(f"Failed to download {filename}: {response.status_code}")

os.makedirs(target_dir, exist_ok=True)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(download_file, name, url) for name, url in files.items()]
    for future in futures:
        future.result()

print("All downloads finished. Unzipping...")
import subprocess
subprocess.run(f"cd {target_dir} && unzip -o '*.zip' && rm *.zip", shell=True)
print("Extraction complete.")
