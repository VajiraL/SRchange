#!/bin/bash
TARGET_DIR="/home/vajira/SRchange/data-raw/GIMMS_LAI4g_AVHRR/"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

urls=(
    "https://zenodo.org/records/8281930/files/GIMMS_LAI4g_AVHRR_solely_1982_1990.zip?download=1"
    "https://zenodo.org/records/8281930/files/GIMMS_LAI4g_AVHRR_solely_1991_2000.zip?download=1"
    "https://zenodo.org/records/8281930/files/GIMMS_LAI4g_AVHRR_solely_2011_2015.zip?download=1"
)

filenames=(
    "GIMMS_LAI4g_AVHRR_solely_1982_1990.zip"
    "GIMMS_LAI4g_AVHRR_solely_1991_2000.zip"
    "GIMMS_LAI4g_AVHRR_solely_2011_2015.zip"
)

for i in "${!urls[@]}"; do
    wget -c -O "${filenames[$i]}" "${urls[$i]}"
done

echo "Downloads finished. Unzipping..."
for f in *.zip; do
    unzip -o "$f"
done
rm *.zip
echo "All done."
