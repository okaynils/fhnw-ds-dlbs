#!/bin/bash

set -e

FOLDER_ID="1nVg_DMUSTxuMEIVfgCvFBZSSVSZ4WcdU"
OUTPUT_DIR="./models"

if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Install it with 'pip install gdown'."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" --output "$OUTPUT_DIR"

find "$OUTPUT_DIR" -type f -name "*.zip" -exec unzip -o {} -d "$OUTPUT_DIR" \;
find "$OUTPUT_DIR" -type f -name "*.zip" -delete

echo "Download and extraction complete. Files are in $OUTPUT_DIR."
