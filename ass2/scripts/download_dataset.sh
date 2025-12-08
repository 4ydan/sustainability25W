#!/bin/bash

# This script downloads and decompresses the LamaH-CE daily dataset.

# Get the directory of the script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR="$SCRIPT_DIR/../data"
DATA_FILE="$DATA_DIR/LamaH-CE_daily.tar.gz"

# Ensure the data directory exists
mkdir -p "$DATA_DIR"

echo "Downloading LamaH-CE_daily.tar.gz to $DATA_DIR..."
wget -O "$DATA_FILE" "https://zenodo.org/record/5153305/files/2_LamaH-CE_daily.tar.gz?download=1"

if [ $? -eq 0 ]; then
    echo "Download complete. Decompressing the dataset..."
    tar -xzvf "$DATA_FILE" -C "$DATA_DIR/"
    if [ $? -eq 0 ]; then
        echo "Decompression complete. The dataset is now available in $DATA_DIR/"
        rm "$DATA_FILE" # Remove the tar.gz file after successful extraction
        echo "Removed the compressed file."
    else
        echo "Error: Decompression failed."
    fi
else
    echo "Error: Download failed."
fi
