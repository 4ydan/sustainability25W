#!/bin/bash

# This script downloads and decompresses the LamaH-CE daily dataset.

echo "Downloading LamaH-CE_daily.tar.gz..."
wget -O ass2/data/LamaH-CE_daily.tar.gz "https://zenodo.org/record/5153305/files/2_LamaH-CE_daily.tar.gz?download=1"

if [ $? -eq 0 ]; then
    echo "Download complete. Decompressing the dataset..."
    tar -xzvf ass2/data/LamaH-CE_daily.tar.gz -C ass2/data/
    if [ $? -eq 0 ]; then
        echo "Decompression complete. The dataset is now available in ass2/data/"
    else
        echo "Error: Decompression failed."
    fi
else
    echo "Error: Download failed."
fi