import itertools
import os
import zipfile

import requests

import config


def download_coco():
    os.makedirs(config.DATA_BASE_DIR, exist_ok=True)
    ZIP = os.path.join(config.DATA_BASE_DIR, "images.zip")

    if not os.path.exists(ZIP):
        print(f"Downloading COCO2017 validation set from: \n {config.COCO_URL}")
        total = 0
        req = requests.get(config.COCO_URL, stream=True)

        with open(ZIP, "wb") as f:
            spinner = itertools.cycle(
                ["Downloading |", "Downloading /", "Downloading —", "Downloading \\"]
            )
            for i, chunk in enumerate(req.iter_content(chunk_size=config.CHUNK_SIZE)):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    if i % 50 == 0:
                        print(next(spinner), end="\r", flush=True)
        print("Download done")
    else:
        print("Dataset already downloaded.")

    if not os.path.exists(config.IMAGES_DIR):
        print("Extracting dataset ...")
        with zipfile.ZipFile(ZIP, "r") as zip_f:
            zip_f.extractall(config.DATA_BASE_DIR)
        print("Extraction done")
    else:
        print("Dataset already extracted.")
