import itertools
import os
import zipfile

import requests


def download_coco():
    base_dir = "data/coco2017"
    os.makedirs(base_dir, exist_ok=True)
    URL = "http://images.cocodataset.org/zips/val2017.zip"
    ZIP = os.path.join(base_dir, "images.zip")

    if not os.path.exists(ZIP):
        print(f"Downloading COCO2017 validation set from: \n {URL}")
        total = 0
        req = requests.get(URL, stream=True)

        with open(ZIP, "wb") as f:
            spinner = itertools.cycle(
                ["Downloading |", "Downloading /", "Downloading —", "Downloading \\"]
            )
            for i, chunk in enumerate(req.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    if i % 50 == 0:
                        print(next(spinner), end="\r", flush=True)
        print("Download done")
    else:
        print("Dataset already present.")
    print("Extracting dataset ...")
    with zipfile.ZipFile(ZIP, "r") as zip_f:
        zip_f.extractall(base_dir)
    print("Extraction done")
