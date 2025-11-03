import itertools
import os
import zipfile

import requests

import config
from logger import setup_logger

logger = setup_logger(__name__)


def download_coco():
    os.makedirs(config.DATA_BASE_DIR, exist_ok=True)
    ZIP = os.path.join(config.DATA_BASE_DIR, "images.zip")

    if not os.path.exists(ZIP):
        logger.info(f"Downloading COCO2017 validation set from: {config.COCO_URL}")
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
                        logger.debug(next(spinner))
        logger.info("Download complete")
    else:
        logger.debug("Dataset already downloaded.")

    if not os.path.exists(config.IMAGES_DIR):
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(ZIP, "r") as zip_f:
            zip_f.extractall(config.DATA_BASE_DIR)
        logger.info("Extraction complete")
    else:
        logger.debug("Dataset already extracted.")
