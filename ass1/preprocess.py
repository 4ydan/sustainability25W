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
        req = requests.get(config.COCO_URL, stream=True)

        with open(ZIP, "wb") as f:
            for chunk in req.iter_content(chunk_size=config.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
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


def download_coco_captions():
    os.makedirs(config.DATA_BASE_DIR, exist_ok=True)
    ZIP = os.path.join(config.DATA_BASE_DIR, "captions.zip")

    if not os.path.exists(ZIP):
        logger.info(f"Downloading COCO2017 annotations set from: {config.COCO_ANNOTATIONS}")
        req = requests.get(config.COCO_ANNOTATIONS, stream=True)

        with open(ZIP, "wb") as f:
            for chunk in req.iter_content(chunk_size=config.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        logger.info("Download complete")
    else:
        logger.debug("Dataset already downloaded.")

   
    if not os.path.exists(config.ANNOTATIONS_DIR):
        target_filename = "annotations/captions_val2017.json" 
        logger.info("Extracting captions")
        with zipfile.ZipFile(ZIP, "r") as zip_f:
            zip_f.extract(target_filename, path=config.DATA_BASE_DIR)
        logger.info("Extraction complete")
    else:
        logger.debug("Dataset already extracted.")
