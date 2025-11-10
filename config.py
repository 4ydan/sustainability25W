import os

# Data configuration
DATA_BASE_DIR = "data/coco2017"
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_DIR = os.path.join(DATA_BASE_DIR, "val2017")
ANNOTATIONS_DIR = os.path.join(DATA_BASE_DIR, "annotations")
OUTPUT_DIR = os.path.join(DATA_BASE_DIR, "captions_val2017")

# Model configuration
# Options: "Qwen/Qwen2-VL-2B-Instruct", "HuggingFaceTB/SmolVLM-Instruct"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-Instruct"
PROMPT = "Describe this image in a short caption"
MAX_NEW_TOKENS = 128

# Download configuration
CHUNK_SIZE = 8192  # Download chunk size in bytes
