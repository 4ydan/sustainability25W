import os

# Data configuration
DATA_BASE_DIR = "data/coco2017"
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMAGES_DIR = os.path.join(DATA_BASE_DIR, "val2017")
OUTPUT_DIR = os.path.join(DATA_BASE_DIR, "captions_val2017")

# Model configuration
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = "Describe this image in a caption"
MAX_NEW_TOKENS = 128

# Processing configuration
# Set to None to process all images, or an integer for testing subset
NUM_IMAGES = 10

# Device configuration
# Options: "cuda", "cpu", "auto"
# "auto" will use CUDA if available and supported
DEVICE = "cpu"  # GTX 1060 not supported by current PyTorch build (requires sm_70+)

# Output configuration
SAVE_CAPTIONS = False  # Enable to save captions to disk
CHUNK_SIZE = 8192  # Download chunk size in bytes
