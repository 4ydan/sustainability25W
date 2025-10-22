"""Configuration file for image captioning experiments."""

import os

# Model configuration
MODEL_NAME = "Salesforce/blip-image-captioning-base"  # Using BLIP as it's efficient and effective
DEVICE = "cuda"  # or "cpu"

# Dataset configuration
DATASET_NAME = "coco"
DATASET_SPLIT = "validation"
DATASET_SIZE = 5000  # COCO 2017 val has 5k images
DATA_DIR = "./data"

# Experiment configuration
BATCH_SIZE = 1  # For accurate per-image latency measurement
NUM_WORKERS = 4
MAX_LENGTH = 50  # Maximum caption length
NUM_BEAMS = 5  # For beam search during generation

# Output configuration
OUTPUT_DIR = "./results"
SAVE_PREDICTIONS = True
SAVE_METRICS = True

# Efficiency measurement
MEASURE_VRAM = True
MEASURE_ENERGY = True  # Bonus: requires nvidia-smi
LOG_INTERVAL = 100  # Log every N images

# Random seed for reproducibility
RANDOM_SEED = 42
