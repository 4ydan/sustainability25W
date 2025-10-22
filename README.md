# Sustainable AI - Image Captioning Assignment

This repository contains the implementation for Assignment 1: Efficient Vision Language - Image Captioning.

## Overview

The goal is to generate descriptive captions for images while measuring and reporting:
- **Efficiency Metrics**: Peak VRAM (MiB), latency per image, throughput (img/s), and model size on disk
- **Captioning Metrics**: CIDEr (primary), BLEU-4, and SPICE
- **Energy Metrics** (Bonus): Estimated using nvidia-smi logging

## Features

- Image captioning using pre-trained BLIP model (efficient vision-language model)
- Support for COCO 2017 validation dataset (5k images)
- Comprehensive efficiency monitoring (VRAM, latency, throughput, model size)
- Energy consumption tracking (bonus feature)
- Evaluation metrics: CIDEr, BLEU-4, SPICE
- Random 25-sample human spot-check reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/4ydan/sustainability25W.git
cd sustainability25W/ass1
```

2. Install dependencies using `uv`:
```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv sync
```

3. (Optional) For SPICE metric, you need Java:
```bash
# Ubuntu/Debian
sudo apt-get install default-jre

# macOS
brew install openjdk
```

## Dataset Setup

### Option 1: COCO 2017 Validation Dataset (Recommended)

1. Download COCO 2017 validation images (~1GB):
```bash
mkdir -p data
cd data
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

2. Download COCO annotations (~241MB):
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cd ..
```

### Option 2: Dummy Dataset (For Testing)

Run with `--use-dummy` flag to test the pipeline without downloading the full dataset.

## Usage

### Basic Usage

Run captioning on COCO validation set:
```bash
cd ass1
uv run python run_captioning.py
```

### With Dummy Dataset (Testing)

```bash
cd ass1
uv run python run_captioning.py --use-dummy --max-samples 50
```

### Custom Options

```bash
cd ass1
uv run python run_captioning.py \
    --data-dir ./data \
    --output-dir ./results \
    --model-name Salesforce/blip-image-captioning-base \
    --device cuda \
    --max-samples 5000
```

### Command-line Arguments

- `--data-dir`: Directory containing COCO dataset (default: `./data`)
- `--output-dir`: Output directory for results (default: `./results`)
- `--model-name`: HuggingFace model name (default: `Salesforce/blip-image-captioning-base`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--max-samples`: Maximum number of images to process (default: all 5000)
- `--use-dummy`: Use dummy dataset for testing
- `--batch-size`: Batch size (default: 1 for accurate latency measurement)

## Output

The script generates the following outputs in the `results/` directory:

1. **predictions.json**: All generated captions
2. **random_samples.json**: Random 25 samples for human spot-check
3. **metrics.json**: Complete evaluation and efficiency metrics

### Example Output

```
==============================================================
EFFICIENCY METRICS SUMMARY
==============================================================
Peak VRAM Usage:        1234.56 MiB
Model Size on Disk:     456.78 MB
Average Latency:        123.45 ms/image
Median Latency:         120.30 ms/image
Throughput:             8.10 images/sec

ENERGY METRICS (Bonus):
Average Power:          150.25 W
Total Energy:           0.0521 Wh
==============================================================

==============================================================
EVALUATION METRICS
==============================================================
CIDEr (Primary):        0.9876
BLEU-4:                 0.3456
BLEU-1:                 0.7234
BLEU-2:                 0.5678
BLEU-3:                 0.4321
SPICE:                  0.2109
==============================================================
```

## Model Information

**Default Model**: Salesforce/blip-image-captioning-base
- Efficient vision-language model
- Good balance between performance and efficiency
- ~500MB model size
- Suitable for sustainable AI applications

## Configuration

Edit `config.py` to modify:
- Model selection
- Batch size
- Maximum caption length
- Number of beams for generation
- Output directories
- Measurement options

## Project Structure

```
sustainability25W/
├── README.md                  # This file
└── ass1/                      # Assignment 1 implementation
    ├── pyproject.toml         # uv project configuration
    ├── run_captioning.py      # Main experiment script
    ├── config.py              # Configuration settings
    ├── captioning_model.py    # Model wrapper
    ├── dataset_loader.py      # Dataset loading utilities
    ├── efficiency_metrics.py  # Efficiency monitoring
    ├── evaluation_metrics.py  # Evaluation metrics (CIDEr, BLEU, SPICE)
    ├── test_basic.py          # Test suite
    ├── example_usage.py       # Usage examples
    ├── USAGE.md               # Detailed usage guide
    ├── ASSIGNMENT_SUMMARY.md  # Assignment summary
    └── results/               # Output directory
        ├── predictions.json
        ├── random_samples.json
        └── metrics.json
```

## Requirements

- Python 3.8+
- `uv` package manager
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended)
- ~2GB GPU VRAM for inference
- ~10GB disk space for COCO dataset

## Metrics Explanation

### Efficiency Metrics
- **Peak VRAM**: Maximum GPU memory used during inference
- **Latency**: Average time to generate caption per image
- **Throughput**: Images processed per second
- **Model Size**: Size of model parameters on disk
- **Energy** (Bonus): Power consumption and total energy used

### Captioning Metrics
- **CIDEr**: Consensus-based Image Description Evaluation (primary metric)
- **BLEU-4**: 4-gram BLEU score
- **SPICE**: Semantic Propositional Image Caption Evaluation

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 1`
- Use CPU: `--device cpu`
- Process fewer images: `--max-samples 1000`

### Missing SPICE Metric
- Install Java runtime
- SPICE downloads required files on first run

### Dataset Not Found
- Use dummy dataset: `--use-dummy`
- Or download COCO dataset following instructions above

## License

This project is for educational purposes as part of the Sustainable AI course.

## References

- BLIP: [Salesforce/BLIP](https://github.com/salesforce/BLIP)
- COCO Dataset: [cocodataset.org](https://cocodataset.org/)
- pycocoevalcap: [COCO Caption Evaluation](https://github.com/tylin/coco-caption)