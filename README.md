# sustainability25W

Image captioning pipeline using Qwen2-VL-2B-Instruct on COCO2017 validation set.

## Setup

Install dependencies:
```bash
uv sync
```

## Usage

Download COCO2017 validation dataset:
```bash
uv run main.py
```

Generate captions:
```bash
uv run basemodel.py
```

## Requirements

- Python >=3.10
- CUDA-capable GPU (optional, falls back to CPU)

## Configuration

Edit `basemodel.py` to adjust:
- `MODEL_NAME`: Vision-language model to use
- `PROMPT`: Captioning prompt
- `MAX_NEW_TOKENS`: Maximum caption length
- Line 38: Number of images to process (default: 10 for testing)

## Output

Captions saved to `data/coco2017/captions_val2017/` as `.txt` files.
Currently disabled (line 92-94 in `basemodel.py`).
