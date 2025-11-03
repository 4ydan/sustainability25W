# sustainability25W

Image captioning pipeline using SmolVLM on COCO2017 validation set with configurable quantization.

TUWEL document: https://tuwel.tuwien.ac.at/pluginfile.php/4677714/mod_resource/content/1/Lecture%202.pdf

## Setup

Install dependencies:
```bash
uv sync
```

## Usage

Run with all options (full quantization, 100 images, CUDA device, save captions):
```bash
DEBUG=1 uv run main.py --quantization full --num-images 100 --device cuda --save-captions
```

### CLI Options

- `--quantization`, `-q`: Quantization mode
  - `none` (default): Unquantized baseline
  - `skip_vision_tower`: Quantize everything except vision tower
  - `full`: Quantize everything including vision tower
- `--num-images`, `-n`: Number of images to process (default: 1, use 0 for all)
- `--device`, `-d`: Device to use - `auto` (default), `cuda`, or `cpu`
- `--save-captions`: Save captions to disk (flag)

### Examples

Baseline unquantized model on 10 images:
```bash
uv run main.py -q none -n 10
```

Quantize without vision tower, process all images:
```bash
uv run main.py -q skip_vision_tower -n 0 --save-captions
```

Quick test with 1 image:
```bash
uv run main.py
```

## Requirements

- Python >=3.11
- CUDA-capable GPU (quantization requires CUDA)

## Project Structure

```
sustainability25W/
├── main.py           # CLI entry point with click interface
├── config.py         # Configuration constants
├── model_utils.py    # Model loading (all quantization modes)
├── inference.py      # Common inference logic
└── preprocess.py     # COCO dataset download
```

## Configuration

Edit `config.py` to adjust:
- `SMOLVLM_MODEL`: Model name
- `PROMPT`: Captioning prompt
- `MAX_NEW_TOKENS`: Maximum caption length

## Output

Captions saved to `data/coco2017/captions_val2017/` as `.txt` files (when `--save-captions` is used).
