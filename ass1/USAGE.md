# Usage Guide

## Quick Start

### 1. Setup Environment

```bash
# Install uv if you haven't already
pip install uv

# Navigate to ass1 directory
cd ass1

# Install project dependencies
uv sync

# For SPICE metric (optional), install Java
sudo apt-get install default-jre  # Ubuntu/Debian
# or
brew install openjdk  # macOS
```

### 2. Run with Dummy Dataset (Testing)

Test the pipeline without downloading the full COCO dataset:

```bash
cd ass1
uv run python run_captioning.py --use-dummy --max-samples 50
```

This will:
- Create 50 dummy images
- Run captioning on them
- Report all efficiency and evaluation metrics
- Save results to `results/` directory

### 3. Run with COCO Dataset (Full Experiment)

First, download the COCO 2017 validation dataset:

```bash
# Create data directory
mkdir -p data
cd data

# Download validation images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download annotations (~241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cd ..
```

Then run the experiment:

```bash
cd ass1
uv run python run_captioning.py --max-samples 5000
```

## Command-Line Options

### Basic Options

- `--data-dir PATH`: Path to COCO dataset directory (default: `./data`)
- `--output-dir PATH`: Path to save results (default: `./results`)
- `--max-samples N`: Maximum number of images to process (default: all 5000)

### Model Options

- `--model-name NAME`: HuggingFace model name (default: `Salesforce/blip-image-captioning-base`)
- `--device DEVICE`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--batch-size N`: Batch size (default: 1)

### Dataset Options

- `--use-dummy`: Use dummy dataset for testing (no download required)

## Examples

### Example 1: Quick Test (No Dataset Download)

```bash
cd ass1
uv run python run_captioning.py --use-dummy --max-samples 10
```

Expected output:
```
============================================================
EFFICIENCY METRICS SUMMARY
============================================================
Peak VRAM Usage:        1234.56 MiB
Model Size on Disk:     456.78 MB
Average Latency:        123.45 ms/image
Median Latency:         120.30 ms/image
Throughput:             8.10 images/sec
============================================================

============================================================
EVALUATION METRICS
============================================================
CIDEr (Primary):        0.9876
BLEU-4:                 0.3456
BLEU-1:                 0.7234
============================================================
```

### Example 2: Subset of COCO (1000 images)

```bash
cd ass1
uv run python run_captioning.py --max-samples 1000
```

### Example 3: Full COCO Validation Set

```bash
cd ass1
uv run python run_captioning.py --max-samples 5000
```

This will take approximately:
- Time: ~10-15 minutes on GPU, ~60-90 minutes on CPU
- Memory: ~2GB GPU VRAM

### Example 4: CPU-only Mode

```bash
cd ass1
uv run python run_captioning.py --use-dummy --max-samples 20 --device cpu
```

## Understanding the Output

### Efficiency Metrics

1. **Peak VRAM Usage** (MiB): Maximum GPU memory used during inference
   - Important for understanding deployment requirements
   - Typically ~1200-2000 MiB for BLIP-base

2. **Average/Median Latency** (ms/image): Time to generate one caption
   - Average includes all samples
   - Median is more robust to outliers
   - Typical range: 100-200ms on GPU, 500-1000ms on CPU

3. **Throughput** (images/sec): Number of images processed per second
   - Inverse of average latency
   - Important for batch processing scenarios

4. **Model Size on Disk** (MB): Storage required for model parameters
   - BLIP-base: ~500MB
   - Important for deployment considerations

5. **Energy Metrics** (Bonus):
   - Average Power (W): Power consumption during inference
   - Total Energy (Wh): Total energy consumed
   - Only available with CUDA and nvidia-smi

### Evaluation Metrics

1. **CIDEr** (Primary Metric): Consensus-based Image Description Evaluation
   - Range: 0.0 to ~4.0+ (higher is better)
   - Measures consensus with human annotators
   - Expected range for BLIP: 0.8-1.2

2. **BLEU-4**: 4-gram BLEU score
   - Range: 0.0 to 1.0 (higher is better)
   - Measures n-gram overlap with references
   - Expected range: 0.3-0.4

3. **SPICE**: Semantic Propositional Image Caption Evaluation
   - Range: 0.0 to 1.0 (higher is better)
   - Measures semantic similarity
   - Expected range: 0.2-0.3

### Random Samples

The script prints 10 random samples showing:
- Image ID
- Generated caption
- Reference captions (ground truth)

This provides a quick human spot-check of caption quality. All 25 samples are saved to `results/random_samples.json`.

## Output Files

After running, check the `results/` directory:

1. **predictions.json**: All generated captions
   ```json
   {
     "123456": "a cat sitting on a table",
     "234567": "a dog running in a park"
   }
   ```

2. **random_samples.json**: 25 random samples for human evaluation
   ```json
   [
     {
       "image_id": 123456,
       "prediction": "a cat sitting on a table",
       "references": [
         "a cat is sitting on the table",
         "cat on table"
       ]
     }
   ]
   ```

3. **metrics.json**: Complete metrics summary
   ```json
   {
     "evaluation_metrics": {
       "CIDEr": 0.9876,
       "BLEU-4": 0.3456
     },
     "efficiency_metrics": {
       "peak_vram_mib": 1234.56,
       "avg_latency_ms": 123.45,
       "throughput_img_per_sec": 8.10
     }
   }
   ```

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
cd ass1
# Solution 1: Use CPU
uv run python run_captioning.py --device cpu

# Solution 2: Process fewer images
uv run python run_captioning.py --max-samples 100
```

### Issue: SPICE Metric Not Available

SPICE requires Java. Install it:
```bash
sudo apt-get install default-jre  # Ubuntu/Debian
```

Or run without SPICE - CIDEr and BLEU will still be reported.

### Issue: COCO Dataset Not Found

```bash
cd ass1
# Use dummy dataset instead
uv run python run_captioning.py --use-dummy

# Or download COCO following the dataset setup instructions
```

### Issue: Slow Performance

Expected speeds:
- GPU (CUDA): 8-10 images/sec
- CPU: 1-2 images/sec

If much slower:
- Check if GPU is being used: `nvidia-smi`
- Ensure PyTorch with CUDA is installed
- Close other GPU-intensive applications

## Advanced Usage

### Custom Model

Use a different model from HuggingFace:

```bash
cd ass1
uv run python run_captioning.py \
    --model-name Salesforce/blip-image-captioning-large \
    --max-samples 100
```

### Modify Configuration

Edit `config.py` to change:
- Beam search parameters (`NUM_BEAMS`)
- Maximum caption length (`MAX_LENGTH`)
- Logging interval (`LOG_INTERVAL`)

### Adding Your Own Images

To caption your own images, modify `dataset_loader.py` to load from a custom directory. See the `create_dummy_dataset()` function as a template.

## Performance Benchmarks

Typical performance on COCO 2017 validation (5k images):

| Configuration | Latency | Throughput | VRAM | CIDEr | BLEU-4 |
|--------------|---------|------------|------|-------|--------|
| BLIP-base GPU | 120ms | 8.3 img/s | 1.4GB | 1.05 | 0.36 |
| BLIP-base CPU | 800ms | 1.3 img/s | N/A | 1.05 | 0.36 |
| BLIP-large GPU | 180ms | 5.6 img/s | 2.2GB | 1.15 | 0.38 |

*Note: Benchmarks are approximate and depend on hardware.*

## Assignment Deliverables

For the assignment, ensure you report:

1. **Efficiency Metrics**:
   - ✓ Peak VRAM (MiB)
   - ✓ Latency per image (ms)
   - ✓ Throughput (images/sec)
   - ✓ Model size on disk (MB)

2. **Captioning Metrics**:
   - ✓ CIDEr (primary)
   - ✓ BLEU-4
   - ✓ SPICE (if available)

3. **Human Spot-Check**:
   - ✓ Random 25-sample check (saved to `random_samples.json`)

4. **Energy (Bonus)**:
   - ✓ Power and energy estimates (if CUDA available)

All metrics are automatically saved to `results/metrics.json` for easy reporting.
