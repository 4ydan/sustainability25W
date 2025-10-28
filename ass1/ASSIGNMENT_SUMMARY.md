# Assignment 1: Efficient Vision Language - Image Captioning

## Summary

This repository contains a complete implementation for Assignment 1 focused on efficient image captioning with comprehensive metrics tracking.

## Implementation Overview

### Core Components

1. **Image Captioning Model** (`captioning_model.py`)
   - Uses BLIP (Bootstrapping Language-Image Pre-training) model
   - Efficient vision-language model with ~500MB size
   - Supports both CUDA and CPU inference
   - Configurable beam search and caption length

2. **Dataset Loader** (`dataset_loader.py`)
   - COCO 2017 validation dataset support (5k images)
   - Automatic annotation loading
   - Dummy dataset option for testing without full download
   - Efficient data loading with PyTorch DataLoader

3. **Efficiency Metrics** (`efficiency_metrics.py`)
   - Peak VRAM monitoring (MiB)
   - Per-image latency tracking (milliseconds)
   - Throughput calculation (images/sec)
   - Model size measurement (MB on disk)
   - Energy consumption tracking (bonus feature)

4. **Evaluation Metrics** (`evaluation_metrics.py`)
   - CIDEr (primary metric): Consensus-based evaluation
   - BLEU-4: N-gram overlap metric
   - SPICE: Semantic similarity metric
   - Automated evaluation pipeline

5. **Main Experiment Script** (`run_captioning.py`)
   - End-to-end pipeline
   - Command-line interface
   - Automatic result saving
   - Random 25-sample human spot-check

## Key Features

✅ **All Required Metrics Implemented:**
- Peak VRAM usage (MiB)
- Latency per image-prompt (ms)
- Throughput (images/sec)
- Model size on disk (MB)
- CIDEr (primary evaluation metric)
- BLEU-4 score
- SPICE score (when available)

✅ **Bonus Features:**
- Energy consumption estimation using nvidia-smi
- Power usage tracking (Watts)
- Total energy calculation (Watt-hours)

✅ **Quality Assurance:**
- Random 25-sample human spot-check
- Comprehensive test suite
- Example usage scripts
- Detailed documentation

## Usage

### Quick Test (No Dataset Required)

```bash
python run_captioning.py --use-dummy --max-samples 50
```

### Full COCO Experiment

```bash
# Download COCO dataset first (see README.md)
python run_captioning.py --max-samples 5000
```

## Expected Results

### Efficiency Metrics

Based on BLIP-base model on typical hardware:

| Metric | GPU (CUDA) | CPU |
|--------|------------|-----|
| Peak VRAM | ~1.4 GB | N/A |
| Latency | ~120 ms/image | ~800 ms/image |
| Throughput | ~8.3 img/sec | ~1.3 img/sec |
| Model Size | ~500 MB | ~500 MB |

### Captioning Metrics

Expected performance on COCO 2017 validation:

| Metric | Expected Range |
|--------|----------------|
| CIDEr | 1.0 - 1.2 |
| BLEU-4 | 0.35 - 0.40 |
| SPICE | 0.20 - 0.25 |

## Output Files

After running an experiment, check `results/` directory:

1. **predictions.json**: All generated captions for each image
2. **random_samples.json**: 25 random samples for human evaluation
3. **metrics.json**: Complete efficiency and evaluation metrics

## Documentation

- **README.md**: Project overview and setup
- **USAGE.md**: Detailed usage instructions and troubleshooting
- **example_usage.py**: Example code for individual components
- **test_basic.py**: Automated test suite

## Assignment Requirements Checklist

### Track: Image Captioning

- [x] Dataset: COCO 2017 validation (5k images) support
- [x] Alternative: Dummy dataset for testing

### Efficiency Metrics (All Required)

- [x] Peak VRAM (MiB) - Tracked using PyTorch CUDA memory
- [x] Latency per image-prompt - Measured for each inference
- [x] Throughput (img/s) - Calculated from average latency
- [x] Model size on disk - Measured from model parameters

### Energy Metrics (Bonus)

- [x] Energy estimation with nvidia-smi logging
- [x] Power usage tracking (Watts)
- [x] Total energy calculation (Watt-hours)

### Evaluation Metrics

- [x] CIDEr (primary) - Main evaluation metric
- [x] BLEU-4 - N-gram based metric
- [x] SPICE - Semantic evaluation (when available)

### Deliverables

- [x] Random 25-sample human spot-check
- [x] Automated metrics reporting
- [x] JSON output files for all results
- [x] Comprehensive documentation

## Technical Details

### Model Choice: BLIP

**Why BLIP?**
- Efficient: ~500MB model size
- Effective: Strong performance on captioning tasks
- Well-supported: Available on HuggingFace
- Flexible: Works on both GPU and CPU
- Sustainable: Good performance-to-efficiency ratio

### Implementation Highlights

1. **Minimal Dependencies**: Only essential libraries required
2. **Modular Design**: Each component can be used independently
3. **Flexible Configuration**: Easy to modify via config.py
4. **Error Handling**: Graceful fallbacks for missing resources
5. **Testing Support**: Dummy dataset for development/testing

### Sustainability Considerations

- Efficient model selection (BLIP-base vs larger alternatives)
- Energy monitoring to track power consumption
- Batch size optimization for memory efficiency
- CPU fallback option for systems without GPU
- Clear documentation to minimize trial-and-error

## Development Process

1. ✅ Created project structure
2. ✅ Implemented core modules (model, dataset, metrics)
3. ✅ Added main experiment script
4. ✅ Created comprehensive documentation
5. ✅ Added test suite and examples
6. ✅ Verified all components work correctly
7. ✅ Ran security analysis (0 vulnerabilities)

## Testing

### Automated Tests

```bash
python test_basic.py
```

Tests cover:
- Dummy dataset creation
- Efficiency monitoring
- Caption evaluation
- Model loading (when internet available)

### Manual Testing

```bash
# Test with dummy data
python run_captioning.py --use-dummy --max-samples 10

# Test example scripts
python example_usage.py
```

## Known Limitations

1. **Model Download**: Requires internet access to download BLIP model on first run
2. **SPICE Metric**: Requires Java runtime environment
3. **Energy Tracking**: Only available with NVIDIA GPUs and nvidia-smi
4. **Dataset Size**: Full COCO dataset is ~1GB (images) + 241MB (annotations)

## Future Enhancements

Potential improvements for future work:
- Support for additional models (CLIP+GPT, Git, etc.)
- Multi-GPU support for parallel processing
- Streaming dataset loading for memory efficiency
- Web interface for interactive captioning
- Fine-tuning capabilities on custom datasets

## References

- **BLIP Paper**: Li et al., "BLIP: Bootstrapping Language-Image Pre-training" (2022)
- **COCO Dataset**: Lin et al., "Microsoft COCO: Common Objects in Context" (2014)
- **CIDEr Metric**: Vedantam et al., "CIDEr: Consensus-based Image Description Evaluation" (2015)

## Contact & Support

For questions or issues:
1. Check USAGE.md for troubleshooting
2. Review example_usage.py for code examples
3. Run test_basic.py to verify installation

## License

This project is for educational purposes as part of the Sustainable AI course.
