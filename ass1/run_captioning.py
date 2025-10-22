"""Main script to run image captioning experiments."""

import os
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

from config import *
from dataset_loader import create_coco_dataloader, create_dummy_dataset, COCOCaptionDataset
from captioning_model import ImageCaptioningModel
from efficiency_metrics import EfficiencyMonitor
from evaluation_metrics import CaptionEvaluator, save_predictions, save_metrics


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    model: ImageCaptioningModel,
    data: List,
    efficiency_monitor: EfficiencyMonitor,
    max_samples: int = None,
    log_interval: int = 100
) -> Dict[int, str]:
    """
    Run captioning experiment on dataset.
    
    Args:
        model: Image captioning model
        data: Dataset or list of (image, captions, image_id) tuples
        efficiency_monitor: Efficiency monitor instance
        max_samples: Maximum number of samples to process
        log_interval: Logging interval
        
    Returns:
        Dictionary mapping image_id to predicted caption
    """
    predictions = {}
    
    # Reset efficiency metrics
    efficiency_monitor.reset()
    
    print(f"\nRunning captioning on {len(data)} images...")
    
    for idx, batch in enumerate(tqdm(data, desc="Generating captions")):
        if max_samples and idx >= max_samples:
            break
        
        # Extract batch data (handle both dataloader and list)
        if isinstance(batch, list):
            image, captions, image_id = batch[0]
        else:
            image, captions, image_id = batch
        
        # Measure inference time
        start_time = time.time()
        
        # Generate caption
        predicted_caption = model.generate_caption(
            image,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS
        )
        
        # Record latency
        latency = time.time() - start_time
        efficiency_monitor.record_latency(latency)
        
        # Update VRAM metrics
        efficiency_monitor.update_peak_vram()
        
        # Record energy sample periodically
        if idx % 10 == 0:
            efficiency_monitor.record_energy_sample()
        
        # Store prediction
        predictions[image_id] = predicted_caption
        
        # Log progress
        if (idx + 1) % log_interval == 0:
            print(f"\nProcessed {idx + 1} images")
            print(f"Sample prediction: {predicted_caption}")
            current_summary = efficiency_monitor.get_summary()
            print(f"Current avg latency: {current_summary.get('avg_latency_ms', 0):.2f} ms")
    
    return predictions


def get_random_sample(predictions: Dict[int, str], references: Dict[int, List[str]], n: int = 25):
    """
    Get random sample of predictions for human spot-check.
    
    Args:
        predictions: Dictionary mapping image_id to predicted caption
        references: Dictionary mapping image_id to reference captions
        n: Number of samples
        
    Returns:
        List of samples
    """
    common_ids = list(set(predictions.keys()) & set(references.keys()))
    sample_ids = random.sample(common_ids, min(n, len(common_ids)))
    
    samples = []
    for img_id in sample_ids:
        samples.append({
            'image_id': img_id,
            'prediction': predictions[img_id],
            'references': references[img_id]
        })
    
    return samples


def print_random_samples(samples: List[Dict], n: int = 10):
    """Print random samples for visual inspection."""
    print("\n" + "="*80)
    print(f"RANDOM SAMPLE OF PREDICTIONS (showing {min(n, len(samples))} of {len(samples)})")
    print("="*80)
    
    for i, sample in enumerate(samples[:n]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Image ID: {sample['image_id']}")
        print(f"Predicted: {sample['prediction']}")
        print(f"References:")
        for j, ref in enumerate(sample['references'][:3]):  # Show max 3 references
            print(f"  {j+1}. {ref}")
    
    print("="*80 + "\n")


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Run image captioning experiment")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--use-dummy", action="store_true", help="Use dummy dataset for testing")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(RANDOM_SEED)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize efficiency monitor
    print("Initializing efficiency monitor...")
    efficiency_monitor = EfficiencyMonitor(device=args.device, log_energy=MEASURE_ENERGY)
    
    # Load model
    print("\nLoading image captioning model...")
    model = ImageCaptioningModel(model_name=args.model_name, device=args.device)
    
    # Get model size
    efficiency_monitor.get_model_size(model=model.get_model())
    print(f"Model size: {efficiency_monitor.model_size_mb:.2f} MB")
    
    # Load dataset
    print("\nLoading dataset...")
    if args.use_dummy:
        print("Using dummy dataset for testing...")
        data = create_dummy_dataset(num_samples=args.max_samples or 100)
        # Create dummy references
        references = {img_id: captions for _, captions, img_id in data}
    else:
        try:
            dataset = COCOCaptionDataset(
                data_dir=args.data_dir,
                split="val2017",
                max_samples=args.max_samples
            )
            data = [(dataset[i]) for i in range(len(dataset))]
            references = dataset.get_annotations_dict()
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nCOCO dataset not found. Using dummy dataset instead...")
            print("To use real COCO dataset:")
            print("  1. Download images: http://images.cocodataset.org/zips/val2017.zip")
            print("  2. Download annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            print(f"  3. Extract to: {args.data_dir}")
            print("\nContinuing with dummy dataset...")
            data = create_dummy_dataset(num_samples=args.max_samples or 100)
            references = {img_id: captions for _, captions, img_id in data}
    
    # Run experiment
    print("\n" + "="*80)
    print("STARTING IMAGE CAPTIONING EXPERIMENT")
    print("="*80)
    
    predictions = run_experiment(
        model=model,
        data=data,
        efficiency_monitor=efficiency_monitor,
        max_samples=args.max_samples,
        log_interval=LOG_INTERVAL
    )
    
    print(f"\nGenerated {len(predictions)} captions")
    
    # Evaluate
    print("\nEvaluating predictions...")
    evaluator = CaptionEvaluator()
    
    # Convert predictions to evaluation format
    pred_dict = {img_id: [caption] if isinstance(caption, str) else caption 
                 for img_id, caption in predictions.items()}
    
    scores = evaluator.evaluate(pred_dict, references)
    
    # Get random sample for human spot-check
    random_samples = get_random_sample(predictions, references, n=25)
    
    # Print results
    efficiency_monitor.print_summary()
    evaluator.print_scores(scores)
    print_random_samples(random_samples, n=10)
    
    # Save results
    if SAVE_PREDICTIONS:
        pred_path = os.path.join(args.output_dir, "predictions.json")
        save_predictions(predictions, pred_path)
        
        sample_path = os.path.join(args.output_dir, "random_samples.json")
        import json
        with open(sample_path, 'w') as f:
            json.dump(random_samples, f, indent=2)
        print(f"Random samples saved to {sample_path}")
    
    if SAVE_METRICS:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        save_metrics(scores, efficiency_monitor.get_summary(), metrics_path)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
