"""
Example usage of the image captioning pipeline.

This script demonstrates how to use individual components
for custom experiments.
"""

from PIL import Image
from dataset_loader import create_dummy_dataset
from captioning_model import ImageCaptioningModel
from efficiency_metrics import EfficiencyMonitor
from evaluation_metrics import CaptionEvaluator


def example_basic_captioning():
    """Example: Basic image captioning."""
    print("=" * 60)
    print("Example 1: Basic Image Captioning")
    print("=" * 60)
    
    # Create a simple test image
    image = Image.new('RGB', (224, 224), color='blue')
    
    # Load model (will try to download, may fail without internet)
    try:
        model = ImageCaptioningModel(
            model_name="Salesforce/blip-image-captioning-base",
            device="cpu"  # Use CPU for this example
        )
        
        # Generate caption
        caption = model.generate_caption(image, max_length=50, num_beams=5)
        print(f"\nGenerated caption: {caption}")
        
    except Exception as e:
        print(f"Model loading failed (expected without internet): {e}")
        print("In real usage, this would work with internet access.")


def example_efficiency_monitoring():
    """Example: Monitoring efficiency metrics."""
    print("\n" + "=" * 60)
    print("Example 2: Efficiency Monitoring")
    print("=" * 60)
    
    # Initialize monitor
    monitor = EfficiencyMonitor(device="cpu", log_energy=False)
    
    # Simulate some inference runs
    import time
    for i in range(5):
        start = time.time()
        # Simulate inference work
        time.sleep(0.1)
        latency = time.time() - start
        monitor.record_latency(latency)
    
    # Get summary
    summary = monitor.get_summary()
    print(f"\nEfficiency Summary:")
    print(f"  Average Latency: {summary['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {summary['throughput_img_per_sec']:.2f} img/sec")


def example_evaluation():
    """Example: Evaluating captions."""
    print("\n" + "=" * 60)
    print("Example 3: Caption Evaluation")
    print("=" * 60)
    
    # Create sample predictions and references
    predictions = {
        0: ["a cat sitting on a table"],
        1: ["a dog running in the park"],
        2: ["a bird flying in the sky"]
    }
    
    references = {
        0: ["a cat is sitting on the table", "cat on table", "the cat sits"],
        1: ["a dog is running in the park", "dog running outdoors"],
        2: ["a bird flies in the sky", "bird in flight"]
    }
    
    # Evaluate
    evaluator = CaptionEvaluator()
    scores = evaluator.evaluate(predictions, references)
    
    print("\nEvaluation Scores:")
    for metric, score in scores.items():
        if isinstance(score, (int, float)):
            print(f"  {metric}: {score:.4f}")
        else:
            print(f"  {metric}: {score}")


def example_dummy_dataset():
    """Example: Working with dummy dataset."""
    print("\n" + "=" * 60)
    print("Example 4: Dummy Dataset")
    print("=" * 60)
    
    # Create dummy dataset
    data = create_dummy_dataset(num_samples=5)
    
    print(f"\nCreated {len(data)} dummy samples")
    
    # Inspect first sample
    image, captions, img_id = data[0]
    print(f"\nFirst sample:")
    print(f"  Image ID: {img_id}")
    print(f"  Image size: {image.size}")
    print(f"  Reference captions: {captions}")


def example_complete_pipeline():
    """Example: Complete pipeline with dummy data."""
    print("\n" + "=" * 60)
    print("Example 5: Complete Pipeline")
    print("=" * 60)
    
    # Create dummy data
    data = create_dummy_dataset(num_samples=3)
    
    # Initialize components
    monitor = EfficiencyMonitor(device="cpu", log_energy=False)
    evaluator = CaptionEvaluator()
    
    # Simulate captioning (without actual model)
    predictions = {}
    references = {}
    
    import time
    for image, captions, img_id in data:
        # Record latency (simulated)
        start = time.time()
        time.sleep(0.05)  # Simulate inference
        latency = time.time() - start
        monitor.record_latency(latency)
        
        # Fake prediction
        predictions[img_id] = [f"Sample caption for image {img_id}"]
        references[img_id] = captions
    
    # Get results
    efficiency = monitor.get_summary()
    scores = evaluator.evaluate(predictions, references)
    
    print("\nPipeline Results:")
    print(f"  Processed {len(predictions)} images")
    print(f"  Average Latency: {efficiency['avg_latency_ms']:.2f} ms")
    print(f"  Evaluation Score: {scores}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IMAGE CAPTIONING - EXAMPLE USAGE")
    print("=" * 60)
    
    # Run examples
    example_dummy_dataset()
    example_efficiency_monitoring()
    example_evaluation()
    example_complete_pipeline()
    
    # This one may fail without internet
    example_basic_captioning()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Install all dependencies: pip install -r requirements.txt")
    print("  2. Test with dummy data: python run_captioning.py --use-dummy --max-samples 50")
    print("  3. Download COCO dataset and run full experiment")
    print("  4. Check USAGE.md for detailed instructions")
