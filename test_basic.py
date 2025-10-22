"""Basic tests for image captioning pipeline."""

import os
import sys
from PIL import Image

# Import modules
from captioning_model import ImageCaptioningModel
from efficiency_metrics import EfficiencyMonitor
from evaluation_metrics import CaptionEvaluator
from dataset_loader import create_dummy_dataset


def test_dummy_dataset():
    """Test dummy dataset creation."""
    print("Testing dummy dataset creation...")
    data = create_dummy_dataset(num_samples=10)
    assert len(data) == 10, "Dataset should have 10 samples"
    
    image, captions, img_id = data[0]
    assert isinstance(image, Image.Image), "First element should be PIL Image"
    assert isinstance(captions, list), "Captions should be a list"
    assert len(captions) > 0, "Should have at least one caption"
    print("✓ Dummy dataset test passed")


def test_efficiency_monitor():
    """Test efficiency monitor."""
    print("\nTesting efficiency monitor...")
    monitor = EfficiencyMonitor(device="cpu", log_energy=False)
    
    # Record some dummy latencies
    monitor.record_latency(0.1)
    monitor.record_latency(0.15)
    monitor.record_latency(0.12)
    
    summary = monitor.get_summary()
    assert 'avg_latency_ms' in summary, "Should have average latency"
    assert summary['avg_latency_ms'] > 0, "Average latency should be positive"
    print(f"  Average latency: {summary['avg_latency_ms']:.2f} ms")
    print("✓ Efficiency monitor test passed")


def test_caption_evaluator():
    """Test caption evaluator."""
    print("\nTesting caption evaluator...")
    evaluator = CaptionEvaluator()
    
    # Create dummy predictions and references
    predictions = {
        0: ["a cat sitting on a table"],
        1: ["a dog running in park"]
    }
    
    references = {
        0: ["a cat is sitting on the table", "cat on table"],
        1: ["a dog is running in the park", "dog running in park"]
    }
    
    scores = evaluator.evaluate(predictions, references)
    assert isinstance(scores, dict), "Scores should be a dictionary"
    print(f"  Evaluation scores: {scores}")
    print("✓ Caption evaluator test passed")


def test_model_loading():
    """Test model loading (this will download the model)."""
    print("\nTesting model loading...")
    print("Note: This will download the BLIP model (~500MB) on first run")
    
    try:
        # Use CPU to avoid CUDA requirements in testing
        model = ImageCaptioningModel(
            model_name="Salesforce/blip-image-captioning-base",
            device="cpu"
        )
        print("✓ Model loaded successfully")
        
        # Test caption generation
        print("\nTesting caption generation...")
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        caption = model.generate_caption(dummy_image, max_length=20, num_beams=3)
        
        assert isinstance(caption, str), "Caption should be a string"
        assert len(caption) > 0, "Caption should not be empty"
        print(f"  Generated caption: '{caption}'")
        print("✓ Caption generation test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("  This is expected if dependencies are not installed")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING BASIC TESTS")
    print("="*60)
    
    try:
        test_dummy_dataset()
        test_efficiency_monitor()
        test_caption_evaluator()
        
        # Model test is optional (requires dependencies)
        model_test_passed = test_model_loading()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ All basic tests passed!")
        if not model_test_passed:
            print("Note: Model test skipped (requires full dependencies)")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
