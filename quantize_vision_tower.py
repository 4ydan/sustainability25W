"""
Vision Tower INT8 Quantization using bitsandbytes.

Compares inference speed and model size between quantized and original models.
"""

import os
import time
from typing import Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

import config


class VisionTowerQuantizer:
    """Handles quantization of vision tower layers using bitsandbytes INT8."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.dtype = self._get_dtype()

    def _setup_device(self, device: str) -> str:
        """Configure device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "gpu":
            return "cuda"
        return device

    def _get_dtype(self) -> torch.dtype:
        """Get appropriate dtype based on device."""
        if self.device == "cpu":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def load_model(self, quantize: bool = False) -> Tuple[AutoProcessor, Qwen2VLForConditionalGeneration]:
        """Load model with optional INT8 quantization."""
        print(f"\nLoading {'quantized' if quantize else 'original'} model...")

        processor = AutoProcessor.from_pretrained(self.model_name)

        if quantize and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

        model.eval()
        return processor, model

    def benchmark(self, processor, model, image_path: str, num_runs: int = 3) -> float:
        """Run inference and return average time in ms."""
        Image.open(image_path).convert("RGB")  # Verify image loads

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": config.PROMPT},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(self.device)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
                else:
                    model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
            times.append((time.perf_counter() - start) * 1000)

        return sum(times) / len(times)


def main():
    """Benchmark speed and size comparison."""
    images_dir = config.IMAGES_DIR

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        print("Please run main.py first to download the dataset.")
        return

    all_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not all_images:
        print(f"No images found in {images_dir}")
        return

    test_image = all_images[0]  # Use one image for benchmark
    print(f"Benchmarking with: {os.path.basename(test_image)}")

    quantizer = VisionTowerQuantizer(config.MODEL_NAME, device=config.DEVICE)

    print("\n" + "=" * 80)
    print("QUANTIZATION BENCHMARK")
    print("=" * 80)

    # Benchmark original model
    processor_orig, model_orig = quantizer.load_model(quantize=False)
    orig_time = quantizer.benchmark(processor_orig, model_orig, test_image)
    orig_size = sum(p.numel() * p.element_size() for p in model_orig.parameters()) / (1024 ** 2)

    del model_orig, processor_orig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Benchmark quantized model
    processor_quant, model_quant = quantizer.load_model(quantize=True)
    quant_time = quantizer.benchmark(processor_quant, model_quant, test_image)
    quant_size = sum(p.numel() * p.element_size() for p in model_quant.parameters()) / (1024 ** 2)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nInference Time (avg of 3 runs):")
    print(f"  Original:  {orig_time:.1f} ms")
    print(f"  Quantized: {quant_time:.1f} ms")
    print(f"  Speedup:   {orig_time/quant_time:.2f}x")
    print(f"\nModel Size:")
    print(f"  Original:  {orig_size:.1f} MB")
    print(f"  Quantized: {quant_size:.1f} MB")
    print(f"  Compression: {orig_size/quant_size:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
