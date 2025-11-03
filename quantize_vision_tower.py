"""
Vision Tower INT8 Quantization using bitsandbytes.

This module implements INT8 quantization for the vision tower (ViT) component
of Qwen2-VL model using bitsandbytes, and compares accuracy drop between
the quantized and original models.
"""

import os
import time
from typing import Dict, List, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    import bitsandbytes as bnb
except ImportError:
    raise ImportError(
        "bitsandbytes not installed. Install with: uv add bitsandbytes"
    )

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
        """
        Load model with optional INT8 quantization.

        Args:
            quantize: If True, quantize vision tower to INT8

        Returns:
            Tuple of (processor, model)
        """
        print(f"\nLoading {'quantized' if quantize else 'original'} model...")
        print(f"Device: {self.device}, dtype: {self.dtype}")

        processor = AutoProcessor.from_pretrained(self.model_name)

        if quantize and self.device == "cuda":
            # Load model with 8-bit quantization config
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            )

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,  # Consistent dtype
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

        model.eval()

        # Print quantization status
        if quantize:
            self._print_quantization_info(model)

        return processor, model

    def _print_quantization_info(self, model):
        """Print information about quantized layers."""
        total_params = 0
        quantized_params = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                param_count = module.weight.numel()
                total_params += param_count

                # Check if using bnb.nn.Linear8bitLt
                if isinstance(module, bnb.nn.Linear8bitLt):
                    quantized_params += param_count
                    if 'visual' in name.lower() or 'vision' in name.lower():
                        print(f"  Quantized vision layer: {name}")

        print(f"\nQuantization summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Quantized parameters: {quantized_params:,}")
        print(f"  Quantization ratio: {quantized_params/total_params*100:.2f}%")

    def generate_caption(
        self,
        processor,
        model,
        image_path: str,
        prompt: str = "Describe this image in a caption",
        max_new_tokens: int = 128,
    ) -> Tuple[str, float]:
        """
        Generate caption for an image.

        Args:
            processor: Model processor
            model: Model instance
            image_path: Path to image
            prompt: Text prompt for captioning
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (caption, inference_time_ms)
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image {image_path}: {e}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        generated_ids = [
            out[len(in_ids) :] for in_ids, out in zip(inputs.input_ids, output_ids)
        ]
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return caption, inference_time


class AccuracyComparator:
    """Compare accuracy between quantized and original models."""

    def __init__(self, quantizer: VisionTowerQuantizer):
        self.quantizer = quantizer

    def compute_caption_similarity(self, caption1: str, caption2: str) -> float:
        """
        Compute similarity between two captions using token overlap.

        Args:
            caption1: First caption
            caption2: Second caption

        Returns:
            Similarity score (0-1)
        """
        # Simple token-based similarity (Jaccard similarity)
        tokens1 = set(caption1.lower().split())
        tokens2 = set(caption2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def compare_models(
        self,
        image_paths: List[str],
        prompt: str = "Describe this image in a caption",
        max_new_tokens: int = 128,
    ) -> Dict:
        """
        Compare quantized vs original model on a set of images.

        Args:
            image_paths: List of image paths to test
            prompt: Caption prompt
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with comparison metrics
        """
        print("\n" + "=" * 80)
        print("VISION TOWER QUANTIZATION COMPARISON")
        print("=" * 80)

        # Load original model
        processor_orig, model_orig = self.quantizer.load_model(quantize=False)

        # Load quantized model
        processor_quant, model_quant = self.quantizer.load_model(quantize=True)

        results = {
            "images": [],
            "original_captions": [],
            "quantized_captions": [],
            "similarities": [],
            "original_times": [],
            "quantized_times": [],
        }

        print(f"\nTesting on {len(image_paths)} images...\n")

        for idx, img_path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")

            # Generate with original model
            caption_orig, time_orig = self.quantizer.generate_caption(
                processor_orig, model_orig, img_path, prompt, max_new_tokens
            )

            # Generate with quantized model
            caption_quant, time_quant = self.quantizer.generate_caption(
                processor_quant, model_quant, img_path, prompt, max_new_tokens
            )

            # Compute similarity
            similarity = self.compute_caption_similarity(caption_orig, caption_quant)

            results["images"].append(os.path.basename(img_path))
            results["original_captions"].append(caption_orig)
            results["quantized_captions"].append(caption_quant)
            results["similarities"].append(similarity)
            results["original_times"].append(time_orig)
            results["quantized_times"].append(time_quant)

            print(f"  Original:  {caption_orig}")
            print(f"  Quantized: {caption_quant}")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Time - Original: {time_orig:.1f}ms, Quantized: {time_quant:.1f}ms")
            print()

        # Compute aggregate metrics
        avg_similarity = sum(results["similarities"]) / len(results["similarities"])
        avg_time_orig = sum(results["original_times"]) / len(results["original_times"])
        avg_time_quant = sum(results["quantized_times"]) / len(results["quantized_times"])
        speedup = avg_time_orig / avg_time_quant if avg_time_quant > 0 else 0

        # Calculate model sizes
        orig_size_mb = sum(p.numel() * p.element_size() for p in model_orig.parameters()) / (1024 ** 2)
        quant_size_mb = sum(
            p.numel() * p.element_size() for p in model_quant.parameters()
        ) / (1024 ** 2)

        results["summary"] = {
            "avg_similarity": avg_similarity,
            "avg_time_original_ms": avg_time_orig,
            "avg_time_quantized_ms": avg_time_quant,
            "speedup": speedup,
            "accuracy_drop": 1 - avg_similarity,
            "model_size_original_mb": orig_size_mb,
            "model_size_quantized_mb": quant_size_mb,
            "compression_ratio": orig_size_mb / quant_size_mb if quant_size_mb > 0 else 0,
        }

        self._print_summary(results["summary"])

        return results

    def _print_summary(self, summary: Dict):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Average Caption Similarity:    {summary['avg_similarity']:.3f}")
        print(f"Average Accuracy Drop:         {summary['accuracy_drop']:.3f} ({summary['accuracy_drop']*100:.1f}%)")
        print(f"\nInference Time:")
        print(f"  Original Model:  {summary['avg_time_original_ms']:.1f} ms")
        print(f"  Quantized Model: {summary['avg_time_quantized_ms']:.1f} ms")
        print(f"  Speedup:         {summary['speedup']:.2f}x")
        print(f"\nModel Size:")
        print(f"  Original:        {summary['model_size_original_mb']:.1f} MB")
        print(f"  Quantized:       {summary['model_size_quantized_mb']:.1f} MB")
        print(f"  Compression:     {summary['compression_ratio']:.2f}x")
        print("=" * 80)


def main():
    """Main execution function."""
    # Get test images
    images_dir = config.IMAGES_DIR

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        print("Please run main.py first to download the dataset.")
        return

    all_images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_images:
        print(f"No images found in {images_dir}")
        return

    # Use subset for testing (default to 5 images)
    num_test_images = min(5, len(all_images))
    test_images = all_images[:num_test_images]

    print(f"Found {len(all_images)} images, testing on {num_test_images}")

    # Run comparison
    quantizer = VisionTowerQuantizer(config.MODEL_NAME, device=config.DEVICE)
    comparator = AccuracyComparator(quantizer)

    results = comparator.compare_models(
        test_images,
        prompt=config.PROMPT,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    # Optionally save results
    if config.SAVE_CAPTIONS:
        output_file = os.path.join(config.OUTPUT_DIR, "quantization_comparison.txt")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Vision Tower Quantization Comparison Results\n")
            f.write("=" * 80 + "\n\n")

            for i, img in enumerate(results["images"]):
                f.write(f"Image: {img}\n")
                f.write(f"  Original:  {results['original_captions'][i]}\n")
                f.write(f"  Quantized: {results['quantized_captions'][i]}\n")
                f.write(f"  Similarity: {results['similarities'][i]:.3f}\n\n")

            f.write("\nSummary:\n")
            for key, value in results["summary"].items():
                f.write(f"  {key}: {value}\n")

        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
