"""Common inference logic for all model configurations."""

import os

import torch
from PIL import Image

import config
from logger import setup_logger

logger = setup_logger(__name__)


def run_inference(
    processor,
    model,
    device: str,
    dtype: torch.dtype,
    num_images: int = None,
    save_captions: bool = False,
    quantization_mode: str = "none",
):
    """
    Run inference on images using any loaded model.

    Args:
        processor: Model processor
        model: Loaded model
        device: Device string ("cuda" or "cpu")
        dtype: Model dtype
        num_images: Number of images to process (None for all)
        save_captions: Whether to save captions to disk
        quantization_mode: Quantization mode (none, skip_vision_tower, full)
    """
    # Create output directory based on quantization mode
    output_dir = f"{config.OUTPUT_DIR}_{quantization_mode}"
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    all_images_files = [
        f for f in os.listdir(config.IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    subset = all_images_files[:num_images] if num_images else all_images_files
    logger.info(f"Processing {len(subset)} images...")

    for idx, fname in enumerate(subset, 1):
        images_path = os.path.join(config.IMAGES_DIR, fname)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".txt")

        if os.path.exists(out_path):
            logger.debug(f"{fname} already captioned")
            continue

        try:
            image = Image.open(images_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Couldn't open {images_path}: {e}")
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": config.PROMPT},
            ],
        }]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)

        generated_ids = [
            out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, output_ids)
        ]
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if save_captions:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(caption)

        logger.info(f"[{idx}/{len(subset)}] {fname}: {caption}")

    logger.info(f"Completed {len(subset)} images")
