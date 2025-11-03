"""Common inference logic for all model configurations."""

import os

import torch
from PIL import Image

import config


def run_inference(
    processor,
    model,
    device: str,
    dtype: torch.dtype,
    num_images: int = None,
    save_captions: bool = False,
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
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    all_images_files = [
        f for f in os.listdir(config.IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    subset = all_images_files[:num_images] if num_images else all_images_files

    for idx, fname in enumerate(subset, 1):
        images_path = os.path.join(config.IMAGES_DIR, fname)
        out_path = os.path.join(config.OUTPUT_DIR, os.path.splitext(fname)[0] + ".txt")

        if os.path.exists(out_path):
            print(f"{fname} already captioned")
            continue

        try:
            image = Image.open(images_path).convert("RGB")
        except Exception as e:
            print(f"Couldn't open {images_path}: {e}")
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

        print(f"[{idx}/{len(subset)}] {fname}: {caption}")

    print("Run complete.")
