"""Common inference logic for all model configurations."""

import os
import time

import torch
from PIL import Image

import config
from logger import setup_logger

import json

logger = setup_logger(__name__)

def get_model_size(model: torch.nn.Module, dtype: torch.dtype) -> float:
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
    }
    bytes_per_param = dtype_sizes.get(dtype, 4)
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * bytes_per_param
    return total_bytes / (1024 ** 2)


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


    predictions = []
    per_image_latencies = []
    peak_vram = None
    num_processed = 0

    json_path = os.path.join(output_dir, "predicted_captions.json")
    
    existing_predictions = {}

    if(os.path.exists(json_path)):
        with open(json_path, "r", encoding="utf-8") as jf:
            existing_list = json.load(jf)
            existing_predictions = {str(p["image_id"]): p["caption"] for p in existing_list}
          
     

        

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    

    overall_start = time.time()
    

    for idx, fname in enumerate(subset, 1):
        images_path = os.path.join(config.IMAGES_DIR, fname)
        image_id = os.path.splitext(fname)[0]

        if image_id in existing_predictions:
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

        img_start = time.time()
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)

        img_time = time.time() - img_start
        per_image_latencies.append(img_time)
        num_processed += 1

        generated_ids = [
            out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, output_ids)
        ]
        num_tokens = len(generated_ids[0])
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if save_captions:     
            image_id = os.path.splitext(fname)[0] 
            predictions.append({
                "image_id": image_id,
                "caption": caption,
            })


        logger.info(f"[{idx}/{len(subset)}] {fname} - {img_time:.2f}s, {num_tokens} tokens: {caption}")

    total_time = time.time() - overall_start

    latency_per_image = 0
    throughput = 0
    


    if num_processed > 0:
        latency_per_image = sum(per_image_latencies) / num_processed
        throughput = num_processed / total_time 

    if device == "cuda" and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) 

    model_size = get_model_size(model, dtype)

    metrics = {
        "peak_VRAM": peak_vram,
        "latency_per_image": latency_per_image,
        "througput": throughput,
        "model_size": model_size,
    }

    

    if save_captions and len(predictions) > 0:
        merged_predictions = {**existing_predictions, **{str(p["image_id"]): p["caption"] for p in predictions}}
        merged_list = [{"image_id": k, "caption": v} for k, v in merged_predictions.items()]

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(merged_list, jf, ensure_ascii=False, indent=2)
        logger.info(f"Saved captions to {json_path}")

        json_path = os.path.join(output_dir, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metrics, jf, ensure_ascii=False, indent=2)
        logger.info(f"Saved metrics to {json_path}")

    

    logger.info(f"Completed {len(subset)} images in {total_time:.2f}s")

    

