import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import config

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# device setup
if config.DEVICE == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
else:
    device = config.DEVICE
    dtype = torch.float32 if device == "cpu" else torch.float16

print(f"Using device: {device}, dtype: {dtype}")

# load model
processor = AutoProcessor.from_pretrained(config.MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.MODEL_NAME, dtype=dtype, device_map="auto" if device == "cuda" else None
)
model.eval()

# caption
all_images_files = [
    f for f in os.listdir(config.IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

subset = all_images_files[:config.NUM_IMAGES] if config.NUM_IMAGES else all_images_files

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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images_path},
                {"type": "text", "text": config.PROMPT},
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
    ).to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                output_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
        else:
            output_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)

    generated_ids = [
        out[len(in_ids) :] for in_ids, out in zip(inputs.input_ids, output_ids)
    ]
    caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if config.SAVE_CAPTIONS:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(caption)

    print(f"[{idx}/{len(subset)}] {fname}: {caption}")

print("Run complete.")
