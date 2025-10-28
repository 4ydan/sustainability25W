import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# config
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
IMAGES_DIR = "./data/coco2017/val2017/"
OUTPUT_DIR = "./data/coco2017/captions_val2017"
PROMPT = "Describe this image in one detailed caption"
MAX_NEW_TOKENS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# check cuda availability
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16
else:
    print("cuda not available, defaulting to CPU")
    device = "cpu"
    dtype = torch.float32

# load model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
)
model.eval()

# caption
all_images_files = [
    f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

subset = all_images_files[:10]

# WARN: change subset to full images_files after testing is done
for idx, fname in enumerate(subset, 1):
    images_path = os.path.join(IMAGES_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname + ".txt")

    if os.path.exists(out_path):
        print(f"{fname} already captioned")
        continue

    try:
        image = Image.open(images_path).convert("RGB")
    except Exception as e:
        print(f"Couldn't open {images_path}: {e}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=True, add_generationprompt=True
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
                output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        else:
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids = [
        out[len(in_ids) :] for in_ids, out in zip(inputs.input_ids, output_ids)
    ]
    caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(caption)

    print(f"[{idx}/{len(subset)}] {fname}: {caption}")

print("Run complete.")
