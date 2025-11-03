"""Find the vision tower module name in Qwen2-VL model."""

import torch
from transformers import Qwen2VLForConditionalGeneration
import config

print("Loading model to inspect architecture...")
print("(This will load on CPU to avoid GPU memory issues)\n")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

print("=" * 80)
print("TOP-LEVEL MODULE NAMES")
print("=" * 80)
for name, _ in model.named_children():
    print(f"  {name}")

print("\n" + "=" * 80)
print("ALL MODULE NAMES (first 50)")
print("=" * 80)
all_modules = list(model.named_modules())
for i, (name, module) in enumerate(all_modules[:50]):
    module_type = type(module).__name__
    print(f"  {name:60s} {module_type}")

print(f"\n... ({len(all_modules)} total modules)")

print("\n" + "=" * 80)
print("VISION/VISUAL RELATED MODULES")
print("=" * 80)
vision_keywords = ['vis', 'image', 'patch', 'vit', 'visual', 'vision']
found = False
for name, module in model.named_modules():
    if any(keyword in name.lower() for keyword in vision_keywords):
        module_type = type(module).__name__
        print(f"  {name:60s} {module_type}")
        found = True

if not found:
    print("  No modules found with vision-related keywords")

print("\n" + "=" * 80)
print("SUGGESTION FOR llm_int8_skip_modules")
print("=" * 80)
print("Based on top-level modules, try one of these:")
for name, _ in model.named_children():
    if any(keyword in name.lower() for keyword in vision_keywords):
        print(f'  llm_int8_skip_modules=["{name}"]')
