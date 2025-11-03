"""Common utilities for model loading and device configuration."""

from typing import Tuple

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

import config


def setup_device(device_config: str) -> str:
    """
    Configure device for inference.

    Args:
        device_config: Device configuration from config ("auto", "gpu", "cpu")

    Returns:
        Device string ("cuda" or "cpu")
    """
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_config == "gpu":
        return "cuda"
    return device_config


def get_dtype(device: str) -> torch.dtype:
    """
    Get appropriate dtype based on device.

    Args:
        device: Device string ("cuda" or "cpu")

    Returns:
        torch.dtype for model inference
    """
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(
    device_config: str,
    quantization_mode: str = "none"
) -> Tuple[AutoProcessor, AutoModelForImageTextToText, str, torch.dtype]:
    """
    Load SmolVLM model with optional quantization.

    Args:
        device_config: Device configuration ("auto", "cuda", "cpu")
        quantization_mode: "none", "skip_vision_tower", or "full"

    Returns:
        Tuple of (processor, model, device, dtype)
    """
    device = setup_device(device_config)
    dtype = get_dtype(device)

    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Loading model: {config.SMOLVLM_MODEL}")
    if quantization_mode != "none":
        print(f"Quantization mode: {quantization_mode}")

    processor = AutoProcessor.from_pretrained(config.SMOLVLM_MODEL)

    if quantization_mode == "none":
        # Load unquantized model
        model = AutoModelForImageTextToText.from_pretrained(
            config.SMOLVLM_MODEL,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
    elif quantization_mode in ["skip_vision_tower", "full"]:
        # Load quantized model (requires CUDA)
        if device != "cuda":
            raise RuntimeError("Quantization requires CUDA device")

        if quantization_mode == "skip_vision_tower":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_tower", "visual"]
            )
        else:  # full
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForImageTextToText.from_pretrained(
            config.SMOLVLM_MODEL,
            device_map="auto",
            quantization_config=quantization_config,
        )
    else:
        raise ValueError(f"Invalid quantization mode: {quantization_mode}")

    model.eval()
    return processor, model, device, dtype
