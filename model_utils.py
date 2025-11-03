"""Common utilities for model loading and device configuration."""

import time
from typing import Tuple

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

import config
from logger import setup_logger

logger = setup_logger(__name__)


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


def get_dtype(device: str, quantization_mode: str = "none") -> torch.dtype:
    """
    Get appropriate dtype based on device and quantization mode.

    Args:
        device: Device string ("cuda" or "cpu")
        quantization_mode: Quantization mode ("none", "skip_vision_tower", "full")

    Returns:
        torch.dtype for model inference
    """
    if device == "cpu":
        return torch.float32

    # Use float16 for quantized models (matches BitsAndBytes output)
    # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1030#issuecomment-2691474347
    if quantization_mode in ["skip_vision_tower", "full"]:
        return torch.float16

    # Use bfloat16 for unquantized models (better stability)
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
    dtype = get_dtype(device, quantization_mode)
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Load processor
    start_time = time.time()
    logger.debug(f"Loading processor from: {config.SMOLVLM_MODEL}")
    processor = AutoProcessor.from_pretrained(config.SMOLVLM_MODEL)
    logger.debug(f"Processor loaded in {time.time() - start_time:.2f}s")

    # Load model
    model_start_time = time.time()

    if quantization_mode == "none":
        # Load unquantized model
        logger.info("Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            config.SMOLVLM_MODEL,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
    elif quantization_mode in ["skip_vision_tower", "full"]:
        # Load quantized model (requires CUDA)
        if device != "cuda":
            raise RuntimeError("Quantization requires CUDA device")

        logger.debug(f"Creating {quantization_mode} quantization config")
        if quantization_mode == "skip_vision_tower":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_tower", "visual"]
            )
        else:  # full
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading model with {quantization_mode} quantization...")
        model = AutoModelForImageTextToText.from_pretrained(
            config.SMOLVLM_MODEL,
            device_map="auto",
            quantization_config=quantization_config,
        )
    else:
        raise ValueError(f"Invalid quantization mode: {quantization_mode}")

    model_load_time = time.time() - model_start_time
    logger.info(f"Model weights loaded in {model_load_time:.2f}s")

    model.eval()
    total_time = time.time() - start_time
    logger.info(f"Model ready (total: {total_time:.2f}s)")
    return processor, model, device, dtype
