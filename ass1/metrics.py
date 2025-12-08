"""Functions used to calculate the performance metrics of each model."""

from typing import List, Dict, Any
import torch



def get_model_size(model) -> float:
   """
    Used to calculate the size of a mode based on the amount of parameters it contains

    Args:
        model: The model of which the size of should be calculated

    Returns:
        The size in mbs
   """
   total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
   return total_bytes / (1024 ** 2)



def compute_performance_metrics(
    per_image_latencies: List[float],
    num_processed: int,
    total_time: float,
    model,
    device: str,
) -> Dict[str, Any]:
    """
    This function computes various performance metrics such as VRAM usage, latency per image, throughput and model size 
    and saves them into a dictionary

    Args:
        per_image_latencies: for all images that were processed the times it took the model
        num_processed: the total amount of images that were processed
        total_time: the total time that the model took to run
        model: the specified model
        device: the device used to run the model could for example be "cuda" or "cpu"

    Returns:
        a dictionary that contains all the afore mentioned stats
    """
    if num_processed > 0:
        latency_per_image = sum(per_image_latencies) / num_processed
        throughput = num_processed / total_time if total_time > 0 else 0.0
    else:
        latency_per_image = 0.0
        throughput = 0.0

    peak_vram = None
    if device == "cuda" and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model_size = get_model_size(model)

    return {
        "peak_VRAM": peak_vram,
        "latency_per_image": latency_per_image,
        "throughput": throughput,
        "model_size": model_size,
    }
