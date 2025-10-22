"""Utilities for measuring efficiency metrics."""

import os
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch
import pynvml


class EfficiencyMonitor:
    """Monitor efficiency metrics during model inference."""
    
    def __init__(self, device: str = "cuda", log_energy: bool = True):
        """
        Initialize efficiency monitor.
        
        Args:
            device: Device to monitor (cuda or cpu)
            log_energy: Whether to log energy consumption
        """
        self.device = device
        self.log_energy = log_energy
        
        # Initialize NVML for GPU monitoring
        self.gpu_available = False
        if device == "cuda" and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
                self.gpu_available = False
        
        # Metrics storage
        self.peak_vram_mib = 0
        self.latencies = []
        self.energy_samples = []
        self.model_size_mb = 0
    
    def reset(self):
        """Reset all metrics."""
        self.peak_vram_mib = 0
        self.latencies = []
        self.energy_samples = []
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_vram_usage(self) -> float:
        """
        Get current VRAM usage in MiB.
        
        Returns:
            VRAM usage in MiB
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
    
    def get_peak_vram(self) -> float:
        """
        Get peak VRAM usage in MiB.
        
        Returns:
            Peak VRAM usage in MiB
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0
    
    def update_peak_vram(self):
        """Update peak VRAM metric."""
        current_peak = self.get_peak_vram()
        self.peak_vram_mib = max(self.peak_vram_mib, current_peak)
    
    def get_power_usage(self) -> Optional[float]:
        """
        Get current GPU power usage in watts.
        
        Returns:
            Power usage in watts, or None if not available
        """
        if not self.gpu_available:
            return None
        
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0  # Convert to watts
        except Exception as e:
            return None
    
    def record_latency(self, latency: float):
        """
        Record a latency measurement.
        
        Args:
            latency: Latency in seconds
        """
        self.latencies.append(latency)
    
    def record_energy_sample(self):
        """Record current energy/power sample."""
        if self.log_energy and self.gpu_available:
            power = self.get_power_usage()
            if power is not None:
                self.energy_samples.append({
                    'timestamp': time.time(),
                    'power_watts': power
                })
    
    def get_model_size(self, model_path: Optional[str] = None, model=None) -> float:
        """
        Get model size on disk in MB.
        
        Args:
            model_path: Path to model file
            model: PyTorch model (will estimate size from parameters)
            
        Returns:
            Model size in MB
        """
        if model_path and os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            self.model_size_mb = size_bytes / (1024 ** 2)
        elif model is not None:
            # Estimate from parameter count
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            self.model_size_mb = param_size / (1024 ** 2)
        
        return self.model_size_mb
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all efficiency metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        summary = {
            'peak_vram_mib': self.peak_vram_mib,
            'model_size_mb': self.model_size_mb,
        }
        
        if self.latencies:
            summary['avg_latency_ms'] = (sum(self.latencies) / len(self.latencies)) * 1000
            summary['median_latency_ms'] = sorted(self.latencies)[len(self.latencies) // 2] * 1000
            summary['throughput_img_per_sec'] = 1.0 / (sum(self.latencies) / len(self.latencies))
        
        if self.energy_samples:
            avg_power = sum(s['power_watts'] for s in self.energy_samples) / len(self.energy_samples)
            total_time = self.energy_samples[-1]['timestamp'] - self.energy_samples[0]['timestamp']
            summary['avg_power_watts'] = avg_power
            summary['total_energy_wh'] = (avg_power * total_time) / 3600  # Watt-hours
        
        return summary
    
    def print_summary(self):
        """Print formatted summary of efficiency metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("EFFICIENCY METRICS SUMMARY")
        print("="*60)
        print(f"Peak VRAM Usage:        {summary.get('peak_vram_mib', 0):.2f} MiB")
        print(f"Model Size on Disk:     {summary.get('model_size_mb', 0):.2f} MB")
        
        if 'avg_latency_ms' in summary:
            print(f"Average Latency:        {summary['avg_latency_ms']:.2f} ms/image")
            print(f"Median Latency:         {summary['median_latency_ms']:.2f} ms/image")
            print(f"Throughput:             {summary['throughput_img_per_sec']:.2f} images/sec")
        
        if 'avg_power_watts' in summary:
            print(f"\nENERGY METRICS (Bonus):")
            print(f"Average Power:          {summary['avg_power_watts']:.2f} W")
            print(f"Total Energy:           {summary['total_energy_wh']:.4f} Wh")
        
        print("="*60 + "\n")
    
    def __del__(self):
        """Cleanup NVML."""
        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def measure_inference_time(func):
    """Decorator to measure inference time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper
