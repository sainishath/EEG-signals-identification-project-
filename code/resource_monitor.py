import psutil
import torch
import logging

def get_gpu_memory_percent():
    """Return the percentage of GPU memory currently allocated.
    If no CUDA device is available, returns 0.
    """
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    return (allocated / total) * 100.0

def should_pause(max_gpu_percent: float, max_cpu_percent: float) -> bool:
    """Check if GPU or CPU usage exceeds given thresholds.
    Returns True if either resource exceeds its limit.
    """
    gpu_percent = get_gpu_memory_percent()
    cpu_percent = psutil.cpu_percent(interval=1)
    if gpu_percent > max_gpu_percent or cpu_percent > max_cpu_percent:
        logging.warning(f"Resource limits exceeded: GPU {gpu_percent:.2f}% > {max_gpu_percent}%, CPU {cpu_percent:.2f}% > {max_cpu_percent}%")
        return True
    return False

def log_resource_usage(max_gpu_percent: float, max_cpu_percent: float):
    """Log current GPU and CPU usage.
    This function is called each batch to record resource usage.
    """
    gpu_percent = get_gpu_memory_percent()
    cpu_percent = psutil.cpu_percent(interval=None)
    logging.info(f"Resource usage - GPU: {gpu_percent:.2f}% (limit {max_gpu_percent}%), CPU: {cpu_percent:.2f}% (limit {max_cpu_percent}%)")
