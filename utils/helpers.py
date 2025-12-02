import os
import time
import numpy as np
import tensorflow as tf
from typing import Any

def create_directory(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.set_random_seed(seed)

def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def count_parameters() -> int:
    """Count total trainable parameters"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_memory_usage() -> str:
    """Get current memory usage"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"{memory_info.rss / 1024 / 1024:.2f} MB"