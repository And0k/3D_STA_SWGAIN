import os
import psutil
import time
import logging
import threading
import torch
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)

def setup_memory_logging(file_path: str = '/content/drive/MyDrive/3D_STA_SWGAIN/memory_debug.log'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(file_path, mode='a'),
            logging.StreamHandler()
        ],
        force=True
    )

def get_mem_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def log_memory_diff(stage_name: str):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            before = get_mem_usage()
            logger.debug(f"[MEM] Stage '{stage_name}' - Start. Current RSS: {before:.2f} MB")
            result = func(*args, **kwargs)
            after = get_mem_usage()
            diff = after - before
            logger.debug(f"[MEM] Stage '{stage_name}' - End. RSS: {after:.2f} MB, Diff: {diff:+.2f} MB")
            return result
        return wrapper
    return decorator

def memory_monitor_loop(threshold_mb: int = 400, interval: int = 1):
    process = psutil.Process(os.getpid())
    thread = threading.current_thread()
    while getattr(thread, 'do_run', True):
        try:
            avail = psutil.virtual_memory().available / (1024 ** 2)
            if avail < threshold_mb:
                logger.error(f"EMERGENCY: Available {avail:.1f}MB < {threshold_mb}MB. RSS: {get_mem_usage():.1f}MB")
                os._exit(1)
            time.sleep(interval)
        except Exception:
            break
