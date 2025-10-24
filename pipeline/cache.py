"""
Caching system using joblib.Memory for expensive preprocessing and feature extraction operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import hashlib
import joblib
import pandas as pd
import numpy as np


def _create_cache_key(*args, **kwargs) -> str:
    """
    Create a deterministic cache key from function arguments.
    Handles pandas DataFrames, numpy arrays, and other common types.
    """
    def _hash_object(obj):
        if isinstance(obj, pd.DataFrame):
            # Use a combination of shape, dtypes, and a sample of data
            return f"df_{obj.shape}_{hashlib.md5(str(obj.dtypes).encode()).hexdigest()[:8]}_{hashlib.md5(str(obj.iloc[:min(100, len(obj))].values).encode()).hexdigest()[:8]}"
        elif isinstance(obj, np.ndarray):
            return f"array_{obj.shape}_{obj.dtype}_{hashlib.md5(obj.tobytes()).hexdigest()[:8]}"
        elif isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}_{len(obj)}_{hashlib.md5(str(obj).encode()).hexdigest()[:8]}"
        elif isinstance(obj, dict):
            # Sort keys for deterministic hashing
            sorted_items = sorted(obj.items())
            return f"dict_{len(obj)}_{hashlib.md5(str(sorted_items).encode()).hexdigest()[:8]}"
        else:
            return f"{type(obj).__name__}_{str(obj)}"
    
    # Combine all arguments into a single string
    key_parts = []
    
    # Add positional arguments
    for i, arg in enumerate(args):
        key_parts.append(f"arg_{i}_{_hash_object(arg)}")
    
    # Add keyword arguments (sorted for deterministic hashing)
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"kw_{key}_{_hash_object(value)}")
    
    # Create final hash
    full_key = "|".join(key_parts)
    return hashlib.md5(full_key.encode()).hexdigest()


def setup_memory(cache_dir: Path = None, verbose: int = 0) -> joblib.Memory:
    """
    Setup joblib.Memory with the specified cache directory.
    
    Args:
        cache_dir: Directory for cache storage. If None, uses "cache/" in project root.
        verbose: Verbosity level for joblib.Memory (0=silent, 1=info, 2=debug)
    
    Returns:
        Configured joblib.Memory instance
    """
    if cache_dir is None:
        # Use project root / cache
        project_root = Path(__file__).parent.parent
        cache_dir = project_root / "cache"
    
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Memory instance
    memory = joblib.Memory(location=str(cache_dir), verbose=verbose)
    
    print(f"Cache initialized at: {cache_dir}")
    return memory


def clear_cache(memory: joblib.Memory) -> None:
    """
    Clear all cached results.
    
    Args:
        memory: joblib.Memory instance to clear
    """
    memory.clear()
    print("Cache cleared successfully")


def get_cache_info(memory: joblib.Memory) -> Dict[str, Any]:
    """
    Get information about the current cache state.
    
    Args:
        memory: joblib.Memory instance
    
    Returns:
        Dictionary with cache information
    """
    cache_dir = Path(memory.location)
    
    if not cache_dir.exists():
        return {"status": "no_cache", "size_mb": 0, "n_files": 0}
    
    # Calculate cache size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    n_files = len(list(cache_dir.rglob('*')))
    
    return {
        "status": "active",
        "cache_dir": str(cache_dir),
        "size_mb": total_size / (1024 * 1024),
        "n_files": n_files
    }


# Global memory instance (will be initialized in main.py)
_memory = None


def get_memory() -> joblib.Memory:
    """
    Get the global memory instance, initializing it if necessary.
    
    Returns:
        Global joblib.Memory instance
    """
    global _memory
    if _memory is None:
        _memory = setup_memory()
    return _memory


def reset_memory() -> None:
    """
    Reset the global memory instance (useful for testing or when parameters change).
    """
    global _memory
    if _memory is not None:
        clear_cache(_memory)
    _memory = None
