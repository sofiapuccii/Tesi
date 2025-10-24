"""
Cache management utilities for handling cache invalidation when parameters change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
from .cache import get_memory, clear_cache, get_cache_info


class CacheManager:
    """
    Manages cache invalidation based on parameter changes.
    """
    
    def __init__(self, cache_params_file: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_params_file: Path to file storing cached parameters. 
                              If None, uses "cache/cached_params.json"
        """
        if cache_params_file is None:
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            cache_params_file = cache_dir / "cached_params.json"
        
        self.cache_params_file = cache_params_file
        self.memory = get_memory()
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """
        Create a hash of the parameters for comparison.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Hash string of the parameters
        """
        # Sort parameters for consistent hashing
        sorted_params = {k: v for k, v in sorted(params.items())}
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _load_cached_params(self) -> Optional[Dict[str, Any]]:
        """
        Load previously cached parameters.
        
        Returns:
            Dictionary of cached parameters or None if no cache exists
        """
        if not self.cache_params_file.exists():
            return None
        
        try:
            with open(self.cache_params_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def _save_cached_params(self, params: Dict[str, Any]) -> None:
        """
        Save current parameters to cache.
        
        Args:
            params: Dictionary of parameters to save
        """
        with open(self.cache_params_file, 'w') as f:
            json.dump(params, f, indent=2)
    
    def check_and_invalidate(self, current_params: Dict[str, Any], 
                           force_clear: bool = False) -> bool:
        """
        Check if parameters have changed and invalidate cache if necessary.
        
        Args:
            current_params: Current parameters to check
            force_clear: If True, clear cache regardless of parameter changes
            
        Returns:
            True if cache was cleared, False otherwise
        """
        if force_clear:
            clear_cache(self.memory)
            self._save_cached_params(current_params)
            print("Cache cleared (forced)")
            return True
        
        # Load previous parameters
        cached_params = self._load_cached_params()
        
        if cached_params is None:
            # No previous cache, save current parameters
            self._save_cached_params(current_params)
            print("No previous cache found, parameters saved")
            return False
        
        # Compare parameter hashes
        current_hash = self._hash_params(current_params)
        cached_hash = self._hash_params(cached_params)
        
        if current_hash != cached_hash:
            # Parameters changed, clear cache
            clear_cache(self.memory)
            self._save_cached_params(current_params)
            print("Parameters changed, cache cleared")
            print(f"Previous: {cached_params}")
            print(f"Current:  {current_params}")
            return True
        else:
            print("Parameters unchanged, using existing cache")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = get_cache_info(self.memory)
        cached_params = self._load_cached_params()
        
        return {
            **cache_info,
            "cached_params": cached_params,
            "params_file": str(self.cache_params_file)
        }


def create_preprocessing_params(window_size: int, overlap: float, 
                              sampling_rate: int, data_dir: str) -> Dict[str, Any]:
    """
    Create a standardized parameter dictionary for preprocessing operations.
    
    Args:
        window_size: Window size parameter
        overlap: Overlap parameter
        sampling_rate: Sampling rate parameter
        data_dir: Data directory path
        
    Returns:
        Dictionary of preprocessing parameters
    """
    return {
        "window_size": window_size,
        "overlap": overlap,
        "sampling_rate": sampling_rate,
        "data_dir": str(Path(data_dir).resolve()),
        "preprocessing_version": "1.0"  # Version for future compatibility
    }


def create_feature_params(sampling_rate: int) -> Dict[str, Any]:
    """
    Create a standardized parameter dictionary for feature extraction operations.
    
    Args:
        sampling_rate: Sampling rate parameter
        
    Returns:
        Dictionary of feature extraction parameters
    """
    return {
        "sampling_rate": sampling_rate,
        "feature_extraction_version": "1.0"  # Version for future compatibility
    }
