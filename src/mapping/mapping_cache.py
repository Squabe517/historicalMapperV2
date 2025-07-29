"""
Image cache manager for storing map images with TTL and size management.

Provides a filesystem-based cache with automatic cleanup, collision-resistant
key generation, and robust error handling.
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config.config_module import get_config
from ..config.logger_module import log_info, log_warning, log_error
from .mapping_errors import CacheError


class ImageCacheManager:
    """
    Caches raw image bytes in a local directory with TTL, size management, 
    and robust key generation.
    
    Features:
    - MD5-based collision-resistant cache keys
    - Automatic TTL-based expiration
    - Size-based cleanup with LRU eviction
    - Cross-platform filesystem compatibility
    """

    def __init__(self,
                 cache_dir: str = None,
                 ttl_seconds: int = 86400,  # 24 hours
                 max_cache_size_mb: int = 100,
                 cleanup_threshold: float = 0.8):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live for cached images
            max_cache_size_mb: Maximum cache size in MB
            cleanup_threshold: Cleanup when cache reaches this fraction of max size
        """
        self.cache_dir = Path(
            cache_dir or get_config("CACHE_DIR", "./.cache_maps")
        )
        self.ttl = ttl_seconds
        self.max_size_bytes = max_cache_size_mb * 1024 * 1024
        self.cleanup_threshold = cleanup_threshold
        
        # Ensure cache directory exists
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            log_info(
                f"ImageCacheManager using {self.cache_dir} "
                f"(TTL={self.ttl}s, max_size={max_cache_size_mb}MB)"
            )
        except Exception as e:
            raise CacheError(f"Failed to create cache directory: {e}")

    def _generate_cache_key(self, 
                            place: str, 
                            zoom: int, 
                            size: str, 
                            map_type: str = "roadmap") -> str:
        """
        Generate a robust, collision-resistant cache key using MD5 hash.
        
        This approach ensures:
        1. No filesystem-unsafe characters
        2. Collision resistance via cryptographic hashing
        3. Fixed length filenames
        4. Cross-platform compatibility
        
        Args:
            place: Place name
            zoom: Zoom level
            size: Map size
            map_type: Type of map
            
        Returns:
            Safe cache key like "safe_place_name_<32-char-hash>.png"
        """
        # Combine all parameters with pipe separator
        content = f"{place}|{zoom}|{size}|{map_type}"
        
        # Generate MD5 hash for collision resistance
        hash_key = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Create safe prefix from place name (for debugging)
        # Keep only ASCII alphanumeric characters, dash, and dot
        # This ensures cross-platform compatibility
        safe_place = re.sub(r'[^a-zA-Z0-9\-.]', '_', place)[:20]
        
        # Construct final cache key
        cache_key = f"{safe_place}_{hash_key}.png"
        
        return cache_key

    def _cache_path(self, cache_key: str) -> Path:
        """Get full path for cache key."""
        return self.cache_dir / cache_key

    def _get_cache_size(self) -> int:
        """
        Calculate total cache size in bytes.
        
        Returns:
            Total size of all files in cache directory
        """
        total_size = 0
        try:
            for file_path in self.cache_dir.glob("*.png"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            log_warning(f"Error calculating cache size: {e}")
        
        return total_size

    def _get_cache_files_with_info(self) -> List[Tuple[Path, float, int]]:
        """
        Get all cache files with modification time and size.
        
        Returns:
            List of (path, mtime, size) tuples
        """
        files_info = []
        try:
            for file_path in self.cache_dir.glob("*.png"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append((file_path, stat.st_mtime, stat.st_size))
        except Exception as e:
            log_warning(f"Error scanning cache files: {e}")
        
        return files_info

    def _cleanup_cache(self) -> None:
        """
        Remove expired files and enforce size limits.
        
        Cleanup strategy:
        1. Remove all files older than TTL
        2. If still over threshold, remove oldest files (LRU)
        """
        log_info("Starting cache cleanup")
        
        current_time = time.time()
        files_info = self._get_cache_files_with_info()
        
        # Phase 1: Remove expired files
        removed_count = 0
        removed_size = 0
        
        for file_path, mtime, size in files_info:
            age = current_time - mtime
            if age > self.ttl:
                try:
                    file_path.unlink()
                    removed_count += 1
                    removed_size += size
                    log_info(f"Removed expired file: {file_path.name}")
                except Exception as e:
                    log_warning(f"Failed to remove expired file {file_path}: {e}")
        
        if removed_count > 0:
            log_info(
                f"Removed {removed_count} expired files "
                f"({removed_size / 1024 / 1024:.1f} MB)"
            )
        
        # Phase 2: Size-based cleanup if needed
        current_size = self._get_cache_size()
        threshold_size = self.max_size_bytes * self.cleanup_threshold
        
        if current_size > threshold_size:
            # Get remaining files sorted by modification time (oldest first)
            remaining_files = [
                (p, m, s) for p, m, s in self._get_cache_files_with_info()
            ]
            remaining_files.sort(key=lambda x: x[1])  # Sort by mtime
            
            # Remove oldest files until under threshold
            for file_path, mtime, size in remaining_files:
                if current_size <= threshold_size:
                    break
                
                try:
                    file_path.unlink()
                    current_size -= size
                    log_info(f"Removed for size limit: {file_path.name}")
                except Exception as e:
                    log_warning(f"Failed to remove file {file_path}: {e}")
        
        log_info(
            f"Cache cleanup complete. Current size: "
            f"{current_size / 1024 / 1024:.1f} MB"
        )

    def cache_bytes(self, 
                    place: str, 
                    zoom: int, 
                    size: str, 
                    data: bytes, 
                    map_type: str = "roadmap") -> str:
        """
        Write bytes to disk using robust cache key.
        
        Args:
            place: Place name
            zoom: Zoom level
            size: Map size
            data: Raw image bytes
            map_type: Type of map
            
        Returns:
            The cache key used
            
        Raises:
            CacheError: On write failures
        """
        if not data:
            raise CacheError("Cannot cache empty data")
        
        # Check if cleanup needed
        current_size = self._get_cache_size()
        threshold_size = self.max_size_bytes * self.cleanup_threshold
        
        if current_size > threshold_size:
            self._cleanup_cache()
        
        # Generate cache key
        cache_key = self._generate_cache_key(place, zoom, size, map_type)
        cache_path = self._cache_path(cache_key)
        
        try:
            # Write data to file
            cache_path.write_bytes(data)
            
            log_info(
                f"Cached {len(data)} bytes for '{place}' as {cache_key}"
            )
            
            return cache_key
            
        except Exception as e:
            log_error(f"Failed to cache data: {e}")
            raise CacheError(f"Failed to write cache file: {e}")

    def get_cached_bytes(self, 
                         place: str, 
                         zoom: int, 
                         size: str, 
                         map_type: str = "roadmap") -> Optional[bytes]:
        """
        Retrieve cached bytes if available and not expired.
        
        Args:
            place: Place name
            zoom: Zoom level
            size: Map size
            map_type: Type of map
            
        Returns:
            Cached bytes if available and valid, None otherwise
            
        Raises:
            CacheError: On read failures
        """
        cache_key = self._generate_cache_key(place, zoom, size, map_type)
        cache_path = self._cache_path(cache_key)
        
        if not cache_path.exists():
            log_info(f"Cache miss: {cache_key} not found")
            return None
        
        try:
            # Check file age
            stat = cache_path.stat()
            age = time.time() - stat.st_mtime
            
            if age > self.ttl:
                # File expired, remove it
                log_info(f"Cache expired: {cache_key} ({age:.0f}s old)")
                try:
                    cache_path.unlink()
                except Exception:
                    pass  # Ignore removal errors
                return None
            
            # Read and return cached data
            data = cache_path.read_bytes()
            log_info(
                f"Cache hit: {cache_key} ({len(data)} bytes, {age:.0f}s old)"
            )
            return data
            
        except Exception as e:
            log_error(f"Failed to read cache file {cache_key}: {e}")
            raise CacheError(f"Failed to read cache file: {e}")

    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        files_info = self._get_cache_files_with_info()
        total_size = sum(size for _, _, size in files_info)
        
        return {
            "total_files": len(files_info),
            "total_size_mb": total_size / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
            "usage_percent": (total_size / self.max_size_bytes * 100) 
                           if self.max_size_bytes > 0 else 0.0
        }

    def clear_cache(self) -> None:
        """
        Clear all cached files.
        
        Raises:
            CacheError: On deletion failures
        """
        log_info("Clearing entire cache")
        
        try:
            removed_count = 0
            for file_path in self.cache_dir.glob("*.png"):
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1
            
            log_info(f"Cleared {removed_count} files from cache")
            
        except Exception as e:
            log_error(f"Failed to clear cache: {e}")
            raise CacheError(f"Failed to clear cache: {e}")