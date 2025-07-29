"""
Mapping module for the Historical ePub Map Enhancer.

This module provides functionality for:
- Geocoding place names using Google Maps API
- Fetching static map images for coordinates
- Caching map images with TTL and size management
- Rate limiting API requests
- Orchestrating the complete mapping workflow

Main classes:
- MappingOrchestrator: High-level interface for mapping operations
- GoogleMapsClient: Google Maps API wrapper with rate limiting
- ImageCacheManager: Local cache for map images
- TokenBucketRateLimiter: Rate limiting implementation

Errors:
- GeocodingError: Geocoding failures
- MapFetchError: Map fetching failures
- CacheError: Cache operation failures
- RateLimitError: Rate limit exceeded
"""

from .mapping_cache import ImageCacheManager
from .mapping_client import GoogleMapsClient
from .mapping_errors import CacheError, GeocodingError, MapFetchError, RateLimitError
from .mapping_rate_limiter import TokenBucketRateLimiter
from .mapping_workflow import MappingOrchestrator

__all__ = [
    # Main classes
    "MappingOrchestrator",
    "GoogleMapsClient",
    "ImageCacheManager",
    "TokenBucketRateLimiter",
    
    # Errors
    "GeocodingError",
    "MapFetchError",
    "CacheError",
    "RateLimitError",
]

# Version info
__version__ = "1.0.0"
__author__ = "Historical ePub Map Enhancer Team"
