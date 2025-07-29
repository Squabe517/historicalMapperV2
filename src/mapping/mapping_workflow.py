"""
High-level orchestrator for the mapping workflow.

Coordinates geocoding, caching, and map fetching for single places
or batches, providing a simple interface for the rest of the application.
"""

from typing import Any, Dict, List, Optional

from ..config.logger_module import log_info, log_warning, log_error
from .mapping_cache import ImageCacheManager
from .mapping_client import GoogleMapsClient
from .mapping_errors import CacheError, GeocodingError, MapFetchError, RateLimitError


class MappingOrchestrator:
    """
    Coordinates geocoding, caching, and fetching for a batch of places.
    
    This class provides a high-level interface that combines:
    - Google Maps API client for geocoding and map fetching
    - Cache manager for storing and retrieving map images
    - Error handling and logging
    """

    def __init__(self,
                 maps_client: GoogleMapsClient = None,
                 cache_manager: ImageCacheManager = None,
                 default_zoom: int = 12,
                 default_size: str = "600x400"):
        """
        Initialize the orchestrator.
        
        Args:
            maps_client: Google Maps client instance
            cache_manager: Cache manager instance
            default_zoom: Default zoom level for maps
            default_size: Default map size
        """
        self.client = maps_client or GoogleMapsClient()
        self.cache = cache_manager or ImageCacheManager()
        self.default_zoom = default_zoom
        self.default_size = default_size
        
        log_info(
            f"MappingOrchestrator initialized "
            f"(default_zoom={default_zoom}, default_size={default_size})"
        )

    def get_map_for_place(self,
                          place: str,
                          zoom: int = None,
                          size: str = None,
                          map_type: str = "roadmap") -> bytes:
        """
        Get map image for a single place, using cache when possible.
        
        Workflow:
        1. Check cache for existing image
        2. If not cached:
           a. Geocode the place name
           b. Fetch map image from Google Maps
           c. Cache the image
        3. Return image bytes
        
        Args:
            place: Place name to map
            zoom: Zoom level (uses default if None)
            size: Map size (uses default if None)
            map_type: Type of map (roadmap, satellite, hybrid, terrain)
            
        Returns:
            Raw map image bytes
            
        Raises:
            GeocodingError: If place cannot be geocoded
            MapFetchError: If map cannot be fetched
            CacheError: If caching fails (non-fatal, logged)
            RateLimitError: If rate limit exceeded
        """
        # Use defaults if not specified
        zoom = zoom or self.default_zoom
        size = size or self.default_size
        
        log_info(
            f"Getting map for '{place}' "
            f"(zoom={zoom}, size={size}, type={map_type})"
        )
        
        # Step 1: Check cache
        try:
            cached_data = self.cache.get_cached_bytes(place, zoom, size, map_type)
            if cached_data is not None:
                return cached_data
        except CacheError as e:
            # Cache errors are non-fatal, log and continue
            log_warning(f"Cache read error (continuing): {e}")
        
        # Step 2: Geocode place
        try:
            coords = self.client.geocode_place(place)
        except (GeocodingError, RateLimitError) as e:
            log_error(f"Failed to geocode '{place}': {e}")
            raise
        
        # Step 3: Fetch map image
        try:
            image_data = self.client.fetch_map_bytes(
                lat=coords["lat"],
                lng=coords["lng"],
                zoom=zoom,
                size=size,
                map_type=map_type
            )
        except (MapFetchError, RateLimitError) as e:
            log_error(
                f"Failed to fetch map for '{place}' at "
                f"({coords['lat']}, {coords['lng']}): {e}"
            )
            raise
        
        # Step 4: Cache the image (non-fatal if fails)
        try:
            cache_key = self.cache.cache_bytes(
                place, zoom, size, image_data, map_type
            )
            log_info(f"Cached map for '{place}' as {cache_key}")
        except CacheError as e:
            log_warning(f"Failed to cache map (continuing): {e}")
        
        return image_data

    def batch_get_maps(self,
                       places: List[Dict[str, Any]]) -> Dict[str, bytes]:
        """
        Process multiple places and return mapping from cache keys to bytes.
        
        Processes places sequentially to respect rate limiting naturally.
        Continues on individual failures, logging errors.
        
        Args:
            places: List of place dictionaries with keys:
                    - "place" (required): Place name
                    - "zoom" (optional): Zoom level
                    - "size" (optional): Map size
                    - "map_type" (optional): Map type
                    
        Returns:
            Dictionary mapping cache keys to raw image bytes
            
        Example:
            Input: [
                {"place": "Istanbul", "zoom": 10},
                {"place": "Venice", "zoom": 12, "map_type": "satellite"}
            ]
            Output: {
                "Istanbul_a1b2c3d4....png": b'<image data>',
                "Venice_e5f6g7h8....png": b'<image data>'
            }
        """
        if not places:
            log_warning("batch_get_maps called with empty places list")
            return {}
        
        log_info(f"Starting batch processing of {len(places)} places")
        
        results = {}
        success_count = 0
        error_count = 0
        
        for i, entry in enumerate(places, 1):
            # Validate entry
            if not isinstance(entry, dict) or "place" not in entry:
                log_warning(
                    f"Skipping invalid entry {i}: {entry}"
                )
                error_count += 1
                continue
            
            place = entry["place"]
            
            try:
                # Get map image
                image_data = self.get_map_for_place(
                    place=place,
                    zoom=entry.get("zoom", self.default_zoom),
                    size=entry.get("size", self.default_size),
                    map_type=entry.get("map_type", "roadmap")
                )
                
                # Generate cache key for consistency
                cache_key = self.cache._generate_cache_key(
                    place=place,
                    zoom=entry.get("zoom", self.default_zoom),
                    size=entry.get("size", self.default_size),
                    map_type=entry.get("map_type", "roadmap")
                )
                
                results[cache_key] = image_data
                success_count += 1
                
                log_info(
                    f"Processed {i}/{len(places)}: '{place}' -> {cache_key}"
                )
                
            except (GeocodingError, MapFetchError, RateLimitError) as e:
                log_error(
                    f"Failed to process place {i}/{len(places)} '{place}': {e}"
                )
                error_count += 1
                
                # Decide whether to continue or propagate based on error type
                if isinstance(e, RateLimitError):
                    # Rate limit errors might affect all subsequent requests
                    log_error("Rate limit reached, stopping batch processing")
                    break
                # Other errors are per-place, continue processing
            
            except Exception as e:
                # Unexpected errors
                log_error(
                    f"Unexpected error processing '{place}': {e}"
                )
                error_count += 1
        
        log_info(
            f"Batch processing complete: "
            f"{success_count} successful, {error_count} failed"
        )
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics from cache and rate limiter.
        
        Returns:
            Dictionary with cache and rate limit statistics
        """
        cache_stats = self.cache.get_cache_stats()
        rate_limit_stats = self.client.get_rate_limit_status()
        
        return {
            "cache": cache_stats,
            "rate_limit": rate_limit_stats
        }

    def validate_place_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Validate a place entry for batch processing.
        
        Args:
            entry: Place dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(entry, dict):
            return False
        
        if "place" not in entry or not entry["place"]:
            return False
        
        # Validate optional fields if present
        if "zoom" in entry:
            zoom = entry["zoom"]
            if not isinstance(zoom, int) or not (1 <= zoom <= 20):
                return False
        
        if "map_type" in entry:
            map_type = entry["map_type"]
            if map_type not in ["roadmap", "satellite", "hybrid", "terrain"]:
                return False
        
        return True

    def preprocess_places(self, places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess and validate place entries.
        
        Args:
            places: Raw place entries
            
        Returns:
            Validated and normalized place entries
        """
        processed = []
        
        for entry in places:
            if self.validate_place_entry(entry):
                # Normalize entry with defaults
                normalized = {
                    "place": entry["place"].strip(),
                    "zoom": entry.get("zoom", self.default_zoom),
                    "size": entry.get("size", self.default_size),
                    "map_type": entry.get("map_type", "roadmap")
                }
                processed.append(normalized)
            else:
                log_warning(f"Skipping invalid place entry: {entry}")
        
        return processed