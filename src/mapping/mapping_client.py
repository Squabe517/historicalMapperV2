"""
Google Maps API client with rate limiting.

Wraps the googlemaps SDK to provide geocoding and static map fetching
with built-in rate limiting and error handling.
"""

from typing import Dict, Optional
import googlemaps
import requests
from urllib.parse import urlencode

from ..config.config_module import get_config
from ..config.logger_module import log_info, log_error
from .mapping_errors import GeocodingError, MapFetchError, RateLimitError
from .mapping_rate_limiter import TokenBucketRateLimiter


class GoogleMapsClient:
    """
    Wraps googlemaps SDK to geocode and retrieve static map images with rate limiting.
    
    Provides two main functions:
    1. Geocoding place names to coordinates
    2. Fetching static map images for coordinates
    """

    # Static Maps API base URL
    STATIC_MAPS_BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

    def __init__(self,
                 api_key: str = None,
                 rate_limit_per_sec: float = 5.0,
                 burst_capacity: int = 10,
                 request_timeout: int = 30):
        """
        Initialize the Google Maps client.
        
        Args:
            api_key: Google Maps API key (loaded from config if not provided)
            rate_limit_per_sec: Throttle outgoing requests
            burst_capacity: Max burst requests allowed
            request_timeout: HTTP request timeout in seconds
        """
        # Get API key from config if not provided
        self.api_key = api_key or get_config("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not provided or found in config")
        
        self.request_timeout = request_timeout
        
        # Initialize rate limiter
        self._rate_limiter = TokenBucketRateLimiter(
            rate_per_second=rate_limit_per_sec,
            burst_capacity=burst_capacity
        )
        
        # Initialize Google Maps client
        self._gmaps = googlemaps.Client(key=self.api_key)
        
        # Initialize requests session for static maps
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'HistoricalEpubMapEnhancer/1.0'
        })
        
        log_info(
            f"GoogleMapsClient initialized (rate_limit={rate_limit_per_sec}/sec)"
        )

    def geocode_place(self, place: str) -> Dict[str, float]:
        """
        Geocode a place name to coordinates.
        
        Args:
            place: Place name to geocode
            
        Returns:
            Dictionary with 'lat' and 'lng' keys
            
        Raises:
            GeocodingError: On failure or empty results
            RateLimitError: If rate limiting fails
        """
        if not place or not place.strip():
            raise GeocodingError("Empty place name provided")
        
        # Apply rate limiting
        try:
            self._rate_limiter.wait_for_token()
        except RateLimitError as e:
            log_error(f"Rate limit exceeded for geocoding: {e}")
            raise
        
        try:
            log_info(f"Geocoding place: {place}")
            
            # Call Google Maps geocoding API
            results = self._gmaps.geocode(place)
            
            if not results:
                raise GeocodingError(f"No coordinates found for '{place}'")
            
            # Extract coordinates from first result
            location = results[0]['geometry']['location']
            lat = location['lat']
            lng = location['lng']
            
            log_info(f"Geocoded '{place}' to ({lat}, {lng})")
            
            return {"lat": lat, "lng": lng}
            
        except googlemaps.exceptions.ApiError as e:
            log_error(f"Google Maps API error for '{place}': {e}")
            raise GeocodingError(f"API error geocoding '{place}': {str(e)}")
        except googlemaps.exceptions.Timeout as e:
            log_error(f"Timeout geocoding '{place}': {e}")
            raise GeocodingError(f"Timeout geocoding '{place}'")
        except Exception as e:
            log_error(f"Unexpected error geocoding '{place}': {e}")
            raise GeocodingError(f"Failed to geocode '{place}': {str(e)}")

    def fetch_map_bytes(self,
                        lat: float,
                        lng: float,
                        zoom: int,
                        size: str = "600x400",
                        map_type: str = "roadmap") -> bytes:
        """
        Fetch static map image as raw bytes.
        
        Args:
            lat: Latitude
            lng: Longitude
            zoom: Zoom level (1-20)
            size: Image size (e.g., "600x400")
            map_type: Map type (roadmap, satellite, hybrid, terrain)
            
        Returns:
            Raw PNG/JPEG bytes
            
        Raises:
            MapFetchError: On HTTP errors or empty response
            RateLimitError: If rate limiting fails
        """
        # Validate inputs
        if not (-90 <= lat <= 90):
            raise MapFetchError(f"Invalid latitude: {lat}")
        if not (-180 <= lng <= 180):
            raise MapFetchError(f"Invalid longitude: {lng}")
        if not (1 <= zoom <= 20):
            raise MapFetchError(f"Invalid zoom level: {zoom}")
        
        # Apply rate limiting
        try:
            self._rate_limiter.wait_for_token()
        except RateLimitError as e:
            log_error(f"Rate limit exceeded for map fetch: {e}")
            raise
        
        # Build request parameters
        params = {
            "center": f"{lat},{lng}",
            "zoom": str(zoom),
            "size": size,
            "maptype": map_type,
            "key": self.api_key,
            "format": "png"
        }
        
        try:
            log_info(
                f"Fetching map for ({lat}, {lng}) at zoom {zoom}"
            )
            
            # Make request
            response = self._session.get(
                self.STATIC_MAPS_BASE_URL,
                params=params,
                timeout=self.request_timeout
            )
            
            # Check response status
            if response.status_code != 200:
                log_error(
                    f"HTTP {response.status_code} fetching map: "
                    f"{response.text[:200]}"
                )
                raise MapFetchError(
                    f"HTTP {response.status_code} fetching map"
                )
            
            # Check for valid image data
            content = response.content
            if not content:
                raise MapFetchError("Empty response from Static Maps API")
            
            # Verify it's an image (basic check)
            if len(content) < 100:  # Minimum reasonable image size
                raise MapFetchError(
                    f"Response too small ({len(content)} bytes), "
                    "likely an error message"
                )
            
            log_info(
                f"Successfully fetched map ({len(content)} bytes)"
            )
            
            return content
            
        except requests.exceptions.Timeout:
            log_error(f"Timeout fetching map for ({lat}, {lng})")
            raise MapFetchError("Request timeout")
        except requests.exceptions.RequestException as e:
            log_error(f"Request error fetching map: {e}")
            raise MapFetchError(f"Request failed: {str(e)}")
        except Exception as e:
            log_error(f"Unexpected error fetching map: {e}")
            raise MapFetchError(f"Failed to fetch map: {str(e)}")

    def build_static_map_url(self,
                             lat: float,
                             lng: float,
                             zoom: int,
                             size: str = "600x400",
                             map_type: str = "roadmap") -> str:
        """
        Build a Google Static Maps URL (for debugging/logging).
        
        Args:
            lat: Latitude
            lng: Longitude
            zoom: Zoom level
            size: Image size
            map_type: Map type
            
        Returns:
            Complete URL for the static map
        """
        params = {
            "center": f"{lat},{lng}",
            "zoom": str(zoom),
            "size": size,
            "maptype": map_type,
            "key": self.api_key,
            "format": "png"
        }
        
        return f"{self.STATIC_MAPS_BASE_URL}?{urlencode(params)}"

    def get_rate_limit_status(self) -> Dict[str, float]:
        """
        Get current rate limiting status.
        
        Returns:
            Dictionary with available tokens and wait time
        """
        return {
            "available_tokens": self._rate_limiter.get_available_tokens(),
            "wait_time_seconds": self._rate_limiter.get_wait_time()
        }