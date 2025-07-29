"""
Custom exceptions for the mapping module.

These exceptions provide granular error handling for different failure modes
in geocoding, map fetching, caching, and rate limiting operations.
"""


class GeocodingError(Exception):
    """Raised when the Google Maps geocode API fails or returns no results."""
    pass


class MapFetchError(Exception):
    """Raised when fetching the static map image fails."""
    pass


class CacheError(Exception):
    """Raised on cache read/write failures."""
    pass


class RateLimitError(Exception):
    """Raised when rate limiting prevents API calls."""
    pass