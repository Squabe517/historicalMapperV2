"""
Comprehensive test suite for the mapping module.

Tests all components including rate limiting, geocoding, caching,
and the orchestration workflow with proper mocking of external services.

To run tests:
- In VSCode: Use the Testing sidebar or click "Run Test" above test methods
- Command line: python -m pytest src/mapping/test_mapping.py -v
- Direct execution: python src/mapping/test_mapping.py
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Import all mapping components (relative imports since we're in the module)
from .mapping_errors import (
    GeocodingError,
    MapFetchError,
    CacheError,
    RateLimitError
)
from .mapping_rate_limiter import TokenBucketRateLimiter
from .mapping_client import GoogleMapsClient
from .mapping_cache import ImageCacheManager
from .mapping_workflow import MappingOrchestrator


# ==================== FIXTURES ====================

@pytest.fixture(autouse=True)
def mock_logging():
    """Mock all logging functions to prevent actual logging during tests."""
    # Mock all logging functions in all mapping modules
    with patch('src.mapping.mapping_rate_limiter.log_info') as mock1:
        with patch('src.mapping.mapping_client.log_info') as mock2:
            with patch('src.mapping.mapping_client.log_error') as mock3:
                with patch('src.mapping.mapping_cache.log_info') as mock4:
                    with patch('src.mapping.mapping_cache.log_warning') as mock5:
                        with patch('src.mapping.mapping_cache.log_error') as mock6:
                            with patch('src.mapping.mapping_workflow.log_info') as mock7:
                                with patch('src.mapping.mapping_workflow.log_warning') as mock8:
                                    with patch('src.mapping.mapping_workflow.log_error') as mock9:
                                        yield

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp()
    cache_dir = Path(temp_dir) / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    yield cache_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Mock config module functions for tests."""
    with patch('src.mapping.mapping_client.get_config') as mock_get:
        mock_get.return_value = "test_api_key"
        yield mock_get


@pytest.fixture
def mock_logger():
    """Mock logger module functions for tests."""
    with patch('src.mapping.mapping_client.log_info') as mock_info:
        with patch('src.mapping.mapping_client.log_error') as mock_error:
            with patch('src.mapping.mapping_client.log_warning') as mock_warning:
                yield {
                    'log_info': mock_info,
                    'log_error': mock_error,
                    'log_warning': mock_warning
                }


# ==================== TEST CLASSES ====================

class TestErrors:
    """Test custom exceptions."""
    
    def test_geocoding_error(self):
        """Test GeocodingError creation and message."""
        error = GeocodingError("Place not found")
        assert str(error) == "Place not found"
        assert isinstance(error, Exception)
    
    def test_map_fetch_error(self):
        """Test MapFetchError creation and message."""
        error = MapFetchError("HTTP 404")
        assert str(error) == "HTTP 404"
        assert isinstance(error, Exception)
    
    def test_cache_error(self):
        """Test CacheError creation and message."""
        error = CacheError("Write failed")
        assert str(error) == "Write failed"
        assert isinstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test RateLimitError creation and message."""
        error = RateLimitError("Quota exceeded")
        assert str(error) == "Quota exceeded"
        assert isinstance(error, Exception)


class TestTokenBucketRateLimiter:
    """Test the rate limiter implementation."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = TokenBucketRateLimiter(
            rate_per_second=2.0,
            burst_capacity=5,
            retry_attempts=3,
            backoff_factor=2.0
        )
        
        assert limiter.rate_per_second == 2.0
        assert limiter.burst_capacity == 5
        assert limiter.tokens == 5.0  # Starts full
        assert limiter.retry_attempts == 3
        assert limiter.backoff_factor == 2.0
    
    def test_acquire_success(self):
        """Test successful token acquisition."""
        limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_capacity=5)
        
        # Should succeed with available tokens
        assert limiter.acquire(1) is True
        assert limiter.tokens == 4.0
        
        # Multiple acquisitions
        assert limiter.acquire(2) is True
        assert limiter.tokens == 2.0
    
    def test_acquire_failure(self):
        """Test failed token acquisition."""
        limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_capacity=3)
        
        # Exhaust tokens
        assert limiter.acquire(3) is True
        assert limiter.tokens == 0.0
        
        # Should fail
        assert limiter.acquire(1) is False
    
    @patch('src.mapping.mapping_rate_limiter.time.time')
    def test_token_refill(self, mock_time):
        """Test token refilling over time."""
        # Start at time 0
        mock_time.return_value = 0.0
        
        limiter = TokenBucketRateLimiter(rate_per_second=2.0, burst_capacity=5)
        
        # Use all tokens
        limiter.acquire(5)
        assert limiter.tokens == 0.0
        
        # Simulate 0.5 seconds passing (should add 1 token)
        mock_time.return_value = 0.5
        
        # Check refill
        available = limiter.get_available_tokens()
        assert available >= 0.9  # Allow small variance
        assert available <= 1.1
    
    @patch('src.mapping.mapping_rate_limiter.time.time')
    @patch('src.mapping.mapping_rate_limiter.time.sleep')
    def test_wait_for_token(self, mock_sleep, mock_time):
        """Test blocking wait for tokens."""
        # Setup time simulation
        current_time = [0.0]
        
        def time_side_effect():
            return current_time[0]
        
        def sleep_side_effect(duration):
            current_time[0] += duration
        
        mock_time.side_effect = time_side_effect
        mock_sleep.side_effect = sleep_side_effect
        
        limiter = TokenBucketRateLimiter(
            rate_per_second=10.0,  # Fast refill for testing
            burst_capacity=2,
            retry_attempts=3
        )
        
        # Use all tokens
        limiter.acquire(2)
        
        # Should wait and succeed
        limiter.wait_for_token(1)
        
        # Verify sleep was called (waited for token)
        assert mock_sleep.called
        
        # Should have waited approximately 0.1 seconds
        total_sleep = sum(call[0][0] for call in mock_sleep.call_args_list)
        assert total_sleep >= 0.05
        assert total_sleep <= 0.3
    
    def test_wait_for_token_timeout(self):
        """Test rate limit error after max retries."""
        limiter = TokenBucketRateLimiter(
            rate_per_second=0.1,  # Very slow refill
            burst_capacity=1,
            retry_attempts=2,
            backoff_factor=1.0  # No backoff for predictable test
        )
        
        # Use all tokens
        limiter.acquire(1)
        
        # Should fail after retries
        with pytest.raises(RateLimitError) as exc_info:
            limiter.wait_for_token(10)  # Request many tokens
        
        assert "Failed to acquire" in str(exc_info.value)
    
    def test_get_wait_time(self):
        """Test wait time calculation."""
        limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_capacity=5)
        
        # No wait needed with tokens available
        assert limiter.get_wait_time(1) == 0.0
        
        # Use all tokens
        limiter.acquire(5)
        
        # Should need to wait
        wait_time = limiter.get_wait_time(2)
        assert wait_time >= 1.9  # ~2 seconds for 2 tokens at 1/sec
        assert wait_time <= 2.1


class TestGoogleMapsClient:
    """Test the Google Maps API client."""
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_initialization(self, mock_get_config, mock_gmaps):
        """Test client initialization."""
        mock_get_config.return_value = "test_api_key"
        
        client = GoogleMapsClient(
            rate_limit_per_sec=10.0,
            burst_capacity=20,
            request_timeout=60
        )
        
        assert client.api_key == "test_api_key"
        assert client.request_timeout == 60
        mock_gmaps.assert_called_once_with(key="test_api_key")
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    def test_initialization_with_api_key(self, mock_gmaps):
        """Test initialization with provided API key."""
        client = GoogleMapsClient(api_key="provided_key")
        assert client.api_key == "provided_key"
        mock_gmaps.assert_called_once_with(key="provided_key")
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_geocode_place_success(self, mock_get_config, mock_gmaps):
        """Test successful geocoding."""
        mock_get_config.return_value = "test_key"
        
        # Mock geocoding response
        mock_gmaps_instance = MagicMock()
        mock_gmaps.return_value = mock_gmaps_instance
        mock_gmaps_instance.geocode.return_value = [{
            'geometry': {
                'location': {
                    'lat': 41.0082,
                    'lng': 28.9784
                }
            }
        }]
        
        client = GoogleMapsClient()
        result = client.geocode_place("Istanbul")
        
        assert result == {"lat": 41.0082, "lng": 28.9784}
        mock_gmaps_instance.geocode.assert_called_once_with("Istanbul")
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_geocode_place_no_results(self, mock_get_config, mock_gmaps):
        """Test geocoding with no results."""
        mock_get_config.return_value = "test_key"
        
        mock_gmaps_instance = MagicMock()
        mock_gmaps.return_value = mock_gmaps_instance
        mock_gmaps_instance.geocode.return_value = []
        
        client = GoogleMapsClient()
        
        with pytest.raises(GeocodingError) as exc_info:
            client.geocode_place("Nonexistent Place")
        
        assert "No coordinates found" in str(exc_info.value)
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_geocode_empty_place(self, mock_get_config, mock_gmaps):
        """Test geocoding with empty place name."""
        mock_get_config.return_value = "test_key"
        
        client = GoogleMapsClient()
        
        with pytest.raises(GeocodingError) as exc_info:
            client.geocode_place("")
        
        assert "Empty place name" in str(exc_info.value)
    
    @patch('src.mapping.mapping_client.requests.Session')
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_fetch_map_bytes_success(self, mock_get_config, mock_gmaps, mock_session):
        """Test successful map fetching."""
        mock_get_config.return_value = "test_key"
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"PNG_IMAGE_DATA" * 100  # Fake image data
        
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        
        client = GoogleMapsClient()
        result = client.fetch_map_bytes(
            lat=41.0082,
            lng=28.9784,
            zoom=12,
            size="600x400",
            map_type="roadmap"
        )
        
        assert result == mock_response.content
        assert len(result) > 100
        
        # Verify request parameters
        call_args = mock_session_instance.get.call_args
        assert call_args[0][0] == client.STATIC_MAPS_BASE_URL
        params = call_args[1]['params']
        assert params['center'] == "41.0082,28.9784"
        assert params['zoom'] == "12"
        assert params['size'] == "600x400"
    
    @patch('src.mapping.mapping_client.requests.Session')
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_fetch_map_bytes_http_error(self, mock_get_config, mock_gmaps, mock_session):
        """Test map fetching with HTTP error."""
        mock_get_config.return_value = "test_key"
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        
        client = GoogleMapsClient()
        
        with pytest.raises(MapFetchError) as exc_info:
            client.fetch_map_bytes(40.7128, -74.0060, 10)
        
        assert "HTTP 404" in str(exc_info.value)
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.get_config')
    def test_invalid_coordinates(self, mock_get_config, mock_gmaps):
        """Test fetching with invalid coordinates."""
        mock_get_config.return_value = "test_key"
        
        client = GoogleMapsClient()
        
        # Invalid latitude
        with pytest.raises(MapFetchError) as exc_info:
            client.fetch_map_bytes(lat=91.0, lng=0.0, zoom=10)
        assert "Invalid latitude" in str(exc_info.value)
        
        # Invalid longitude
        with pytest.raises(MapFetchError) as exc_info:
            client.fetch_map_bytes(lat=0.0, lng=181.0, zoom=10)
        assert "Invalid longitude" in str(exc_info.value)
        
        # Invalid zoom
        with pytest.raises(MapFetchError) as exc_info:
            client.fetch_map_bytes(lat=0.0, lng=0.0, zoom=25)
        assert "Invalid zoom" in str(exc_info.value)


class TestImageCacheManager:
    """Test the image cache manager."""
    
    @patch('src.mapping.mapping_cache.get_config')
    def test_initialization(self, mock_get_config, temp_cache_dir):
        """Test cache manager initialization."""
        mock_get_config.return_value = str(temp_cache_dir)
        
        cache = ImageCacheManager(
            ttl_seconds=3600,
            max_cache_size_mb=50,
            cleanup_threshold=0.9
        )
        
        assert cache.cache_dir == temp_cache_dir
        assert cache.ttl == 3600
        assert cache.max_size_bytes == 50 * 1024 * 1024
        assert cache.cleanup_threshold == 0.9
        assert temp_cache_dir.exists()
    
    def test_generate_cache_key(self, temp_cache_dir):
        """Test cache key generation."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir))
        
        # Test basic key generation
        key1 = cache._generate_cache_key("Istanbul", 12, "600x400", "roadmap")
        assert key1.endswith(".png")
        assert "Istanbul" in key1
        assert len(key1) > 20  # Has hash component
        
        # Test different parameters produce different keys
        key2 = cache._generate_cache_key("Istanbul", 10, "600x400", "roadmap")
        assert key1 != key2
        
        # Test special characters handling
        key3 = cache._generate_cache_key("São Paulo, Brazil", 12, "600x400", "roadmap")
        # Now uses ASCII-only pattern, so ã becomes _
        assert "S_o_Paulo__Brazil" in key3
        assert key3.endswith(".png")
        
        # Test consistency
        key4 = cache._generate_cache_key("Istanbul", 12, "600x400", "roadmap")
        assert key1 == key4
        
        # Test with more special characters
        key5 = cache._generate_cache_key("New York, NY (USA)", 10, "800x600", "satellite")
        # Parentheses, spaces, and commas should be replaced with underscores
        assert key5.startswith("New_York__NY__USA_")
        assert key5.endswith(".png")
        # Verify no unsafe characters remain
        assert not any(char in key5 for char in ['(', ')', ' ', ',', 'ã'])
        
        # Test hash uniqueness
        key6 = cache._generate_cache_key("London", 10, "600x400", "roadmap")
        key7 = cache._generate_cache_key("London", 10, "600x400", "satellite")
        # Same place and params but different map type should produce different keys
        assert key6 != key7
        # But the prefix should be the same
        assert key6.split('_')[0] == key7.split('_')[0] == "London"
    
    def test_cache_and_retrieve_bytes(self, temp_cache_dir):
        """Test caching and retrieving bytes."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir), ttl_seconds=3600)
        
        test_data = b"TEST_IMAGE_DATA_12345"
        
        # Cache data
        cache_key = cache.cache_bytes("Venice", 12, "600x400", test_data)
        assert cache_key.endswith(".png")
        
        # Retrieve data
        retrieved = cache.get_cached_bytes("Venice", 12, "600x400")
        assert retrieved == test_data
    
    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss for non-existent data."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir))
        
        result = cache.get_cached_bytes("Nonexistent", 10, "600x400")
        assert result is None
    
    def test_cache_expiration(self, temp_cache_dir):
        """Test TTL expiration by manipulating file modification time."""
        import os
        
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir), ttl_seconds=10)
        
        # Cache data
        test_data = b"EXPIRING_DATA"
        cache_key = cache.cache_bytes("London", 10, "600x400", test_data)
        
        # Verify it can be read initially
        assert cache.get_cached_bytes("London", 10, "600x400") == test_data
        
        # Get the actual file path and set its modification time to the past
        file_path = cache._cache_path(cache_key)
        current_time = time.time()
        old_time = current_time - 20  # 20 seconds ago (older than 10 second TTL)
        os.utime(file_path, (old_time, old_time))
        
        # Now the file should be considered expired
        result = cache.get_cached_bytes("London", 10, "600x400")
        assert result is None
        
        # Verify the expired file was deleted
        assert not file_path.exists()
    
    def test_cache_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir), max_cache_size_mb=1)
        
        # Empty cache stats
        stats = cache.get_cache_stats()
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["usage_percent"] == 0.0
        
        # Add some data
        data1 = b"X" * 1000
        data2 = b"Y" * 2000
        cache.cache_bytes("Place1", 10, "600x400", data1)
        cache.cache_bytes("Place2", 12, "600x400", data2)
        
        # Check updated stats
        stats = cache.get_cache_stats()
        assert stats["total_files"] == 2
        assert stats["total_size_mb"] > 0
        assert stats["usage_percent"] > 0
    
    def test_clear_cache(self, temp_cache_dir):
        """Test clearing entire cache."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir))
        
        # Add some files
        cache.cache_bytes("Place1", 10, "600x400", b"DATA1")
        cache.cache_bytes("Place2", 12, "600x400", b"DATA2")
        
        stats = cache.get_cache_stats()
        assert stats["total_files"] == 2
        
        # Clear cache
        cache.clear_cache()
        
        stats = cache.get_cache_stats()
        assert stats["total_files"] == 0
    
    def test_empty_data_error(self, temp_cache_dir):
        """Test caching empty data raises error."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir))
        
        with pytest.raises(CacheError) as exc_info:
            cache.cache_bytes("Place", 10, "600x400", b"")
        
        assert "Cannot cache empty data" in str(exc_info.value)
    
    @patch('src.mapping.mapping_cache.Path.write_bytes')
    def test_cache_write_error(self, mock_write, temp_cache_dir):
        """Test handling of write errors."""
        cache = ImageCacheManager(cache_dir=str(temp_cache_dir))
        
        mock_write.side_effect = IOError("Disk full")
        
        with pytest.raises(CacheError) as exc_info:
            cache.cache_bytes("Place", 10, "600x400", b"DATA")
        
        assert "Failed to write cache file" in str(exc_info.value)


class TestMappingOrchestrator:
    """Test the mapping orchestrator."""
    
    @patch('src.mapping.mapping_workflow.GoogleMapsClient')
    @patch('src.mapping.mapping_workflow.ImageCacheManager')
    def test_initialization(self, mock_cache_class, mock_client_class):
        """Test orchestrator initialization."""
        mock_client = MagicMock()
        mock_cache = MagicMock()
        mock_client_class.return_value = mock_client
        mock_cache_class.return_value = mock_cache
        
        orchestrator = MappingOrchestrator(
            default_zoom=15,
            default_size="800x600"
        )
        
        assert orchestrator.client == mock_client
        assert orchestrator.cache == mock_cache
        assert orchestrator.default_zoom == 15
        assert orchestrator.default_size == "800x600"
    
    def test_get_map_for_place_cached(self):
        """Test getting map that's already cached."""
        # Set up mocks
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        cached_data = b"CACHED_MAP_DATA"
        mock_cache.get_cached_bytes.return_value = cached_data
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        # Get map
        result = orchestrator.get_map_for_place("Rome", zoom=10)
        
        assert result == cached_data
        mock_cache.get_cached_bytes.assert_called_once_with("Rome", 10, "600x400", "roadmap")
        # Should not call geocoding or fetching
        mock_client.geocode_place.assert_not_called()
        mock_client.fetch_map_bytes.assert_not_called()
    
    def test_get_map_for_place_not_cached(self):
        """Test getting map that needs to be fetched."""
        # Set up mocks
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        # Cache miss
        mock_cache.get_cached_bytes.return_value = None
        
        # Geocoding result
        mock_client.geocode_place.return_value = {"lat": 41.9028, "lng": 12.4964}
        
        # Map data
        map_data = b"NEW_MAP_DATA"
        mock_client.fetch_map_bytes.return_value = map_data
        
        # Cache key
        mock_cache.cache_bytes.return_value = "Rome_abc123.png"
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        # Get map
        result = orchestrator.get_map_for_place("Rome", zoom=10, size="800x600")
        
        assert result == map_data
        
        # Verify call sequence
        mock_cache.get_cached_bytes.assert_called_once_with("Rome", 10, "800x600", "roadmap")
        mock_client.geocode_place.assert_called_once_with("Rome")
        mock_client.fetch_map_bytes.assert_called_once_with(
            lat=41.9028,
            lng=12.4964,
            zoom=10,
            size="800x600",
            map_type="roadmap"
        )
        mock_cache.cache_bytes.assert_called_once_with(
            "Rome", 10, "800x600", map_data, "roadmap"
        )
    
    def test_get_map_geocoding_error(self):
        """Test handling of geocoding errors."""
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        mock_cache.get_cached_bytes.return_value = None
        mock_client.geocode_place.side_effect = GeocodingError("Place not found")
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        with pytest.raises(GeocodingError):
            orchestrator.get_map_for_place("Unknown Place")
    
    def test_batch_get_maps_success(self):
        """Test successful batch processing."""
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        # Set up mock responses
        mock_cache.get_cached_bytes.return_value = None
        mock_client.geocode_place.side_effect = [
            {"lat": 41.0082, "lng": 28.9784},  # Istanbul
            {"lat": 45.4408, "lng": 12.3155}   # Venice
        ]
        mock_client.fetch_map_bytes.side_effect = [
            b"ISTANBUL_MAP",
            b"VENICE_MAP"
        ]
        mock_cache.cache_bytes.side_effect = [
            "Istanbul_hash1.png",
            "Venice_hash2.png"
        ]
        mock_cache._generate_cache_key.side_effect = [
            "Istanbul_hash1.png",
            "Venice_hash2.png"
        ]
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        places = [
            {"place": "Istanbul", "zoom": 10},
            {"place": "Venice", "zoom": 12, "map_type": "satellite"}
        ]
        
        results = orchestrator.batch_get_maps(places)
        
        assert len(results) == 2
        assert results["Istanbul_hash1.png"] == b"ISTANBUL_MAP"
        assert results["Venice_hash2.png"] == b"VENICE_MAP"
    
    def test_batch_get_maps_partial_failure(self):
        """Test batch processing with some failures."""
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        mock_cache.get_cached_bytes.return_value = None
        
        # Only return cache keys for successful places (Istanbul and Rome)
        mock_cache._generate_cache_key.side_effect = [
            "Istanbul_hash1.png",  # First successful place
            "Rome_hash2.png"       # Second successful place
        ]
        
        # Set up mixed success/failure
        mock_client.geocode_place.side_effect = [
            {"lat": 41.0082, "lng": 28.9784},  # Istanbul - success
            GeocodingError("Not found"),        # Venice - fail
            {"lat": 41.9028, "lng": 12.4964}   # Rome - success
        ]
        mock_client.fetch_map_bytes.side_effect = [
            b"ISTANBUL_MAP",
            b"ROME_MAP"
        ]
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        places = [
            {"place": "Istanbul"},
            {"place": "Venice"},
            {"place": "Rome"}
        ]
        
        results = orchestrator.batch_get_maps(places)
        
        # Should have 2 successful results
        assert len(results) == 2
        assert "Istanbul_hash1.png" in results
        assert "Rome_hash2.png" in results
        assert results["Istanbul_hash1.png"] == b"ISTANBUL_MAP"
        assert results["Rome_hash2.png"] == b"ROME_MAP"
        
        # Venice should not be in results
    def test_batch_get_maps_partial_failure_realistic(self):
        """Test batch processing with some failures using realistic cache key generation."""
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        mock_cache.get_cached_bytes.return_value = None
        
        # Make cache key generation behave realistically
        def generate_key(place, zoom, size, map_type):
            # Simplified version of real cache key generation
            safe_place = place.replace(" ", "_")[:20]
            return f"{safe_place}_mock_hash.png"
        
        mock_cache._generate_cache_key.side_effect = generate_key
        
        # Set up mixed success/failure
        mock_client.geocode_place.side_effect = [
            {"lat": 41.0082, "lng": 28.9784},  # Istanbul - success
            GeocodingError("Not found"),        # Venice - fail
            {"lat": 41.9028, "lng": 12.4964}   # Rome - success
        ]
        mock_client.fetch_map_bytes.side_effect = [
            b"ISTANBUL_MAP",
            b"ROME_MAP"
        ]
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        places = [
            {"place": "Istanbul", "zoom": 10},
            {"place": "Venice", "zoom": 12},
            {"place": "Rome", "zoom": 11}
        ]
        
        results = orchestrator.batch_get_maps(places)
        
        # Should have 2 successful results
        assert len(results) == 2
        
        # Check that the correct places succeeded
        istanbul_key = "Istanbul_mock_hash.png"
        rome_key = "Rome_mock_hash.png"
        
        assert istanbul_key in results
        assert rome_key in results
        assert results[istanbul_key] == b"ISTANBUL_MAP"
        assert results[rome_key] == b"ROME_MAP"
        
        # Verify cache key generation was called with correct parameters
        assert mock_cache._generate_cache_key.call_count == 2
        mock_cache._generate_cache_key.assert_any_call(place="Istanbul", zoom=10, size="600x400", map_type="roadmap")
        mock_cache._generate_cache_key.assert_any_call(place="Rome", zoom=11, size="600x400", map_type="roadmap")
        assert "Venice_hash2.png" not in results
    
    def test_batch_get_maps_empty_list(self):
        """Test batch processing with empty list."""
        orchestrator = MappingOrchestrator()
        results = orchestrator.batch_get_maps([])
        assert results == {}
    
    def test_validate_place_entry(self):
        """Test place entry validation."""
        orchestrator = MappingOrchestrator()
        
        # Valid entries
        assert orchestrator.validate_place_entry({"place": "London"}) is True
        assert orchestrator.validate_place_entry({"place": "Paris", "zoom": 10}) is True
        assert orchestrator.validate_place_entry({
            "place": "Berlin",
            "zoom": 15,
            "map_type": "satellite"
        }) is True
        
        # Invalid entries
        assert orchestrator.validate_place_entry({}) is False
        assert orchestrator.validate_place_entry({"place": ""}) is False
        assert orchestrator.validate_place_entry({"place": "London", "zoom": 25}) is False
        assert orchestrator.validate_place_entry({"place": "London", "zoom": "ten"}) is False
        assert orchestrator.validate_place_entry({
            "place": "London",
            "map_type": "invalid"
        }) is False
        assert orchestrator.validate_place_entry("not a dict") is False
    
    def test_preprocess_places(self):
        """Test place preprocessing."""
        orchestrator = MappingOrchestrator(default_zoom=12, default_size="600x400")
        
        raw_places = [
            {"place": "  London  "},  # Needs trimming
            {"place": "Paris", "zoom": 10},  # Valid
            {"place": ""},  # Invalid - empty
            {"place": "Berlin", "zoom": "invalid"},  # Invalid zoom
            {"place": "Rome", "map_type": "satellite"}  # Valid
        ]
        
        processed = orchestrator.preprocess_places(raw_places)
        
        assert len(processed) == 3  # Only valid entries
        assert processed[0]["place"] == "London"
        assert processed[0]["zoom"] == 12  # Default
        assert processed[1]["place"] == "Paris"
        assert processed[1]["zoom"] == 10
        assert processed[2]["place"] == "Rome"
        assert processed[2]["map_type"] == "satellite"
    
    def test_get_stats(self):
        """Test statistics gathering."""
        mock_client = MagicMock()
        mock_cache = MagicMock()
        
        mock_cache.get_cache_stats.return_value = {
            "total_files": 10,
            "total_size_mb": 25.5
        }
        mock_client.get_rate_limit_status.return_value = {
            "available_tokens": 8.5,
            "wait_time_seconds": 0.0
        }
        
        orchestrator = MappingOrchestrator(
            maps_client=mock_client,
            cache_manager=mock_cache
        )
        
        stats = orchestrator.get_stats()
        
        assert stats["cache"]["total_files"] == 10
        assert stats["cache"]["total_size_mb"] == 25.5
        assert stats["rate_limit"]["available_tokens"] == 8.5
        assert stats["rate_limit"]["wait_time_seconds"] == 0.0


class TestIntegration:
    """Integration tests for the complete mapping workflow."""
    
    @patch('src.mapping.mapping_client.googlemaps.Client')
    @patch('src.mapping.mapping_client.requests.Session')
    @patch('src.mapping.mapping_client.get_config')
    @patch('src.mapping.mapping_cache.get_config')
    def test_full_workflow(self, mock_cache_get_config, mock_client_get_config, 
                          mock_session, mock_gmaps, temp_cache_dir):
        """Test complete workflow from orchestrator to cache."""
        # Configure all get_config mocks
        mock_client_get_config.return_value = "test_key"
        mock_cache_get_config.return_value = str(temp_cache_dir)
        
        # Mock geocoding
        mock_gmaps_instance = MagicMock()
        mock_gmaps.return_value = mock_gmaps_instance
        mock_gmaps_instance.geocode.return_value = [{
            'geometry': {'location': {'lat': 51.5074, 'lng': -0.1278}}
        }]
        
        # Mock map fetching
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"LONDON_MAP_IMAGE_DATA" * 100
        
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        
        # Create orchestrator with real components
        orchestrator = MappingOrchestrator()
        
        # First request - should fetch and cache
        result1 = orchestrator.get_map_for_place("London", zoom=10)
        assert result1 == mock_response.content
        assert mock_gmaps_instance.geocode.call_count == 1
        assert mock_session_instance.get.call_count == 1
        
        # Second request - should use cache
        result2 = orchestrator.get_map_for_place("London", zoom=10)
        assert result2 == mock_response.content
        assert mock_gmaps_instance.geocode.call_count == 1  # No new call
        assert mock_session_instance.get.call_count == 1  # No new call
        
        # Verify cache file exists
        cache_files = list(temp_cache_dir.glob("*.png"))
        assert len(cache_files) == 1
        assert "London" in cache_files[0].name


# ==================== MAIN EXECUTION ====================

def run_tests():
    """Run all tests with appropriate configuration."""
    import subprocess
    
    # Get the directory of this test file
    test_file = Path(__file__)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "-p", "no:warnings"  # Suppress warnings for cleaner output
    ]
    
    # Add coverage if pytest-cov is installed
    try:
        import pytest_cov
        cmd.extend([
            "--cov=src.mapping",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
        print("Running with coverage reporting...")
    except ImportError:
        print("Running without coverage (install pytest-cov for coverage reports)")
    
    print(f"Executing: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        if 'pytest_cov' in sys.modules:
            print("📊 Coverage report available in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
    
    return result.returncode


if __name__ == "__main__":
    # When run directly, execute the tests
    exit_code = run_tests()
    sys.exit(exit_code)