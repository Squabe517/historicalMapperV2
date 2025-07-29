"""
Test suite for OpenAI client place name extraction with structured outputs.
"""

import json
import logging
import pytest
from unittest.mock import Mock, patch, MagicMock

from .openai_client import OpenAIClient, OpenAIError, Place, PlacesList


@pytest.fixture
def mock_config():
    """Mock configuration values."""
    config_values = {
        "OPENAI_API_KEY": "test-api-key",
        "OPENAI_MODEL": "gpt-4o-2024-08-06",
    }
    
    def get_config(key, default=None):
        return config_values.get(key, default)
    
    with patch('src.ai.openai_client.get_config', side_effect=get_config):
        yield get_config


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI structured response."""
    response = Mock()
    response.choices = [Mock()]
    
    # Create mock parsed output with Pydantic models
    parsed_output = PlacesList(places=[
        Place(place="Paris", zoom=11),
        Place(place="London", zoom=11),
        Place(place="New York", zoom=11)
    ])
    
    response.choices[0].message = Mock()
    response.choices[0].message.parsed = parsed_output
    return response


@pytest.fixture
def client(mock_config):
    """Create a test client with mocked configuration."""
    with patch('src.ai.openai_client.OpenAI'):
        return OpenAIClient()


class TestOpenAIClientInitialization:
    """Test client initialization."""
    
    def test_init_with_defaults(self, mock_config):
        """Test initialization with default configuration."""
        with patch('src.ai.openai_client.OpenAI') as mock_openai:
            client = OpenAIClient()
            
            assert client.api_key == "test-api-key"
            assert client.model == "gpt-4o-2024-08-06"
            mock_openai.assert_called_once_with(api_key="test-api-key")
    
    def test_init_with_custom_values(self, mock_config):
        """Test initialization with custom API key and model."""
        with patch('src.ai.openai_client.OpenAI') as mock_openai:
            client = OpenAIClient(api_key="custom-key", model="gpt-4o-mini")
            
            assert client.api_key == "custom-key"
            assert client.model == "gpt-4o-mini"
            mock_openai.assert_called_once_with(api_key="custom-key")
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch('src.ai.openai_client.get_config', return_value=None):
            with pytest.raises(OpenAIError, match="OpenAI API key not provided"):
                OpenAIClient()


class TestAnalyzeChunk:
    """Test single chunk analysis with structured outputs."""
    
    def test_analyze_chunk_success(self, client, mock_openai_response):
        """Test successful place name extraction with zoom levels."""
        client.client.beta.chat.completions.parse = Mock(return_value=mock_openai_response)
        
        result = client.analyze_chunk("I traveled from Paris to London and then to New York.")
        
        assert len(result) == 3
        assert result[0] == {"place": "Paris", "zoom": 11}
        assert result[1] == {"place": "London", "zoom": 11}
        assert result[2] == {"place": "New York", "zoom": 11}
        client.client.beta.chat.completions.parse.assert_called_once()
    
    def test_analyze_chunk_empty_input(self, client):
        """Test handling of empty input."""
        result = client.analyze_chunk("")
        assert result == []
        
        result = client.analyze_chunk("   ")
        assert result == []
    
    def test_analyze_chunk_different_zoom_levels(self, client):
        """Test extraction with different zoom levels for different place types."""
        response = Mock()
        response.choices = [Mock()]
        
        parsed_output = PlacesList(places=[
            Place(place="Europe", zoom=4),
            Place(place="Italy", zoom=6),
            Place(place="Rome", zoom=11),
            Place(place="Colosseum", zoom=17)
        ])
        
        response.choices[0].message.parsed = parsed_output
        client.client.beta.chat.completions.parse = Mock(return_value=response)
        
        result = client.analyze_chunk("From Europe to Italy, visiting Rome and the Colosseum.")
        
        assert len(result) == 4
        assert result[0]["place"] == "Europe"
        assert result[0]["zoom"] == 4  # Continent
        assert result[1]["place"] == "Italy"
        assert result[1]["zoom"] == 6  # Country
        assert result[2]["place"] == "Rome"
        assert result[2]["zoom"] == 11  # City
        assert result[3]["place"] == "Colosseum"
        assert result[3]["zoom"] == 17  # Landmark
    
    def test_analyze_chunk_retry_on_rate_limit(self, client):
        """Test retry logic for rate limit errors."""
        response = Mock()
        response.choices = [Mock()]
        
        parsed_output = PlacesList(places=[
            Place(place="Madrid", zoom=11)
        ])
        response.choices[0].message.parsed = parsed_output
        
        # First call fails with rate limit, second succeeds
        client.client.beta.chat.completions.parse = Mock(
            side_effect=[
                Exception("Rate limit exceeded"),
                response
            ]
        )
        
        result = client.analyze_chunk("Madrid is the capital of Spain.")
        assert result == [{"place": "Madrid", "zoom": 11}]
        assert client.client.beta.chat.completions.parse.call_count == 2
    
    def test_analyze_chunk_api_error(self, client):
        """Test handling of non-retryable API errors."""
        client.client.beta.chat.completions.parse = Mock(
            side_effect=Exception("Invalid request")
        )
        
        with pytest.raises(OpenAIError, match="OpenAI API call failed"):
            client.analyze_chunk("Some text")


class TestBatchAnalyzeChunks:
    """Test batch chunk analysis."""
    
    def test_batch_analyze_empty_list(self, client):
        """Test handling of empty chunk list."""
        result = client.batch_analyze_chunks([])
        assert result == []
    
    def test_batch_analyze_success(self, client):
        """Test successful batch analysis."""
        chunks = [
            "Paris is beautiful.",
            "London has great museums.",
            "New York is vibrant."
        ]
        
        responses = [
            [{"place": "Paris", "zoom": 11}],
            [{"place": "London", "zoom": 11}],
            [{"place": "New York", "zoom": 11}]
        ]
        
        def mock_analyze(chunk):
            for i, c in enumerate(chunks):
                if chunk == c:
                    return responses[i]
            return []
        
        client.analyze_chunk = Mock(side_effect=mock_analyze)
        
        results = client.batch_analyze_chunks(chunks)
        
        assert len(results) == 3
        assert results[0] == [{"place": "Paris", "zoom": 11}]
        assert results[1] == [{"place": "London", "zoom": 11}]
        assert results[2] == [{"place": "New York", "zoom": 11}]
        assert client.analyze_chunk.call_count == 3
    
    def test_batch_analyze_with_failures(self, client):
        """Test batch analysis with some failures."""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        def mock_analyze(chunk):
            if chunk == "Chunk 2":
                raise Exception("API Error")
            return [{"place": "Place", "zoom": 10}]
        
        client.analyze_chunk = Mock(side_effect=mock_analyze)
        
        with patch.object(client.logger, 'error') as mock_logger:
            results = client.batch_analyze_chunks(chunks)
            
            assert len(results) == 3
            assert results[0] == [{"place": "Place", "zoom": 10}]
            assert results[1] == []  # Failed chunk
            assert results[2] == [{"place": "Place", "zoom": 10}]
            
            # Check that error was logged
            mock_logger.assert_called()
    
    def test_batch_analyze_progress_logging(self, client):
        """Test progress logging for large batches."""
        chunks = [f"Chunk {i}" for i in range(25)]
        
        client.analyze_chunk = Mock(return_value=[{"place": "Place", "zoom": 10}])
        
        with patch.object(client.logger, 'info') as mock_logger:
            client.batch_analyze_chunks(chunks)
            
            # Should log progress at chunks 0, 10, and 20
            progress_logs = [call for call in mock_logger.call_args_list 
                           if "Processing chunk" in str(call)]
            assert len(progress_logs) == 3


class TestGetUsageStats:
    """Test usage statistics."""
    
    def test_get_usage_stats(self, client):
        """Test getting usage statistics."""
        stats = client.get_usage_stats()
        
        assert stats["model"] == "gpt-4o-2024-08-06"
        assert stats["api_key_set"] is True
        assert stats["supports_structured_outputs"] is True


class TestExtractPlaceNamesOnly:
    """Test legacy method for backward compatibility."""
    
    def test_extract_place_names_only(self, client):
        """Test extraction of place names without zoom levels."""
        client.analyze_chunk = Mock(return_value=[
            {"place": "Tokyo", "zoom": 11},
            {"place": "Kyoto", "zoom": 11},
            {"place": "Osaka", "zoom": 11}
        ])
        
        result = client.extract_place_names_only("Japanese cities include Tokyo, Kyoto, and Osaka.")
        
        assert result == ["Tokyo", "Kyoto", "Osaka"]
        client.analyze_chunk.assert_called_once()


class TestIntegrationScenarios:
    """Test real-world usage scenarios."""
    
    def test_historical_text_extraction(self, client):
        """Test extraction from historical text."""
        historical_text = """
        In 1492, Columbus sailed from Palos de la Frontera in Spain,
        across the Atlantic Ocean, and reached the Bahamas. He later
        visited Cuba and Hispaniola before returning to Castile.
        """
        
        response = Mock()
        response.choices = [Mock()]
        
        parsed_output = PlacesList(places=[
            Place(place="Palos de la Frontera", zoom=13),
            Place(place="Spain", zoom=6),
            Place(place="Atlantic Ocean", zoom=3),
            Place(place="Bahamas", zoom=8),
            Place(place="Cuba", zoom=7),
            Place(place="Hispaniola", zoom=8),
            Place(place="Castile", zoom=7)
        ])
        
        response.choices[0].message.parsed = parsed_output
        client.client.beta.chat.completions.parse = Mock(return_value=response)
        
        result = client.analyze_chunk(historical_text)
        assert len(result) == 7
        assert any(p["place"] == "Spain" and p["zoom"] == 6 for p in result)
        assert any(p["place"] == "Cuba" and p["zoom"] == 7 for p in result)
        assert any(p["place"] == "Atlantic Ocean" and p["zoom"] == 3 for p in result)
    
    def test_modern_place_name_conversion(self, client):
        """Test conversion to modern place names."""
        text = "Constantinople was the capital of Byzantium."
        
        response = Mock()
        response.choices = [Mock()]
        
        # API should return modern equivalent
        parsed_output = PlacesList(places=[
            Place(place="Istanbul", zoom=11),
            Place(place="Turkey", zoom=6)  # Modern country name
        ])
        
        response.choices[0].message.parsed = parsed_output
        client.client.beta.chat.completions.parse = Mock(return_value=response)
        
        result = client.analyze_chunk(text)
        assert len(result) == 2
        assert result[0]["place"] == "Istanbul"
        assert result[0]["zoom"] == 11
        assert result[1]["place"] == "Turkey"
        assert result[1]["zoom"] == 6


@pytest.mark.parametrize("chunk,expected", [
    ("", []),
    ("   \n\t  ", []),
    ("No places mentioned here.", []),
    ("Single place: Paris", [{"place": "Paris", "zoom": 11}]),
])
def test_various_inputs(client, chunk, expected):
    """Test various input scenarios."""
    if chunk.strip() and expected:
        response = Mock()
        response.choices = [Mock()]
        
        places = [Place(**p) for p in expected]
        parsed_output = PlacesList(places=places)
        response.choices[0].message.parsed = parsed_output
        
        client.client.beta.chat.completions.parse = Mock(return_value=response)
    
    result = client.analyze_chunk(chunk)
    assert result == expected


class TestStructuredOutputFormat:
    """Test the structured output format requirements."""
    
    def test_parse_method_called_with_correct_params(self, client):
        """Test that parse method is called with correct parameters."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.parsed = PlacesList(places=[])
        
        client.client.beta.chat.completions.parse = Mock(return_value=response)
        
        client.analyze_chunk("Some text")
        
        # Verify the parse method was called with the correct parameters
        call_args = client.client.beta.chat.completions.parse.call_args
        assert call_args[1]["model"] == "gpt-4o-2024-08-06"
        assert call_args[1]["response_format"] == PlacesList
        assert call_args[1]["temperature"] == 0
        assert len(call_args[1]["messages"]) == 2