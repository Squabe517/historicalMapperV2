"""
Unit tests for OpenAI client implementation.

Tests cover API interactions, retry logic, error handling,
and batch processing with comprehensive mocking.
"""

import json
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

from src.ai.openai_client import OpenAIClient, OpenAIError


class TestOpenAIClientInitialization:
    """Test cases for OpenAI client initialization."""
    
    @patch('src.ai.openai_client.get_config')
    def test_initialization_with_defaults(self, mock_get_config):
        """Test client initialization with default config values."""
        mock_get_config.side_effect = lambda key, default=None: {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_MAX_CONCURRENT": "5"
        }.get(key, default)
        
        client = OpenAIClient()
        
        assert client.api_key == "test-key"
        assert client.model == "gpt-4o"
        assert client.max_concurrent == 5
    
    def test_initialization_with_explicit_params(self):
        """Test client initialization with explicit parameters."""
        client = OpenAIClient(api_key="explicit-key", model="gpt-3.5-turbo")
        
        assert client.api_key == "explicit-key"
        assert client.model == "gpt-3.5-turbo"
    
    @patch('src.ai.openai_client.get_config')
    def test_initialization_missing_api_key(self, mock_get_config):
        """Test initialization fails without API key."""
        mock_get_config.return_value = None
        
        with pytest.raises(OpenAIError, match="OpenAI API key not provided"):
            OpenAIClient()


class TestAnalyzeChunk:
    """Test cases for single chunk analysis."""
    
    def setup_method(self):
        """Set up test client instance."""
        self.client = OpenAIClient(api_key="test-key")
    
    def test_analyze_chunk_empty_input(self):
        """Test analyze_chunk with empty input."""
        assert self.client.analyze_chunk("") == []
        assert self.client.analyze_chunk("   ") == []
        assert self.client.analyze_chunk(None) == []
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_success_direct_array(self, mock_openai_class):
        """Test successful chunk analysis with direct JSON array response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["Paris", "London", "Berlin"]'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        result = client.analyze_chunk("Text about Paris, London, and Berlin.")
        
        assert result == ["Paris", "London", "Berlin"]
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_success_object_response(self, mock_openai_class):
        """Test successful chunk analysis with object containing places array."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"places": ["Rome", "Athens"]}'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        result = client.analyze_chunk("Ancient Rome and Athens were great cities.")
        
        assert result == ["Rome", "Athens"]
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_malformed_json(self, mock_openai_class):
        """Test chunk analysis with malformed JSON response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'Not valid JSON at all'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        
        with pytest.raises(OpenAIError, match="Invalid JSON response"):
            client.analyze_chunk("Some text")
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_rate_limit_retry(self, mock_openai_class):
        """Test rate limit handling with eventual success."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # First call raises rate limit error, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["Success"]'
        
        mock_client.chat.completions.create.side_effect = [
            Exception("rate limit exceeded"),
            mock_response
        ]
        
        client = OpenAIClient(api_key="test-key")
        result = client.analyze_chunk("Test text")
        
        assert result == ["Success"]
        assert mock_client.chat.completions.create.call_count == 2
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_persistent_failure(self, mock_openai_class):
        """Test failure after maximum retries."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Persistent rate limit")
        
        client = OpenAIClient(api_key="test-key")
        
        with pytest.raises(OpenAIError):
            client.analyze_chunk("Test text")
        
        assert mock_client.chat.completions.create.call_count == 3  # Default retry count
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_api_connection_error(self, mock_openai_class):
        """Test handling of API connection errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        
        client = OpenAIClient(api_key="test-key")
        
        with pytest.raises(OpenAIError):
            client.analyze_chunk("Test text")
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_unexpected_error(self, mock_openai_class):
        """Test handling of unexpected errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Unexpected error")
        
        client = OpenAIClient(api_key="test-key")
        
        with pytest.raises(OpenAIError, match="OpenAI API call failed"):
            client.analyze_chunk("Test text")
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_filters_empty_places(self, mock_openai_class):
        """Test that empty place names are filtered out."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["Paris", "", "London", null, "Berlin"]'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        result = client.analyze_chunk("Test text")
        
        assert result == ["Paris", "London", "Berlin"]
    
    @patch('src.ai.openai_client.OpenAI')
    def test_analyze_chunk_logging(self, mock_openai_class, caplog):
        """Test that chunk analysis is properly logged."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["Paris"]'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        
        with caplog.at_level(logging.DEBUG):
            client.analyze_chunk("Short text")
        
        assert "Analyzed chunk" in caplog.text
        assert "found 1 places" in caplog.text


class TestBatchAnalyzeChunks:
    """Test cases for batch chunk processing."""
    
    def setup_method(self):
        """Set up test client instance."""
        self.client = OpenAIClient(api_key="test-key")
    
    def test_batch_analyze_empty_input(self, caplog):
        """Test batch analysis with empty input."""
        with caplog.at_level(logging.INFO):
            result = self.client.batch_analyze_chunks([])
        
        assert result == []
        assert "No chunks to analyze" in caplog.text
    
    @patch.object(OpenAIClient, 'analyze_chunk')
    def test_batch_analyze_success(self, mock_analyze, caplog):
        """Test successful batch processing."""
        mock_analyze.side_effect = [
            ["Paris", "London"],
            ["Berlin", "Rome"],
            ["Tokyo"]
        ]
        
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        with caplog.at_level(logging.INFO):
            result = self.client.batch_analyze_chunks(chunks)
        
        assert result == [["Paris", "London"], ["Berlin", "Rome"], ["Tokyo"]]
        assert mock_analyze.call_count == 3
        assert "Sending 3 chunks to OpenAI" in caplog.text
        assert "3/3 chunks successful" in caplog.text
        assert "5 total places found" in caplog.text
    
    @patch.object(OpenAIClient, 'analyze_chunk')
    def test_batch_analyze_partial_failure(self, mock_analyze, caplog):
        """Test batch processing with some failures."""
        mock_analyze.side_effect = [
            ["Paris"],
            OpenAIError("API error"),
            ["London"]
        ]
        
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        with caplog.at_level(logging.INFO):
            result = self.client.batch_analyze_chunks(chunks)
        
        # Should return empty list for failed chunk
        assert result == [["Paris"], [], ["London"]]
        assert "Failed to analyze chunk 1" in caplog.text
        assert "Encountered 1 errors" in caplog.text
    
    @patch.object(OpenAIClient, 'analyze_chunk')
    def test_batch_analyze_preserves_order(self, mock_analyze):
        """Test that batch processing preserves chunk order."""
        # Use different delays to test concurrent processing
        def slow_analyze(chunk):
            if "slow" in chunk:
                import time
                time.sleep(0.1)
                return ["Slow"]
            return ["Fast"]
        
        mock_analyze.side_effect = slow_analyze
        
        chunks = ["fast chunk", "slow chunk", "another fast"]
        result = self.client.batch_analyze_chunks(chunks)
        
        assert result == [["Fast"], ["Slow"], ["Fast"]]
    

class TestUsageStats:
    """Test cases for usage statistics."""
    
    def test_get_usage_stats(self):
        """Test usage statistics collection."""
        client = OpenAIClient(api_key="test-key", model="gpt-4")
        
        stats = client.get_usage_stats()
        
        assert stats["model"] == "gpt-4"
        assert stats["max_concurrent"] == 5  # Default value
        assert stats["api_key_set"] is True



class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def test_openai_error_inheritance(self):
        """Test OpenAIError exception inheritance."""
        assert issubclass(OpenAIError, Exception)
    
    def test_openai_error_message(self):
        """Test OpenAIError message handling."""
        error = OpenAIError("Test error message")
        assert str(error) == "Test error message"
    
    @patch('src.ai.openai_client.OpenAI')
    def test_semaphore_usage(self, mock_openai_class):
        """Test that semaphore is used for rate limiting."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["Test"]'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        
        # Just verify the semaphore exists and is configured correctly
        assert client.semaphore is not None
        assert client.semaphore._value <= client.max_concurrent
        assert client.max_concurrent == 5  # Default value
        
        # Verify the method runs without error (semaphore is used internally)
        result = client.analyze_chunk("Test chunk")
        assert result == ["Test"]


class TestIntegrationScenarios:
    """Integration test scenarios for realistic use cases."""
    
    def setup_method(self):
        """Set up test client instance."""
        self.client = OpenAIClient(api_key="test-key")
    
    @patch('src.ai.openai_client.OpenAI')
    def test_realistic_historical_text(self, mock_openai_class):
        """Test with realistic historical text content."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"places": ["Constantinople", "Ottoman Empire", "Byzantine Empire"]}'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        chunk = "In 1453, Constantinople fell to the Ottoman Empire, ending the Byzantine Empire."
        result = client.analyze_chunk(chunk)
        
        assert "Constantinople" in result
        assert "Ottoman Empire" in result
        assert "Byzantine Empire" in result
    
    @patch('src.ai.openai_client.OpenAI')
    def test_mixed_response_formats(self, mock_openai_class):
        """Test handling of different response formats in batch."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        responses = [
            '["Direct", "Array"]',
            '{"places": ["Object", "Format"]}',
            '{"locations": ["Alt", "Key"]}',
            '{"data": {"nested": "ignored"}}'  # Should result in empty
        ]
        
        mock_responses = []
        for response_text in responses:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = response_text
            mock_responses.append(mock_response)
        
        mock_client.chat.completions.create.side_effect = mock_responses
        
        client = OpenAIClient(api_key="test-key")
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        result = client.batch_analyze_chunks(chunks)
        
        assert result[0] == ["Direct", "Array"]
        assert result[1] == ["Object", "Format"]
        assert result[2] == ["Alt", "Key"]
        assert result[3] == []  # Malformed response should return empty