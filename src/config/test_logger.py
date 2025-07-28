"""
Unit tests for logger.py module.

Tests cover:
- Logger initialization and handler attachment
- Log level configuration
- File and console output
- Idempotency of initialization
- Convenience logging methods
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from .logger_module import initialize_logger, log_info, log_warning, log_error


class TestInitializeLogger:
    """Test cases for initialize_logger function."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        # Clear all handlers and reset initialization flag
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)  # Reset to default
        
        # Reset the module-level flag
        import src.config.logger_module
        src.config.logger_module._logger_initialized = False
    
    def test_initialize_logger_default_parameters(self, tmp_path):
        """Test logger initialization with default parameters."""
        log_file = tmp_path / "logs" / "app.log"
        
        initialize_logger(log_file=str(log_file))
        
        root_logger = logging.getLogger()
        
        # Check logger level
        assert root_logger.level == logging.INFO
        
        # Check exactly 2 handlers (console + file)
        assert len(root_logger.handlers) == 2
        
        # Check handler types
        handler_types = [type(h).__name__ for h in root_logger.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" in handler_types
        
        # Check log file was created
        assert log_file.exists()
    
    def test_initialize_logger_custom_level(self, tmp_path):
        """Test logger initialization with custom log level."""
        log_file = tmp_path / "test.log"
        
        initialize_logger(log_level="DEBUG", log_file=str(log_file))
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_initialize_logger_invalid_level(self, tmp_path):
        """Test logger initialization with invalid log level defaults to INFO."""
        log_file = tmp_path / "test.log"
        
        initialize_logger(log_level="INVALID", log_file=str(log_file))
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    def test_initialize_logger_creates_directory(self, tmp_path):
        """Test that logger creates log directory if it doesn't exist."""
        log_file = tmp_path / "deep" / "nested" / "path" / "app.log"
        
        initialize_logger(log_file=str(log_file))
        
        assert log_file.parent.exists()
        assert log_file.exists()
    
    def test_initialize_logger_idempotency(self, tmp_path):
        """Test that multiple calls to initialize_logger don't duplicate handlers."""
        log_file = tmp_path / "test.log"
        
        # First initialization
        initialize_logger(log_file=str(log_file))
        root_logger = logging.getLogger()
        handler_count_1 = len(root_logger.handlers)
        
        # Second initialization
        initialize_logger(log_file=str(log_file))
        handler_count_2 = len(root_logger.handlers)
        
        # Third initialization with different parameters
        initialize_logger(log_level="DEBUG", log_file=str(log_file))
        handler_count_3 = len(root_logger.handlers)
        
        assert handler_count_1 == 2
        assert handler_count_2 == 2  # Should not increase
        assert handler_count_3 == 2  # Should not increase
    
    def test_initialize_logger_handler_levels(self, tmp_path):
        """Test that handlers have correct log levels."""
        log_file = tmp_path / "test.log"
        
        initialize_logger(log_file=str(log_file))
        
        root_logger = logging.getLogger()
        
        # Find console and file handlers
        console_handler = None
        file_handler = None
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                console_handler = handler
            elif isinstance(handler, logging.FileHandler):
                file_handler = handler
        
        assert console_handler is not None
        assert file_handler is not None
        assert console_handler.level == logging.INFO
        assert file_handler.level == logging.DEBUG
    
    def test_initialize_logger_formatters(self, tmp_path):
        """Test that handlers have appropriate formatters."""
        log_file = tmp_path / "test.log"
        
        initialize_logger(log_file=str(log_file))
        
        root_logger = logging.getLogger()
        
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            format_string = handler.formatter._fmt
            assert "%(asctime)s" in format_string
            assert "%(levelname)s" in format_string
            assert "%(message)s" in format_string


class TestLoggingOutput:
    """Test cases for actual logging output."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)
        
        import src.config.logger_module
        src.config.logger_module._logger_initialized = False
    
    def test_log_messages_written_to_file(self, tmp_path):
        """Test that log messages are actually written to file."""
        log_file = tmp_path / "test.log"
        
        initialize_logger(log_level="DEBUG", log_file=str(log_file))
        
        # Generate test messages
        root_logger = logging.getLogger()
        root_logger.info("Test info message")
        root_logger.warning("Test warning message")
        root_logger.error("Test error message")
        
        # Force flush
        for handler in root_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Read log file content
        log_content = log_file.read_text()
        
        assert "INFO" in log_content
        assert "Test info message" in log_content
        assert "WARNING" in log_content
        assert "Test warning message" in log_content
        assert "ERROR" in log_content
        assert "Test error message" in log_content
    
    def test_log_levels_respected(self, tmp_path):
        """Test that log levels are respected."""
        log_file = tmp_path / "test.log"
        
        # Initialize with WARNING level
        initialize_logger(log_level="WARNING", log_file=str(log_file))
        
        root_logger = logging.getLogger()
        root_logger.debug("Debug message")
        root_logger.info("Info message")
        root_logger.warning("Warning message")
        root_logger.error("Error message")
        
        # Force flush
        for handler in root_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        log_content = log_file.read_text()
        
        # DEBUG and INFO should not appear (below WARNING level)
        assert "Debug message" not in log_content
        assert "Info message" not in log_content
        
        # WARNING and ERROR should appear
        assert "Warning message" in log_content
        assert "Error message" in log_content


class TestConvenienceMethods:
    """Test cases for convenience logging methods."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)
        
        import src.config.logger_module
        src.config.logger_module._logger_initialized = False
    
    def test_log_info(self, tmp_path):
        """Test log_info convenience method."""
        log_file = tmp_path / "test.log"
        initialize_logger(log_level="INFO", log_file=str(log_file))
        
        log_info("Test info message")
        
        # Force flush
        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        log_content = log_file.read_text()
        assert "INFO" in log_content
        assert "Test info message" in log_content
    
    def test_log_warning(self, tmp_path):
        """Test log_warning convenience method."""
        log_file = tmp_path / "test.log"
        initialize_logger(log_level="INFO", log_file=str(log_file))
        
        log_warning("Test warning message")
        
        # Force flush
        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        log_content = log_file.read_text()
        assert "WARNING" in log_content
        assert "Test warning message" in log_content
    
    def test_log_error(self, tmp_path):
        """Test log_error convenience method."""
        log_file = tmp_path / "test.log"
        initialize_logger(log_level="INFO", log_file=str(log_file))
        
        log_error("Test error message")
        
        # Force flush
        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        log_content = log_file.read_text()
        assert "ERROR" in log_content
        assert "Test error message" in log_content
    
    def test_convenience_methods_before_initialization(self, tmp_path):
        """Test that convenience methods work even before explicit initialization."""
        # Don't initialize logger explicitly
        log_warning("Warning without initialization")
        
        # Should not raise an exception
        # Note: This tests the fallback behavior of getting root logger
    
    @patch('logging.getLogger')
    def test_convenience_methods_call_correct_levels(self, mock_get_logger):
        """Test that convenience methods call the correct logging levels."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_info("info message")
        log_warning("warning message")
        log_error("error message")
        
        mock_logger.info.assert_called_once_with("info message")
        mock_logger.warning.assert_called_once_with("warning message")
        mock_logger.error.assert_called_once_with("error message")


class TestThreadSafety:
    """Test cases for thread safety and concurrent access."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        import src.config.logger_module
        src.config.logger_module._logger_initialized = False
    