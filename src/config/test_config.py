"""
Unit tests for config.py module.

Tests cover:
- .env file loading and environment variable overriding
- get_config with present keys, missing keys, and defaults
- validate_config passing and failing scenarios
- ConfigError exception handling
"""

import os
import tempfile
import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config.config_module import load_config, get_config, validate_config, ConfigError


class TestLoadConfig:
    """Test cases for load_config function."""
    
    def test_load_config_existing_file(self, tmp_path, caplog):
        """Test loading configuration from existing .env file."""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test_value\nANOTHER_KEY=another_value\n")
        
        with caplog.at_level(logging.INFO):
            load_config(str(env_file))
        
        assert os.getenv("TEST_KEY") == "test_value"
        assert os.getenv("ANOTHER_KEY") == "another_value"
        assert f"Loaded configuration from {str(env_file)}" in caplog.text
    
    def test_load_config_nonexistent_file(self, caplog):
        """Test loading configuration when .env file doesn't exist."""
        nonexistent_file = "/path/that/does/not/exist/.env"
        
        with caplog.at_level(logging.WARNING):
            load_config(nonexistent_file)
        
        assert f"Configuration file {nonexistent_file} not found" in caplog.text
    
    def test_load_config_override_existing_env(self, tmp_path):
        """Test that .env file values override existing environment variables."""
        # Set initial environment variable
        os.environ["OVERRIDE_TEST"] = "original_value"
        
        # Create .env file with different value
        env_file = tmp_path / ".env"
        env_file.write_text("OVERRIDE_TEST=new_value\n")
        
        load_config(str(env_file))
        
        assert os.getenv("OVERRIDE_TEST") == "new_value"
        
        # Cleanup
        del os.environ["OVERRIDE_TEST"]
    
    def test_load_config_default_path(self, tmp_path, monkeypatch):
        """Test loading configuration with default .env path."""
        # Change to temporary directory
        monkeypatch.chdir(tmp_path)
        
        # Create .env file in current directory
        env_file = tmp_path / ".env"
        env_file.write_text("DEFAULT_PATH_TEST=success\n")
        
        load_config()
        
        assert os.getenv("DEFAULT_PATH_TEST") == "success"
        
        # Cleanup
        if "DEFAULT_PATH_TEST" in os.environ:
            del os.environ["DEFAULT_PATH_TEST"]


class TestGetConfig:
    """Test cases for get_config function."""
    
    def setup_method(self):
        """Set up test environment variables."""
        os.environ["EXISTING_KEY"] = "existing_value"
        os.environ["EMPTY_KEY"] = ""
    
    def teardown_method(self):
        """Clean up test environment variables."""
        keys_to_remove = ["EXISTING_KEY", "EMPTY_KEY"]
        for key in keys_to_remove:
            if key in os.environ:
                del os.environ[key]
    
    def test_get_config_existing_key(self):
        """Test getting value for existing environment variable."""
        result = get_config("EXISTING_KEY")
        assert result == "existing_value"
    
    def test_get_config_missing_key_with_default(self, caplog):
        """Test getting value for missing key with default."""
        with caplog.at_level(logging.WARNING):
            result = get_config("MISSING_KEY", "default_value")
        
        assert result == "default_value"
        assert "Configuration key 'MISSING_KEY' not found, using default value: default_value" in caplog.text
    
    def test_get_config_missing_key_no_default(self, caplog):
        """Test getting value for missing key without default."""
        with caplog.at_level(logging.WARNING):
            result = get_config("MISSING_KEY")
        
        assert result is None
        assert "Configuration key 'MISSING_KEY' not found and no default provided" in caplog.text
    
    def test_get_config_empty_key(self):
        """Test getting value for empty environment variable."""
        result = get_config("EMPTY_KEY")
        assert result == ""
    
    def test_get_config_various_defaults(self):
        """Test get_config with various default value types."""
        assert get_config("MISSING_KEY", "string") == "string"
        assert get_config("MISSING_KEY", 42) == 42
        assert get_config("MISSING_KEY", [1, 2, 3]) == [1, 2, 3]
        assert get_config("MISSING_KEY", {"key": "value"}) == {"key": "value"}


class TestValidateConfig:
    """Test cases for validate_config function."""
    
    def setup_method(self):
        """Set up test environment variables."""
        os.environ["VALID_KEY1"] = "value1"
        os.environ["VALID_KEY2"] = "value2"
        os.environ["EMPTY_KEY"] = ""
        os.environ["WHITESPACE_KEY"] = "   "
    
    def teardown_method(self):
        """Clean up test environment variables."""
        keys_to_remove = ["VALID_KEY1", "VALID_KEY2", "EMPTY_KEY", "WHITESPACE_KEY"]
        for key in keys_to_remove:
            if key in os.environ:
                del os.environ[key]
    
    def test_validate_config_all_present(self, caplog):
        """Test validation when all required keys are present."""
        with caplog.at_level(logging.INFO):
            validate_config(["VALID_KEY1", "VALID_KEY2"])
        
        assert "Configuration validation passed" in caplog.text
    
    def test_validate_config_missing_keys(self, caplog):
        """Test validation when some keys are missing."""
        with pytest.raises(ConfigError) as exc_info:
            validate_config(["VALID_KEY1", "MISSING_KEY1", "MISSING_KEY2"])
        
        assert "Configuration validation failed" in str(exc_info.value)
        assert "Missing keys: MISSING_KEY1, MISSING_KEY2" in str(exc_info.value)
        assert "Configuration validation failed" in caplog.text
    
    def test_validate_config_empty_keys(self, caplog):
        """Test validation when some keys are empty."""
        with pytest.raises(ConfigError) as exc_info:
            validate_config(["VALID_KEY1", "EMPTY_KEY"])
        
        assert "Configuration validation failed" in str(exc_info.value)
        assert "Empty keys: EMPTY_KEY" in str(exc_info.value)
        assert "Configuration validation failed" in caplog.text
    
    def test_validate_config_whitespace_key(self):
        """Test validation treats whitespace-only values as empty."""
        with pytest.raises(ConfigError) as exc_info:
            validate_config(["WHITESPACE_KEY"])
        
        assert "Empty keys: WHITESPACE_KEY" in str(exc_info.value)
    
    def test_validate_config_missing_and_empty(self, caplog):
        """Test validation when both missing and empty keys exist."""
        with pytest.raises(ConfigError) as exc_info:
            validate_config(["VALID_KEY1", "MISSING_KEY", "EMPTY_KEY"])
        
        error_msg = str(exc_info.value)
        assert "Configuration validation failed" in error_msg
        assert "Missing keys: MISSING_KEY" in error_msg
        assert "Empty keys: EMPTY_KEY" in error_msg
    
    def test_validate_config_empty_list(self, caplog):
        """Test validation with empty required keys list."""
        with caplog.at_level(logging.INFO):
            validate_config([])
        
        assert "Configuration validation passed for keys:" in caplog.text


class TestConfigError:
    """Test cases for ConfigError exception."""
    
    def test_config_error_inheritance(self):
        """Test that ConfigError inherits from Exception."""
        assert issubclass(ConfigError, Exception)
    
    def test_config_error_message(self):
        """Test that ConfigError can store and retrieve message."""
        message = "Test configuration error"
        error = ConfigError(message)
        assert str(error) == message
    
    def test_config_error_empty_message(self):
        """Test ConfigError with empty message."""
        error = ConfigError()
        assert str(error) == ""
