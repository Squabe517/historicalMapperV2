"""
Logging utilities for Historical ePub Map Enhancer.

Provides centralized logging configuration and convenience methods
for consistent logging across the application.
"""

import logging
import os
from pathlib import Path
from typing import Optional


# Flag to track if logger has been initialized to ensure idempotency
_logger_initialized = False


def initialize_logger(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """
    Initialize the root logger with console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    global _logger_initialized
    
    # Ensure idempotency - don't re-initialize if already done
    if _logger_initialized:
        return
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger and set level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear any existing handlers to prevent duplicates
    root_logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Mark as initialized
    _logger_initialized = True
    
    # Log initialization success
    root_logger.info(f"Logger initialized with level {log_level}, file: {log_file}")


def log_info(message: str) -> None:
    """
    Log an info message.
    
    Args:
        message: Message to log
    """
    logger = logging.getLogger()
    logger.info(message)


def log_warning(message: str) -> None:
    """
    Log a warning message.
    
    Args:
        message: Message to log
    """
    logger = logging.getLogger()
    logger.warning(message)


def log_error(message: str) -> None:
    """
    Log an error message.
    
    Args:
        message: Message to log
    """
    logger = logging.getLogger()
    logger.error(message)
