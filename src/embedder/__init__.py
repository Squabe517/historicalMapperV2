"""
Embedder module for the Historical ePub Map Enhancer.

This module provides functionality for:
- Embedding map images into EPUB files at appropriate locations
- Supporting multiple embedding strategies (external files vs inline Base64)
- Maintaining EPUB structural integrity
- Direct integration with AI and mapping module outputs

Main classes:
- EpubMapEmbedder: Main embedder that coordinates the embedding process
- EmbedderConfig: Configuration for embedding behavior
- ImageEmbedStrategy: Abstract base for embedding strategies
- ExternalImageStrategy: Embeds images as external files
- InlineImageStrategy: Embeds images as Base64 data URIs

Errors:
- EmbedderError: Base exception for embedder module
- ParagraphNotFoundError: Target paragraph cannot be located
- InvalidEpubStructureError: EPUB structure prevents embedding
- ImageEmbedError: Failed to embed image
"""

from .embedder_config import EmbedderConfig
from .embedder_core import EpubMapEmbedder
from .embedder_errors import (
    EmbedderError,
    ParagraphNotFoundError,
    InvalidEpubStructureError,
    ImageEmbedError
)
from .embedder_strategy import (
    ImageEmbedStrategy,
    ExternalImageStrategy,
    InlineImageStrategy
)

__all__ = [
    # Main classes
    "EpubMapEmbedder",
    "EmbedderConfig",
    
    # Strategies
    "ImageEmbedStrategy",
    "ExternalImageStrategy",
    "InlineImageStrategy",
    
    # Errors
    "EmbedderError",
    "ParagraphNotFoundError",
    "InvalidEpubStructureError",
    "ImageEmbedError",
]

# Version info
__version__ = "1.0.0"
__author__ = "Historical ePub Map Enhancer Team"