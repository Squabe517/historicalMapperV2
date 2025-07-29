"""
Exception classes for the embedder module.

Provides specific exceptions for different failure modes when
embedding maps into EPUB files.
"""


class EmbedderError(Exception):
    """Base exception for embedder module."""
    pass


class ParagraphNotFoundError(EmbedderError):
    """Target paragraph cannot be located in EPUB."""
    pass


class InvalidEpubStructureError(EmbedderError):
    """EPUB structure prevents embedding."""
    pass


class ImageEmbedError(EmbedderError):
    """Failed to embed image into EPUB."""
    pass
