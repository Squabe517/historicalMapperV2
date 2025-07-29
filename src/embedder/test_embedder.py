"""
Test suite for the embedder module.

Tests configuration, strategies, and core embedding functionality
with mock EPUB structures.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from lxml import etree
import ebooklib
from ebooklib import epub

from .embedder_config import EmbedderConfig
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
from .embedder_core import EpubMapEmbedder


class TestEmbedderConfig:
    """Test embedder configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbedderConfig()
        assert config.figure_class == "historical-map"
        assert config.figure_style == "margin: 1em 0; text-align: center;"
        assert config.caption_template == "Map of {place}"
        assert config.image_format == "png"
        assert config.max_image_width == "100%"
        assert config.embed_strategy == "external"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbedderConfig(
            figure_class="custom-map",
            caption_template="Historical map: {place}",
            embed_strategy="inline"
        )
        assert config.figure_class == "custom-map"
        assert config.caption_template == "Historical map: {place}"
        assert config.embed_strategy == "inline"
    
    def test_invalid_embed_strategy(self):
        """Test invalid embed strategy raises error."""
        with pytest.raises(ValueError) as exc_info:
            EmbedderConfig(embed_strategy="invalid")
        assert "Invalid embed_strategy" in str(exc_info.value)
    
    def test_empty_caption_template(self):
        """Test empty caption template raises error."""
        with pytest.raises(ValueError) as exc_info:
            EmbedderConfig(caption_template="")
        assert "caption_template cannot be empty" in str(exc_info.value)
    
    def test_caption_without_placeholder(self):
        """Test caption without {place} placeholder raises error."""
        with pytest.raises(ValueError) as exc_info:
            EmbedderConfig(caption_template="Just a map")
        assert "must contain {place} placeholder" in str(exc_info.value)


class TestExternalImageStrategy:
    """Test external image embedding strategy."""
    
    def test_embed_image(self):
        """Test embedding image as external file."""
        strategy = ExternalImageStrategy()
        book = Mock(spec=epub.EpubBook)
        
        cache_key = "Rome_abc123.png"
        image_bytes = b"fake_image_data"
        
        href = strategy.embed_image(book, cache_key, image_bytes)
        
        assert href == f"images/{cache_key}"
        
        # Verify image was added to book
        book.add_item.assert_called_once()
        img = book.add_item.call_args[0][0]
        assert img.file_name == f"images/{cache_key}"
        assert img.content == image_bytes
        assert img.media_type == "image/png"
    
    def test_embed_jpeg_image(self):
        """Test embedding JPEG image."""
        strategy = ExternalImageStrategy()
        book = Mock(spec=epub.EpubBook)
        
        cache_key = "Paris_xyz789.jpg"
        image_bytes = b"fake_jpeg_data"
        
        href = strategy.embed_image(book, cache_key, image_bytes)
        
        img = book.add_item.call_args[0][0]
        assert img.media_type == "image/jpeg"
    
    def test_create_figure_element(self):
        """Test creating figure element."""
        strategy = ExternalImageStrategy()
        config = EmbedderConfig()
        
        figure = strategy.create_figure_element(
            "images/Rome_abc123.png",
            "Rome",
            config
        )
        
        # Check figure element
        assert figure.tag == "figure"
        assert figure.get("class") == "historical-map"
        assert figure.get("style") == "margin: 1em 0; text-align: center;"
        
        # Check image
        img = figure.find(".//img")
        assert img is not None
        assert img.get("src") == "../images/Rome_abc123.png"
        assert img.get("alt") == "Map of Rome"
        assert "max-width: 100%" in img.get("style")
        
        # Check caption
        figcaption = figure.find(".//figcaption")
        assert figcaption is not None
        assert figcaption.text == "Map of Rome"


class TestInlineImageStrategy:
    """Test inline Base64 image embedding strategy."""
    
    def test_embed_image(self):
        """Test creating Base64 data URI."""
        strategy = InlineImageStrategy()
        book = Mock(spec=epub.EpubBook)
        
        cache_key = "Venice_def456.png"
        image_bytes = b"fake_image_data"
        
        href = strategy.embed_image(book, cache_key, image_bytes)
        
        assert href.startswith("data:image/png;base64,")
        assert "ZmFrZV9pbWFnZV9kYXRh" in href  # Base64 of "fake_image_data"
        
        # Should not add to book
        book.add_item.assert_not_called()
    
    def test_create_figure_element_inline(self):
        """Test creating figure with inline image."""
        strategy = InlineImageStrategy()
        config = EmbedderConfig()
        
        data_uri = "data:image/png;base64,ABC123"
        figure = strategy.create_figure_element(
            data_uri,
            "Venice",
            config
        )
        
        # Check image src is the data URI
        img = figure.find(".//img")
        assert img.get("src") == data_uri


class TestEpubMapEmbedder:
    """Test main embedder functionality."""
    
    @pytest.fixture
    def mock_book(self):
        """Create a mock EPUB book."""
        book = Mock(spec=epub.EpubBook)
        book.spine = [("item1", ""), ("item2", "")]
        
        # Create mock items with XHTML content
        item1 = Mock(spec=epub.EpubItem)
        item1.file_name = "chapter1.xhtml"
        item1.content = b"""<?xml version="1.0" encoding="utf-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml">
        <body>
            <p>In ancient Rome, the forum was busy.</p>
            <p>Constantinople was the capital of Byzantine Empire.</p>
        </body>
        </html>"""
        item1.get_type.return_value = ebooklib.ITEM_DOCUMENT
        
        item2 = Mock(spec=epub.EpubItem)
        item2.file_name = "chapter2.xhtml"
        item2.content = b"""<?xml version="1.0" encoding="utf-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml">
        <body>
            <p>Venice was a major maritime power.</p>
        </body>
        </html>"""
        item2.get_type.return_value = ebooklib.ITEM_DOCUMENT
        
        book.get_item_with_id.side_effect = lambda x: item1 if x == "item1" else item2
        book.get_items_of_type.return_value = [item1, item2]
        
        return book
    
    def test_initialization_default(self):
        """Test embedder initialization with defaults."""
        embedder = EpubMapEmbedder()
        assert isinstance(embedder.config, EmbedderConfig)
        assert isinstance(embedder.strategy, ExternalImageStrategy)
    
    def test_initialization_inline_strategy(self):
        """Test embedder with inline strategy."""
        config = EmbedderConfig(embed_strategy="inline")
        embedder = EpubMapEmbedder(config=config)
        assert isinstance(embedder.strategy, InlineImageStrategy)
    
    def test_validate_epub_structure_valid(self, mock_book):
        """Test validation of valid EPUB structure."""
        embedder = EpubMapEmbedder()
        # Should not raise
        embedder.validate_epub_structure(mock_book)
    
    def test_validate_epub_no_documents(self):
        """Test validation fails with no documents."""
        book = Mock(spec=epub.EpubBook)
        book.get_items_of_type.return_value = []
        
        embedder = EpubMapEmbedder()
        with pytest.raises(InvalidEpubStructureError) as exc_info:
            embedder.validate_epub_structure(book)
        assert "No XHTML documents" in str(exc_info.value)
    
    def test_validate_epub_no_spine(self):
        """Test validation fails with no spine."""
        book = Mock(spec=epub.EpubBook)
        book.spine = []
        book.get_items_of_type.return_value = [Mock()]
        
        embedder = EpubMapEmbedder()
        with pytest.raises(InvalidEpubStructureError) as exc_info:
            embedder.validate_epub_structure(book)
        assert "no spine" in str(exc_info.value)
    
    def test_build_paragraph_index(self, mock_book):
        """Test building paragraph index."""
        embedder = EpubMapEmbedder()
        embedder._build_paragraph_index(mock_book)
        
        # Should have 3 paragraphs total
        assert len(embedder._paragraph_cache) == 3
        assert 0 in embedder._paragraph_cache
        assert 2 in embedder._paragraph_cache
        
        # Check paragraph text
        assert "ancient Rome" in embedder._paragraph_cache[0]['text']
        assert "Constantinople" in embedder._paragraph_cache[1]['text']
        assert "Venice" in embedder._paragraph_cache[2]['text']
    
    def test_find_cache_key_exact_match(self):
        """Test finding cache key with exact match."""
        embedder = EpubMapEmbedder()
        
        place_info = {"place": "Rome", "zoom": 12}
        map_images = {
            "Rome_abc123.png": b"data1",
            "Venice_def456.png": b"data2"
        }
        
        cache_key = embedder._find_cache_key(place_info, map_images)
        assert cache_key == "Rome_abc123.png"
    
    def test_find_cache_key_fuzzy_match(self):
        """Test finding cache key with fuzzy match."""
        embedder = EpubMapEmbedder()
        
        place_info = {"place": "São Paulo", "zoom": 12}
        map_images = {
            "S_o_Paulo_abc123.png": b"data1",
            "Venice_def456.png": b"data2"
        }
        
        cache_key = embedder._find_cache_key(place_info, map_images)
        assert cache_key == "S_o_Paulo_abc123.png"
    
    def test_find_cache_key_no_match(self):
        """Test finding cache key with no match."""
        embedder = EpubMapEmbedder()
        
        place_info = {"place": "Athens", "zoom": 12}
        map_images = {
            "Rome_abc123.png": b"data1",
            "Venice_def456.png": b"data2"
        }
        
        cache_key = embedder._find_cache_key(place_info, map_images)
        assert cache_key is None
    
    @patch('src.embedder.embedder_core.log_info')
    @patch('src.embedder.embedder_core.log_warning')
    def test_embed_maps_success(self, mock_log_warning, mock_log_info, mock_book):
        """Test successful embedding of maps."""
        embedder = EpubMapEmbedder()
        
        ai_results = [
            [{"place": "Rome", "zoom": 12}],
            [{"place": "Venice", "zoom": 13}]
        ]
        
        map_images = {
            "Rome_abc123.png": b"rome_image_data",
            "Venice_def456.png": b"venice_image_data"
        }
        
        result = embedder.embed_maps(mock_book, ai_results, map_images)
        
        assert result == mock_book
        # Check that images were embedded
        assert mock_book.add_item.call_count == 2
    
    def test_embed_maps_missing_image(self, mock_book):
        """Test handling of missing map image."""
        embedder = EpubMapEmbedder()
        
        ai_results = [
            [{"place": "Rome", "zoom": 12}],
            [{"place": "Athens", "zoom": 13}]  # No image for Athens
        ]
        
        map_images = {
            "Rome_abc123.png": b"rome_image_data"
        }
        
        # Should not raise, just skip Athens
        result = embedder.embed_maps(mock_book, ai_results, map_images)
        
        # Only Rome should be embedded
        assert mock_book.add_item.call_count == 1
    
    def test_chunk_to_paragraph_index_with_mapping(self):
        """Test chunk to paragraph conversion with explicit mapping."""
        embedder = EpubMapEmbedder()
        embedder._chunk_to_para_map = {
            0: (0, 4),
            1: (5, 9),
            2: (10, 14)
        }
        embedder._paragraph_cache = {i: {} for i in range(15)}
        
        idx = embedder._chunk_to_paragraph_index(1, {"place": "Rome"})
        assert idx == 9  # End of chunk 1
    
    def test_chunk_to_paragraph_index_estimated(self):
        """Test chunk to paragraph conversion with estimation."""
        embedder = EpubMapEmbedder()
        embedder._chunk_to_para_map = {}  # No explicit mapping
        embedder._paragraph_cache = {i: {} for i in range(20)}
        
        idx = embedder._chunk_to_paragraph_index(2, {"place": "Rome"})
        # Should estimate based on distribution
        assert 0 <= idx < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
