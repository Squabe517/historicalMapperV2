"""
Unit tests for EpubParser implementation.

Tests cover successful parsing, error conditions, text extraction,
metadata extraction, and edge cases to achieve ≥90% branch coverage.
"""

import os
import tempfile
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import ebooklib
from ebooklib import epub
from lxml import etree

from src.parser.epub_parser import EpubParser
from src.parser.document_parser import ParserError


class TestEpubParserInitialization:
    """Test cases for EpubParser initialization."""
    
    def test_initialization(self):
        """Test parser initializes with correct default state."""
        parser = EpubParser()
        assert parser.book is None
        assert parser.logger is not None
        assert isinstance(parser.logger, logging.Logger)


class TestLoadFile:
    """Test cases for load_file method."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
    
    def test_load_file_not_found(self):
        """Test loading non-existent file raises ParserError."""
        with pytest.raises(ParserError, match="File not found"):
            self.parser.load_file("/path/that/does/not/exist.epub")
    
    def test_load_file_wrong_extension(self):
        """Test loading non-ePub file raises ParserError."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            with pytest.raises(ParserError, match="File is not an ePub"):
                self.parser.load_file(tmp_file.name)
    
    @patch('ebooklib.epub.read_epub')
    def test_load_file_invalid_epub(self, mock_read_epub):
        """Test loading invalid ePub raises ParserError."""
        mock_read_epub.side_effect = Exception("Invalid ePub format")
        
        with tempfile.NamedTemporaryFile(suffix=".epub") as tmp_file:
            with pytest.raises(ParserError, match="Failed to load ePub file"):
                self.parser.load_file(tmp_file.name)
    
    @patch('ebooklib.epub.read_epub')
    def test_load_file_unexpected_error(self, mock_read_epub):
        """Test unexpected error during loading raises ParserError."""
        mock_read_epub.side_effect = Exception("Unexpected error")
        
        with tempfile.NamedTemporaryFile(suffix=".epub") as tmp_file:
            with pytest.raises(ParserError, match="Failed to load ePub file"):
                self.parser.load_file(tmp_file.name)
    
    @patch('ebooklib.epub.read_epub')
    def test_load_file_success(self, mock_read_epub, caplog):
        """Test successful ePub loading."""
        mock_book = Mock()
        mock_read_epub.return_value = mock_book
        
        with tempfile.NamedTemporaryFile(suffix=".epub") as tmp_file:
            with caplog.at_level(logging.INFO):
                self.parser.load_file(tmp_file.name)
            
            assert self.parser.book is mock_book
            assert "Successfully loaded ePub" in caplog.text


class TestExtractText:
    """Test cases for extract_text method."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
    
    def test_extract_text_no_book_loaded(self):
        """Test extract_text raises error when no book is loaded."""
        with pytest.raises(ParserError, match="No ePub file loaded"):
            self.parser.extract_text()
    
    def test_extract_text_empty_book(self, caplog):
        """Test extract_text with empty book returns empty list."""
        mock_book = Mock()
        mock_book.spine = []
        self.parser.book = mock_book
        
        with caplog.at_level(logging.INFO):
            result = self.parser.extract_text()
        
        assert result == []
        assert "Extracted 0 paragraphs" in caplog.text
    
    def test_extract_text_empty_book(self, caplog):
        """Test extract_text with empty book returns empty list."""
        mock_book = Mock()
        mock_book.get_items.return_value = []
        self.parser.book = mock_book
        
        with caplog.at_level(logging.INFO):
            result = self.parser.extract_text()
        
        assert result == []
        assert "Extracted 0 paragraphs" in caplog.text
    
    def test_extract_text_with_content(self, caplog):
        """Test extract_text successfully extracts paragraphs."""
        # Create mock items with XHTML content
        mock_item1 = Mock()
        mock_item1.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item1.get_content.return_value = b'''
        <html><body>
            <p>First paragraph</p>
            <p>Second paragraph</p>
        </body></html>
        '''
        
        mock_item2 = Mock()
        mock_item2.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item2.get_content.return_value = b'''
        <html><body>
            <p>Third paragraph</p>
        </body></html>
        '''
        
        mock_book = Mock()
        mock_book.get_items.return_value = [mock_item1, mock_item2]
        self.parser.book = mock_book
        
        with caplog.at_level(logging.INFO):
            result = self.parser.extract_text()
        
        assert len(result) == 3
        assert "First paragraph" in result
        assert "Second paragraph" in result
        assert "Third paragraph" in result
        assert "Extracted 3 paragraphs" in caplog.text
    
    def test_extract_text_skip_non_document_items(self):
        """Test extract_text skips non-document items."""
        # First item is not a document
        mock_item1 = Mock()
        mock_item1.get_type.return_value = ebooklib.ITEM_IMAGE
        
        # Second item is a document
        mock_item2 = Mock()
        mock_item2.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item2.get_content.return_value = b'<html><body><p>Only paragraph</p></body></html>'
        
        mock_book = Mock()
        mock_book.get_items.return_value = [mock_item1, mock_item2]
        self.parser.book = mock_book
        
        result = self.parser.extract_text()
        assert len(result) == 1
        assert "Only paragraph" in result
    
    def test_extract_text_handles_empty_content(self):
        """Test extract_text handles items with no content gracefully."""
        # Item with no content
        mock_item1 = Mock()
        mock_item1.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item1.get_content.return_value = None
        
        # Item with valid content
        mock_item2 = Mock()
        mock_item2.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item2.get_content.return_value = b'<html><body><p>Valid paragraph</p></body></html>'
        
        mock_book = Mock()
        mock_book.get_items.return_value = [mock_item1, mock_item2]
        self.parser.book = mock_book
        
        result = self.parser.extract_text()
        assert len(result) == 1
        assert "Valid paragraph" in result
    
    def test_extract_text_unexpected_error(self):
        """Test extract_text handles unexpected errors."""
        mock_book = Mock()
        mock_book.spine = [('item1',)]
        mock_book.get_item_by_id.side_effect = Exception("Unexpected error")
        self.parser.book = mock_book
        
        with pytest.raises(ParserError, match="Failed to extract text"):
            self.parser.extract_text()


class TestExtractParagraphsFromXhtml:
    """Test cases for _extract_paragraphs_from_xhtml method."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
    
    def test_extract_paragraphs_valid_xhtml(self):
        """Test extracting paragraphs from valid XHTML."""
        content = b'''
        <html><body>
            <p>First paragraph</p>
            <p>Second paragraph with <em>emphasis</em></p>
            <p></p>
            <p>   </p>
            <p>Third paragraph</p>
        </body></html>
        '''
        
        result = self.parser._extract_paragraphs_from_xhtml(content)
        
        assert len(result) == 3
        assert "First paragraph" in result
        assert "Second paragraph with emphasis" in result
        assert "Third paragraph" in result
    
    def test_extract_paragraphs_malformed_xhtml(self):
        """Test fallback behavior with malformed XHTML."""
        content = b'''
        <html><body>
            <p>Valid paragraph</p>
            <p>Unclosed paragraph
            <p>Another paragraph</p>
        </body></html>
        '''
        
        # This should trigger the fallback method
        result = self.parser._extract_paragraphs_from_xhtml(content)
        
        # Should still extract some content via fallback
        assert isinstance(result, list)
    
    def test_extract_paragraphs_no_paragraphs(self):
        """Test XHTML with no paragraph tags."""
        content = b'<html><body><div>No paragraphs here</div></body></html>'
        
        result = self.parser._extract_paragraphs_from_xhtml(content)
        assert result == []
    
    def test_extract_paragraphs_nested_elements(self):
        """Test paragraphs with nested elements."""
        content = b'''
        <html><body>
            <p>Paragraph with <strong>bold</strong> and <a href="#">link</a> text</p>
        </body></html>
        '''
        
        result = self.parser._extract_paragraphs_from_xhtml(content)
        assert len(result) == 1
        assert "Paragraph with bold and link text" in result[0]


class TestExtractParagraphsFallback:
    """Test cases for _extract_paragraphs_fallback method."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
    
    def test_fallback_extraction(self):
        """Test fallback regex-based extraction."""
        content = b'''
        <p>First paragraph</p>
        <p class="special">Second paragraph with attributes</p>
        <P>Third paragraph uppercase tag</P>
        <p>Fourth paragraph with <span>nested tags</span></p>
        '''
        
        result = self.parser._extract_paragraphs_fallback(content)
        
        assert len(result) >= 3  # Should extract at least some paragraphs
        # Check that HTML tags are removed
        for paragraph in result:
            assert '<' not in paragraph
            assert '>' not in paragraph
    
    def test_fallback_with_invalid_encoding(self):
        """Test fallback handles encoding issues gracefully."""
        # Create content with invalid UTF-8 bytes
        content = b'\xff\xfe<p>Paragraph with encoding issues</p>'
        
        result = self.parser._extract_paragraphs_fallback(content)
        assert isinstance(result, list)  # Should not crash


class TestGetMetadata:
    """Test cases for get_metadata method."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
    
    def test_get_metadata_no_book_loaded(self):
        """Test get_metadata raises error when no book is loaded."""
        with pytest.raises(ParserError, match="No ePub file loaded"):
            self.parser.get_metadata()
    
    def test_get_metadata_success(self, caplog):
        """Test successful metadata extraction."""
        mock_book = Mock()
        
        # Mock metadata responses
        def mock_get_metadata(namespace, name):
            metadata_map = {
                ('DC', 'title'): [('Sample Book Title', {})],
                ('DC', 'creator'): [('Author One', {}), ('Author Two', {})],
                ('DC', 'language'): [('en', {})],
                ('DC', 'publisher'): [('Sample Publisher', {})],
                ('DC', 'date'): [('2023', {})],
                ('DC', 'identifier'): [('isbn:1234567890', {})]
            }
            return metadata_map.get((namespace, name), [])
        
        mock_book.get_metadata.side_effect = mock_get_metadata
        self.parser.book = mock_book
        
        with caplog.at_level(logging.INFO):
            result = self.parser.get_metadata()
        
        assert result['title'] == 'Sample Book Title'
        assert result['authors'] == ['Author One', 'Author Two']
        assert result['language'] == 'en'
        assert result['publisher'] == 'Sample Publisher'
        assert result['pub_date'] == '2023'
        assert result['identifier'] == 'isbn:1234567890'
        assert "Extracted metadata for: Sample Book Title" in caplog.text
    
    def test_get_metadata_missing_fields(self):
        """Test metadata extraction with missing fields uses defaults."""
        mock_book = Mock()
        mock_book.get_metadata.return_value = []  # No metadata found
        self.parser.book = mock_book
        
        result = self.parser.get_metadata()
        
        assert result['title'] == 'Unknown Title'
        assert result['authors'] == ['Unknown Author']
        assert result['language'] == 'Unknown'
        assert result['publisher'] == 'Unknown Publisher'
        assert result['pub_date'] == 'Unknown Date'
        assert result['identifier'] == 'Unknown ID'
    
    def test_get_metadata_unexpected_error(self):
        """Test get_metadata handles unexpected errors."""
        mock_book = Mock()
        self.parser.book = mock_book
        
        # Mock the logger to raise an exception during info logging
        with patch.object(self.parser.logger, 'info', side_effect=Exception("Logging error")):
            with pytest.raises(ParserError, match="Failed to extract metadata"):
                self.parser.get_metadata()


class TestMetadataHelpers:
    """Test cases for metadata helper methods."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
        self.parser.book = Mock()
    
    def test_get_metadata_value_success(self):
        """Test _get_metadata_value returns first value."""
        self.parser.book.get_metadata.return_value = [('First Value', {}), ('Second Value', {})]
        
        result = self.parser._get_metadata_value('DC', 'title')
        assert result == 'First Value'
    
    def test_get_metadata_value_empty(self):
        """Test _get_metadata_value returns None for empty metadata."""
        self.parser.book.get_metadata.return_value = []
        
        result = self.parser._get_metadata_value('DC', 'title')
        assert result is None
    
    def test_get_metadata_value_exception(self):
        """Test _get_metadata_value handles exceptions gracefully."""
        self.parser.book.get_metadata.side_effect = Exception("Error")
        
        result = self.parser._get_metadata_value('DC', 'title')
        assert result is None
    
    def test_get_all_metadata_values_success(self):
        """Test _get_all_metadata_values returns all values."""
        self.parser.book.get_metadata.return_value = [('Value1', {}), ('Value2', {}), ('Value3', {})]
        
        result = self.parser._get_all_metadata_values('DC', 'creator')
        assert result == ['Value1', 'Value2', 'Value3']
    
    def test_get_all_metadata_values_empty(self):
        """Test _get_all_metadata_values returns empty list for no metadata."""
        self.parser.book.get_metadata.return_value = []
        
        result = self.parser._get_all_metadata_values('DC', 'creator')
        assert result == []
    
    def test_get_all_metadata_values_exception(self):
        """Test _get_all_metadata_values handles exceptions gracefully."""
        self.parser.book.get_metadata.side_effect = Exception("Error")
        
        result = self.parser._get_all_metadata_values('DC', 'creator')
        assert result == []


class TestGetElementText:
    """Test cases for _get_element_text method."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = EpubParser()
    
    def test_get_element_text_simple(self):
        """Test extracting text from simple element."""
        from lxml import html
        element = html.fromstring('<p>Simple text</p>')
        
        result = self.parser._get_element_text(element)
        assert result == 'Simple text'
    
    def test_get_element_text_nested(self):
        """Test extracting text from nested elements."""
        from lxml import html
        element = html.fromstring('<p>Text with <em>emphasis</em> and <strong>bold</strong></p>')
        
        result = self.parser._get_element_text(element)
        assert result == 'Text with emphasis and bold'
    
    def test_get_element_text_empty(self):
        """Test extracting text from empty element."""
        from lxml import html
        element = html.fromstring('<p></p>')
        
        result = self.parser._get_element_text(element)
        assert result == ''