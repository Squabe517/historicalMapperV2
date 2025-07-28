"""
Unit tests for PDF parser stub implementation.

Tests verify that all methods raise NotImplementedError as expected
for the placeholder implementation.
"""

import pytest

from src.parser.pdf_parser import PdfParser


class TestPdfParserStub:
    """Test cases for PDF parser stub implementation."""
    
    def setup_method(self):
        """Set up test parser instance."""
        self.parser = PdfParser()
    
    def test_load_file_raises_not_implemented(self):
        """Test that load_file raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="PDF parsing not yet supported"):
            self.parser.load_file("test.pdf")
    
    def test_extract_text_raises_not_implemented(self):
        """Test that extract_text raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="PDF parsing not yet supported"):
            self.parser.extract_text()
    
    def test_get_metadata_raises_not_implemented(self):
        """Test that get_metadata raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="PDF parsing not yet supported"):
            self.parser.get_metadata()
    
    def test_parser_inheritance(self):
        """Test that PdfParser inherits from DocumentParser."""
        from src.parser.document_parser import DocumentParser
        assert isinstance(self.parser, DocumentParser)
    
    def test_all_methods_fail_consistently(self):
        """Test that all methods consistently raise the same error."""
        methods = ['load_file', 'extract_text', 'get_metadata']
        
        for method_name in methods:
            method = getattr(self.parser, method_name)
            with pytest.raises(NotImplementedError, match="PDF parsing not yet supported"):
                if method_name == 'load_file':
                    method("dummy_path.pdf")
                else:
                    method()
