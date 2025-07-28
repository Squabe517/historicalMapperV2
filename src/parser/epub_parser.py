"""
ePub document parser implementation.

Extracts text content and metadata from ePub files using ebooklib
and lxml for XHTML parsing.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import ebooklib
from ebooklib import epub
from lxml import html, etree

from .document_parser import DocumentParser, ParserError


class EpubParser(DocumentParser):
    """Parser for ePub document format."""
    
    def __init__(self):
        """Initialize the ePub parser."""
        self.book: epub.EpubBook = None
        self.logger = logging.getLogger(__name__)
    
    def load_file(self, file_path: str) -> None:
        """
        Load an ePub file for processing.
        
        Args:
            file_path: Path to the ePub file
            
        Raises:
            ParserError: If file cannot be loaded or is not a valid ePub
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise ParserError(f"File not found: {file_path}")
            
            # Validate file extension
            if not file_path.lower().endswith('.epub'):
                raise ParserError(f"File is not an ePub: {file_path}")
            
            # Load the ePub file
            self.book = epub.read_epub(file_path)
            self.logger.info(f"Successfully loaded ePub: {Path(file_path).name}")
            
        except ebooklib.epub.EpubException as e:
            raise ParserError(f"Invalid ePub file: {e}")
        except Exception as e:
            raise ParserError(f"Failed to load ePub file: {e}")
    
    def extract_text(self) -> List[str]:
        """
        Extract text paragraphs from the loaded ePub.
        
        Returns:
            List of paragraph strings in document order
            
        Raises:
            ParserError: If no document is loaded or extraction fails
        """
        if self.book is None:
            raise ParserError("No ePub file loaded. Call load_file() first.")
        
        try:
            paragraphs = []
            
            # Get all XHTML documents in reading order
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse XHTML content
                    content = item.get_content()
                    if content:
                        chapter_paragraphs = self._extract_paragraphs_from_xhtml(content)
                        paragraphs.extend(chapter_paragraphs)
            
            self.logger.info(f"Extracted {len(paragraphs)} paragraphs from ePub")
            return paragraphs
            
        except Exception as e:
            raise ParserError(f"Failed to extract text: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the loaded ePub.
        
        Returns:
            Dictionary containing ePub metadata
            
        Raises:
            ParserError: If no document is loaded or metadata extraction fails
        """
        if self.book is None:
            raise ParserError("No ePub file loaded. Call load_file() first.")
        
        try:
            metadata = {
                "title": self._get_metadata_value('DC', 'title') or "Unknown Title",
                "authors": self._get_all_metadata_values('DC', 'creator') or ["Unknown Author"],
                "language": self._get_metadata_value('DC', 'language') or "Unknown",
                "publisher": self._get_metadata_value('DC', 'publisher') or "Unknown Publisher",
                "pub_date": self._get_metadata_value('DC', 'date') or "Unknown Date",
                "identifier": self._get_metadata_value('DC', 'identifier') or "Unknown ID"
            }
            
            self.logger.info(f"Extracted metadata for: {metadata['title']}")
            return metadata
            
        except Exception as e:
            raise ParserError(f"Failed to extract metadata: {e}")
    
    def _extract_paragraphs_from_xhtml(self, content: bytes) -> List[str]:
        """
        Extract paragraph text from XHTML content.
        
        Args:
            content: Raw XHTML content bytes
            
        Returns:
            List of paragraph text strings
        """
        try:
            # Parse XHTML content
            doc = html.fromstring(content)
            
            # Extract all <p> tags
            paragraphs = []
            for p_element in doc.xpath('//p'):
                # Get text content, stripping whitespace
                text = self._get_element_text(p_element).strip()
                if text:  # Only include non-empty paragraphs
                    paragraphs.append(text)
            
            return paragraphs
            
        except etree.XMLSyntaxError:
            # Fallback for malformed XHTML
            return self._extract_paragraphs_fallback(content)
        except Exception:
            # Silent fallback for any parsing issues
            return []
    
    def _get_element_text(self, element) -> str:
        """
        Extract all text content from an element and its children.
        
        Args:
            element: lxml element
            
        Returns:
            Combined text content
        """
        return ''.join(element.itertext())
    
    def _extract_paragraphs_fallback(self, content: bytes) -> List[str]:
        """
        Fallback paragraph extraction using simple string parsing.
        
        Args:
            content: Raw XHTML content bytes
            
        Returns:
            List of paragraph text strings
        """
        try:
            # Convert to string and use simple regex-like extraction
            text = content.decode('utf-8', errors='ignore')
            
            # Simple extraction of content between <p> tags
            paragraphs = []
            import re
            p_pattern = r'<p[^>]*>(.*?)</p>'
            matches = re.findall(p_pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                # Remove HTML tags and clean up
                clean_text = re.sub(r'<[^>]+>', '', match).strip()
                if clean_text:
                    paragraphs.append(clean_text)
            
            return paragraphs
            
        except Exception:
            return []
    
    def _get_metadata_value(self, namespace: str, name: str) -> str:
        """
        Get a single metadata value from the ePub.
        
        Args:
            namespace: Metadata namespace (e.g., 'DC')
            name: Metadata name (e.g., 'title')
            
        Returns:
            Metadata value or None if not found
        """
        try:
            metadata_items = self.book.get_metadata(namespace, name)
            if metadata_items:
                return metadata_items[0][0]  # First item, first element (value)
        except Exception:
            pass
        return None
    
    def _get_all_metadata_values(self, namespace: str, name: str) -> List[str]:
        """
        Get all metadata values for a given namespace and name.
        
        Args:
            namespace: Metadata namespace (e.g., 'DC')
            name: Metadata name (e.g., 'creator')
            
        Returns:
            List of metadata values
        """
        try:
            metadata_items = self.book.get_metadata(namespace, name)
            return [item[0] for item in metadata_items] if metadata_items else []
        except Exception:
            return []
