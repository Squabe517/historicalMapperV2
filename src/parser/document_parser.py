"""
Abstract base class for document parsers.

Defines the interface that all document parsers must implement
for the Historical ePub Map Enhancer project.

Supports both text-only extraction and DOM-based processing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ParagraphElement:
    """Represents a paragraph with its DOM context."""
    text: str                    # The paragraph text
    element: Any                 # The DOM element (e.g., lxml Element)
    container: Any               # The containing document/item
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __hash__(self):
        """Make ParagraphElement hashable based on paragraph index."""
        # Use the paragraph index as the unique identifier
        # This is guaranteed to be unique within a document
        return hash(self.metadata.get('index', id(self)))
    
    def __eq__(self, other):
        """Compare ParagraphElements by their index."""
        if not isinstance(other, ParagraphElement):
            return False
        return self.metadata.get('index') == other.metadata.get('index')


class DocumentParser(ABC):
    """Abstract base class for parsing various document formats."""
    
    @abstractmethod
    def load_file(self, file_path: str) -> None:
        """
        Load a document file for processing.
        
        Args:
            file_path: Path to the document file
            
        Raises:
            ParserError: If file cannot be loaded or is invalid
        """
        pass
    
    @abstractmethod
    def extract_text(self) -> List[str]:
        """
        Extract text content from the loaded document.
        
        Returns:
            List of text paragraphs/sections in document order
            
        Raises:
            ParserError: If no document is loaded or extraction fails
        """
        pass
    
    @abstractmethod
    def extract_paragraphs_with_dom(self) -> List[ParagraphElement]:
        """
        Extract paragraphs while maintaining DOM structure.
        
        Returns:
            List of ParagraphElement objects containing text and DOM references
            
        Raises:
            ParserError: If no document is loaded or extraction fails
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the loaded document.
        
        Returns:
            Dictionary containing document metadata
            
        Raises:
            ParserError: If no document is loaded or metadata extraction fails
        """
        pass
    
    @abstractmethod
    def update_dom_element(self, paragraph_element: ParagraphElement, 
                          new_content: Any) -> None:
        """
        Update a DOM element with new content.
        
        Args:
            paragraph_element: The paragraph element to update
            new_content: New content to insert after the paragraph
            
        Raises:
            ParserError: If update fails
        """
        pass
    
    @abstractmethod
    def save_document(self, output_path: str) -> None:
        """
        Save the modified document.
        
        Args:
            output_path: Path to save the modified document
            
        Raises:
            ParserError: If save fails
        """
        pass


class ParserError(Exception):
    """Custom exception for document parsing errors."""
    pass