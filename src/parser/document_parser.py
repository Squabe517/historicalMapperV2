"""
Abstract base class for document parsers.

Defines the interface that all document parsers must implement
for the Historical ePub Map Enhancer project.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


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
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the loaded document.
        
        Returns:
            Dictionary containing document metadata
            
        Raises:
            ParserError: If no document is loaded or metadata extraction fails
        """
        pass


class ParserError(Exception):
    """Custom exception for document parsing errors."""
    pass
