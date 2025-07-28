"""
PDF document parser stub implementation.

This is a placeholder implementation for future PDF parsing capability.
When implemented, it will likely use pdfminer.six or PyMuPDF for text extraction.
"""

from typing import List, Dict, Any

from .document_parser import DocumentParser


class PdfParser(DocumentParser):
    """
    Stub parser for PDF document format.
    
    This class serves as a placeholder for future PDF parsing functionality.
    All methods raise NotImplementedError to indicate that PDF parsing
    is not yet supported.
    
    Future implementation will likely use:
    - pdfminer.six for robust text extraction
    - PyMuPDF (fitz) for fast processing and metadata
    """
    
    def load_file(self, file_path: str) -> None:
        """
        Load a PDF file for processing.
        
        Args:
            file_path: Path to the PDF file
            
        Raises:
            NotImplementedError: PDF parsing not yet supported
        """
        raise NotImplementedError("PDF parsing not yet supported")
    
    def extract_text(self) -> List[str]:
        """
        Extract text content from the loaded PDF.
        
        Returns:
            List of text paragraphs/sections in document order
            
        Raises:
            NotImplementedError: PDF parsing not yet supported
        """
        raise NotImplementedError("PDF parsing not yet supported")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the loaded PDF.
        
        Returns:
            Dictionary containing PDF metadata
            
        Raises:
            NotImplementedError: PDF parsing not yet supported
        """
        raise NotImplementedError("PDF parsing not yet supported")
