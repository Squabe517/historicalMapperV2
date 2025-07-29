"""
Text chunking utilities for the Historical ePub Map Enhancer.

Splits extracted text into optimal chunks for AI processing while
maintaining sentence boundaries for better context preservation.

Supports both plain text and DOM-based processing.
"""

import re
import logging
from typing import List, Tuple, Union, Dict, Any
from dataclasses import dataclass

from .document_parser import ParagraphElement


@dataclass
class ChunkedParagraph:
    """Represents a chunk with its source paragraph information."""
    text: str                           # The chunk text
    paragraph_element: ParagraphElement # Source paragraph element
    chunk_index: int                    # Index within the paragraph (0 if not split)
    is_split: bool                      # True if paragraph was split into multiple chunks


class TextChunker:
    """Chunks text into optimal sizes for AI processing."""
    
    def __init__(self):
        """Initialize the text chunker."""
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, paragraphs: List[str], chunk_size: int = 1000) -> List[str]:
        """
        Split paragraphs into chunks suitable for AI processing.
        
        Each paragraph that fits within chunk_size is kept as its own chunk.
        Only paragraphs exceeding chunk_size are split.
        
        Args:
            paragraphs: List of paragraph strings
            chunk_size: Maximum characters per chunk (default: 1000)
            
        Returns:
            List of text chunks, with paragraph boundaries preserved when possible
        """
        if not paragraphs:
            self.logger.info("No paragraphs to chunk")
            return []
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        chunks = []
        
        for paragraph in paragraphs:
            # Clean and validate paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph fits within chunk size, keep it as is
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
            else:
                # Split large paragraph into sentence-based chunks
                paragraph_chunks = self._split_paragraph_by_sentences(paragraph, chunk_size)
                chunks.extend(paragraph_chunks)
        
        self.logger.info(f"Split {len(paragraphs)} paragraphs into {len(chunks)} chunks")
        return chunks
    
    def chunk_paragraph_elements(self, paragraph_elements: List[ParagraphElement], 
                               chunk_size: int = 1000) -> List[ChunkedParagraph]:
        """
        Split paragraph elements into chunks while maintaining DOM references.
        
        Args:
            paragraph_elements: List of ParagraphElement objects
            chunk_size: Maximum characters per chunk (default: 1000)
            
        Returns:
            List of ChunkedParagraph objects with DOM references preserved
        """
        if not paragraph_elements:
            self.logger.info("No paragraph elements to chunk")
            return []
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        chunked_paragraphs = []
        
        for para_elem in paragraph_elements:
            text = para_elem.text.strip()
            if not text:
                continue
            
            # If paragraph fits within chunk size, keep it as is
            if len(text) <= chunk_size:
                chunked = ChunkedParagraph(
                    text=text,
                    paragraph_element=para_elem,
                    chunk_index=0,
                    is_split=False
                )
                chunked_paragraphs.append(chunked)
            else:
                # Split large paragraph into sentence-based chunks
                text_chunks = self._split_paragraph_by_sentences(text, chunk_size)
                for idx, chunk_text in enumerate(text_chunks):
                    chunked = ChunkedParagraph(
                        text=chunk_text,
                        paragraph_element=para_elem,
                        chunk_index=idx,
                        is_split=True
                    )
                    chunked_paragraphs.append(chunked)
        
        self.logger.info(f"Split {len(paragraph_elements)} paragraph elements into "
                        f"{len(chunked_paragraphs)} chunks")
        return chunked_paragraphs
    
    def chunk_text_with_mapping(self, paragraphs: List[str], chunk_size: int = 1000) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split paragraphs into chunks and return mapping information.
        
        This variant returns both chunks and mapping of chunks to original paragraphs,
        which is useful for the embedder to know which paragraph each chunk came from.
        
        Args:
            paragraphs: List of paragraph strings
            chunk_size: Maximum characters per chunk (default: 1000)
            
        Returns:
            Tuple of (chunks, chunk_info) where chunk_info is a list of
            (start_para_idx, end_para_idx) tuples for each chunk
        """
        if not paragraphs:
            self.logger.info("No paragraphs to chunk")
            return [], []
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        chunks = []
        chunk_info = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Clean and validate paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph fits within chunk size, keep it as is
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
                chunk_info.append((para_idx, para_idx))  # Single paragraph chunk
            else:
                # Split large paragraph into sentence-based chunks
                paragraph_chunks = self._split_paragraph_by_sentences(paragraph, chunk_size)
                for chunk in paragraph_chunks:
                    chunks.append(chunk)
                    chunk_info.append((para_idx, para_idx))  # All sub-chunks map to same paragraph
        
        self.logger.info(f"Split {len(paragraphs)} paragraphs into {len(chunks)} chunks with mapping")
        return chunks, chunk_info
    
    def aggregate_results_by_paragraph(self, chunked_paragraphs: List[ChunkedParagraph],
                                     chunk_results: List[List[Dict[str, Any]]]) -> Dict[ParagraphElement, List[Dict[str, Any]]]:
        """
        Aggregate AI results back to their source paragraphs.
        
        Args:
            chunked_paragraphs: List of ChunkedParagraph objects
            chunk_results: AI results for each chunk
            
        Returns:
            Dictionary mapping ParagraphElement to list of all results for that paragraph
        """
        if len(chunked_paragraphs) != len(chunk_results):
            raise ValueError("Number of chunks and results must match")
        
        aggregated = {}
        
        for chunked_para, results in zip(chunked_paragraphs, chunk_results):
            para_elem = chunked_para.paragraph_element
            
            if para_elem not in aggregated:
                aggregated[para_elem] = []
            
            aggregated[para_elem].extend(results)
        
        return aggregated
    
    def _split_paragraph_by_sentences(self, paragraph: str, chunk_size: int) -> List[str]:
        """
        Split a large paragraph into chunks at sentence boundaries.
        
        Args:
            paragraph: The paragraph to split
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of chunks split at sentence boundaries
        """
        sentences = self._split_into_sentences(paragraph)
        
        if not sentences:
            return [paragraph[:chunk_size]] if paragraph else []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            
            # Handle sentences that are themselves too long
            if len(current_chunk) > chunk_size:
                if current_chunk == sentence:  # Single long sentence
                    # Force split the sentence
                    word_chunks = self._split_sentence_by_words(sentence, chunk_size)
                    chunks.extend(word_chunks[:-1])
                    current_chunk = word_chunks[-1] if word_chunks else ""
                else:
                    # This shouldn't happen with our logic, but handle gracefully
                    chunks.append(current_chunk)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [paragraph[:chunk_size]]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Enhanced sentence boundary detection
        # Handles common abbreviations and edge cases
        sentence_endings = r'[.!?]+(?:\s|$)'
        
        # Split on sentence endings but keep the punctuation
        sentences = re.split(f'({sentence_endings})', text)
        
        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        # If no sentence endings found, return original text as single sentence
        return result if result else [text]
    
    def _split_sentence_by_words(self, sentence: str, chunk_size: int) -> List[str]:
        """
        Split a very long sentence by words when sentence boundaries fail.
        
        Args:
            sentence: The sentence to split
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of word-based chunks
        """
        words = sentence.split()
        
        if not words:
            return [sentence[:chunk_size]] if sentence else []
        
        chunks = []
        current_chunk = ""
        
        for word in words:
            # Check if adding this word would exceed chunk size
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle very long words
                if len(word) > chunk_size:
                    # Force split the word
                    chunks.append(word[:chunk_size])
                    current_chunk = word[chunk_size:]
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [sentence[:chunk_size]]
    
    def get_chunk_stats(self, chunks: Union[List[str], List[ChunkedParagraph]]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of text chunks or ChunkedParagraph objects
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "split_paragraphs": 0
            }
        
        # Handle both text chunks and ChunkedParagraph objects
        if isinstance(chunks[0], str):
            chunk_sizes = [len(chunk) for chunk in chunks]
            split_paragraphs = 0
        else:
            chunk_sizes = [len(chunk.text) for chunk in chunks]
            split_paragraphs = sum(1 for chunk in chunks if chunk.is_split and chunk.chunk_index == 0)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "split_paragraphs": split_paragraphs
        }