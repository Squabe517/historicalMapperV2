"""
Text chunking utilities for the Historical ePub Map Enhancer.

Splits extracted text into optimal chunks for AI processing while
maintaining sentence boundaries for better context preservation.
"""

import re
import logging
from typing import List


class TextChunker:
    """Chunks text into optimal sizes for AI processing."""
    
    def __init__(self):
        """Initialize the text chunker."""
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, paragraphs: List[str], chunk_size: int = 1000) -> List[str]:
        """
        Split paragraphs into chunks suitable for AI processing.
        
        Args:
            paragraphs: List of paragraph strings
            chunk_size: Maximum characters per chunk (default: 1000)
            
        Returns:
            List of text chunks, split at sentence boundaries when possible
        """
        if not paragraphs:
            self.logger.info("No paragraphs to chunk")
            return []
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Clean and validate paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is small enough, try to add to current chunk
            if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle paragraph that might be larger than chunk_size
                if len(paragraph) <= chunk_size:
                    current_chunk = paragraph
                else:
                    # Split large paragraph into sentence-based chunks
                    paragraph_chunks = self._split_paragraph_by_sentences(paragraph, chunk_size)
                    chunks.extend(paragraph_chunks[:-1])  # Add all but last chunk
                    current_chunk = paragraph_chunks[-1] if paragraph_chunks else ""
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        self.logger.info(f"Split {len(paragraphs)} paragraphs into {len(chunks)} chunks")
        return chunks
    
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
    
    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }
