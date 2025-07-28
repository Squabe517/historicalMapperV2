"""
Integration example for the parser module.

Demonstrates how to use EpubParser and TextChunker together
to process ePub files for the Historical ePub Map Enhancer.
"""

from pathlib import Path
from src.config.config_module import load_config, get_config
from src.config.logger_module import initialize_logger, log_info, log_error
from src.parser.epub_parser import EpubParser
from src.parser.text_chunker import TextChunker
from src.parser.document_parser import ParserError


def demonstrate_epub_processing(epub_path: str):
    """
    Demonstrate complete ePub processing pipeline.
    
    Args:
        epub_path: Path to ePub file to process
    """
    try:
        # Initialize parser and chunker
        parser = EpubParser()
        chunker = TextChunker()
        
        log_info(f"Starting ePub processing for: {Path(epub_path).name}")
        
        # Load and parse ePub
        parser.load_file(epub_path)
        
        # Extract metadata
        metadata = parser.get_metadata()
        log_info(f"Processing: {metadata['title']} by {', '.join(metadata['authors'])}")
        
        # Extract text paragraphs
        paragraphs = parser.extract_text()
        log_info(f"Extracted {len(paragraphs)} paragraphs")
        
        # Chunk text for AI processing
        chunk_size = int(get_config("CHUNK_SIZE", "1000"))
        chunks = chunker.chunk_text(paragraphs, chunk_size=chunk_size)
        
        # Get chunking statistics
        stats = chunker.get_chunk_stats(chunks)
        log_info(f"Created {stats['total_chunks']} chunks, avg size: {stats['avg_chunk_size']:.1f} chars")
        
        # Display sample results
        print(f"\nMetadata:")
        print(f"  Title: {metadata['title']}")
        print(f"  Authors: {', '.join(metadata['authors'])}")
        print(f"  Language: {metadata['language']}")
        print(f"  Publisher: {metadata['publisher']}")
        
        print(f"\nText Processing Results:")
        print(f"  Total paragraphs: {len(paragraphs)}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Average chunk size: {stats['avg_chunk_size']:.1f} characters")
        print(f"  Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters")
        
        if chunks:
            print(f"\nSample chunk (first 200 chars):")
            print(f"  '{chunks[0][:200]}{'...' if len(chunks[0]) > 200 else ''}'")
        
        return {
            "metadata": metadata,
            "paragraphs": paragraphs,
            "chunks": chunks,
            "stats": stats
        }
        
    except ParserError as e:
        log_error(f"Parser error: {e}")
        print(f"Error processing ePub: {e}")
        return None
    
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        return None


def main():
    """Main function to demonstrate parser integration."""
    # Initialize configuration and logging
    load_config()
    initialize_logger(log_level=get_config("LOG_LEVEL", "INFO"))
    
    # Example usage
    epub_file = get_config("SAMPLE_EPUB")
    
    log_info("Parser module integration demonstration")
    
    if Path(epub_file).exists():
        result = demonstrate_epub_processing(epub_file)
        if result:
            log_info("ePub processing completed successfully")
        else:
            log_error("ePub processing failed")
    else:
        print(f"Sample ePub file not found: {epub_file}")
        print("Please provide a valid ePub file path in SAMPLE_EPUB config or place a sample file.")
        
        # Demonstrate error handling
        log_info("Demonstrating error handling with non-existent file")
        demonstrate_epub_processing("non_existent_file.epub")


if __name__ == "__main__":
    main()
