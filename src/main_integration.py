"""
Example usage of the embedder module with simplified paragraph-by-paragraph processing.

Ensures maps are placed directly after the paragraph that references them.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ebooklib import epub

# Import all required modules
from src.parser.epub_parser import EpubParser
from src.parser.text_chunker import TextChunker
from src.ai.openai_client import OpenAIClient
from src.mapping.mapping_workflow import MappingOrchestrator
from src.embedder.embedder_core import EpubMapEmbedder
from src.embedder.embedder_config import EmbedderConfig
from src.config.config_module import load_config, validate_config
from src.config.logger_module import initialize_logger, log_info, log_error


def process_epub_with_maps(input_path: str, output_path: str):
    """
    Complete pipeline to add maps to an EPUB file using paragraph-by-paragraph processing.
    
    Args:
        input_path: Path to input EPUB file
        output_path: Path to save enhanced EPUB
    """
    try:
        # 1. Parse EPUB
        log_info(f"Loading EPUB: {input_path}")
        parser = EpubParser()
        parser.load_file(input_path)
        paragraphs = parser.extract_text()
        book = parser.book  # Keep reference to epub object
        
        metadata = parser.get_metadata()
        log_info(f"Processing: {metadata['title']} by {', '.join(metadata['authors'])}")
        log_info(f"Found {len(paragraphs)} paragraphs")
        
        # 2. Initialize components
        chunker = TextChunker()
        ai_client = OpenAIClient()
        orchestrator = MappingOrchestrator()
        
        # 3. Process each paragraph individually
        log_info("Processing paragraphs individually for accurate map placement...")
        
        # Structure AI results to match paragraph indices
        ai_results_by_paragraph = []
        all_places = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                ai_results_by_paragraph.append([])
                continue
            
            # Check if paragraph needs to be chunked
            if len(paragraph) <= 1000:
                # Process paragraph as-is
                places = ai_client.analyze_chunk(paragraph)
            else:
                # Split large paragraph and combine results
                chunks = chunker._split_paragraph_by_sentences(paragraph, 1000)
                places = []
                for chunk in chunks:
                    chunk_places = ai_client.analyze_chunk(chunk)
                    places.extend(chunk_places)
            
            ai_results_by_paragraph.append(places)
            all_places.extend(places)
            
            if places:
                log_info(f"Paragraph {para_idx}: Found {len(places)} places")
        
        # Count total places found
        total_places = len(all_places)
        log_info(f"Found {total_places} place references total")
        
        if not all_places:
            log_info("No places found to map")
            return
        
        # 4. Get maps for all places at once (for efficiency)
        log_info("Fetching maps from Google Maps...")
        map_images = orchestrator.batch_get_maps(all_places)
        log_info(f"Retrieved {len(map_images)} map images")
        
        # 5. Configure embedder
        config = EmbedderConfig(
            figure_class="historical-map",
            caption_template="Map of {place}",
            embed_strategy="external"  # Use external files
        )
        
        # 6. Create exact paragraph mapping for embedder
        # This ensures each map is placed after its source paragraph
        chunk_info = [(i, i) for i in range(len(paragraphs))]
        
        # 7. Embed maps into EPUB
        log_info("Embedding maps into EPUB...")
        embedder = EpubMapEmbedder(config=config)
        embedder.embed_maps(
            book=book,
            ai_results=ai_results_by_paragraph,
            map_images=map_images,
            chunk_info=chunk_info  # Direct paragraph mapping
        )
        
        # 8. Save enhanced EPUB
        log_info(f"Saving enhanced EPUB to: {output_path}")
        epub.write_epub(output_path, book)
        
        log_info("✅ Successfully created EPUB with embedded maps!")
        
        # Print statistics
        cache_stats = orchestrator.get_stats()
        paragraphs_with_places = sum(1 for places in ai_results_by_paragraph if places)
        
        print(f"\n📊 Statistics:")
        print(f"  - Paragraphs processed: {len(paragraphs)}")
        print(f"  - Paragraphs with places: {paragraphs_with_places}")
        print(f"  - Places identified: {total_places}")
        print(f"  - Maps embedded: {len(map_images)}")
        print(f"  - Cache usage: {cache_stats['cache']['usage_percent']:.1f}%")
        
    except Exception as e:
        log_error(f"Failed to process EPUB: {e}")
        raise


def process_epub_batch_optimized(input_path: str, output_path: str):
    """
    Alternative approach that batches AI calls while maintaining paragraph accuracy.
    More efficient for API usage but slightly more complex.
    """
    try:
        # 1. Parse EPUB
        log_info(f"Loading EPUB: {input_path}")
        parser = EpubParser()
        parser.load_file(input_path)
        paragraphs = parser.extract_text()
        book = parser.book
        
        metadata = parser.get_metadata()
        log_info(f"Processing: {metadata['title']} by {', '.join(metadata['authors'])}")
        log_info(f"Found {len(paragraphs)} paragraphs")
        
        # 2. Create chunks but track paragraph boundaries
        log_info("Creating chunks with paragraph tracking...")
        chunks = []
        chunk_to_paragraph = []  # Maps chunk index to paragraph index
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            if len(paragraph) <= 1000:
                chunks.append(paragraph)
                chunk_to_paragraph.append(para_idx)
            else:
                # Split large paragraph
                chunker = TextChunker()
                para_chunks = chunker._split_paragraph_by_sentences(paragraph, 1000)
                for chunk in para_chunks:
                    chunks.append(chunk)
                    chunk_to_paragraph.append(para_idx)
        
        log_info(f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
        
        # 3. Batch process chunks through AI
        log_info("Extracting place names using AI...")
        ai_client = OpenAIClient()
        ai_chunk_results = ai_client.batch_analyze_chunks(chunks)
        
        # 4. Reorganize results by paragraph
        ai_results_by_paragraph = [[] for _ in range(len(paragraphs))]
        all_places = []
        
        for chunk_idx, places in enumerate(ai_chunk_results):
            para_idx = chunk_to_paragraph[chunk_idx]
            ai_results_by_paragraph[para_idx].extend(places)
            all_places.extend(places)
        
        total_places = len(all_places)
        log_info(f"Found {total_places} place references")
        
        if not all_places:
            log_info("No places found to map")
            return
        
        # 5. Get maps
        log_info("Fetching maps from Google Maps...")
        orchestrator = MappingOrchestrator()
        map_images = orchestrator.batch_get_maps(all_places)
        log_info(f"Retrieved {len(map_images)} map images")
        
        # 6. Configure and embed
        config = EmbedderConfig(
            figure_class="historical-map",
            caption_template="Map of {place}",
            embed_strategy="external"
        )
        
        # Create direct paragraph mapping
        chunk_info = [(i, i) for i in range(len(paragraphs))]
        
        log_info("Embedding maps into EPUB...")
        embedder = EpubMapEmbedder(config=config)
        embedder.embed_maps(
            book=book,
            ai_results=ai_results_by_paragraph,
            map_images=map_images,
            chunk_info=chunk_info
        )
        
        # 7. Save
        log_info(f"Saving enhanced EPUB to: {output_path}")
        epub.write_epub(output_path, book)
        
        log_info("✅ Successfully created EPUB with embedded maps!")
        
        # Statistics
        cache_stats = orchestrator.get_stats()
        paragraphs_with_places = sum(1 for places in ai_results_by_paragraph if places)
        
        print(f"\n📊 Statistics:")
        print(f"  - Paragraphs processed: {len(paragraphs)}")
        print(f"  - Paragraphs with places: {paragraphs_with_places}")
        print(f"  - Places identified: {total_places}")
        print(f"  - Maps embedded: {len(map_images)}")
        print(f"  - Cache usage: {cache_stats['cache']['usage_percent']:.1f}%")
        
    except Exception as e:
        log_error(f"Failed to process EPUB: {e}")
        raise


def main():
    """Run the EPUB processor."""
    # Initialize logging
    initialize_logger(log_level="INFO")
    
    # Load configuration
    load_config()
    
    # Validate required API keys
    try:
        validate_config(["OPENAI_API_KEY", "GOOGLE_MAPS_API_KEY"])
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("Please set required API keys in .env file")
        return
    
    print("🗺️  Historical ePub Map Enhancer")
    print("=" * 60)
    
    # Process a real EPUB if provided
    if len(sys.argv) > 2:
        input_epub = sys.argv[1]
        output_epub = sys.argv[2]
        
        if os.path.exists(input_epub):
            print(f"\n📚 Processing EPUB: {input_epub}")
            
            # You can choose which approach to use:
            # Option 1: Simple paragraph-by-paragraph (more API calls, simpler logic)
            process_epub_with_maps(input_epub, output_epub)
            
            # Option 2: Batch optimized (fewer API calls, tracks paragraph boundaries)
            # process_epub_batch_optimized(input_epub, output_epub)
        else:
            print(f"\n❌ Input file not found: {input_epub}")
    else:
        print("\n💡 To process an EPUB, run:")
        print("   python embedder_example.py input.epub output.epub")


if __name__ == "__main__":
    main()