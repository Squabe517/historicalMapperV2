"""
Example usage of the embedder module.

Demonstrates how to integrate the embedder with other modules
to add maps to an EPUB file.
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
    Complete pipeline to add maps to an EPUB file.
    
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
        
        # 2. Chunk text for AI processing
        log_info("Chunking text for AI analysis...")
        chunker = TextChunker()
        chunks = chunker.chunk_text(paragraphs, chunk_size=1000)
        log_info(f"Created {len(chunks)} chunks")
        
        # 3. Extract places using AI
        log_info("Extracting place names using AI...")
        ai_client = OpenAIClient()
        ai_results = ai_client.batch_analyze_chunks(chunks)
        
        # Count total places found
        total_places = sum(len(chunk_places) for chunk_places in ai_results)
        log_info(f"Found {total_places} place references")
        
        # 4. Get maps for all places
        log_info("Fetching maps from Google Maps...")
        places_flat = [place for chunk in ai_results for place in chunk]
        
        if not places_flat:
            log_info("No places found to map")
            return
        
        orchestrator = MappingOrchestrator()
        map_images = orchestrator.batch_get_maps(places_flat)
        log_info(f"Retrieved {len(map_images)} map images")
        
        # 5. Configure embedder
        config = EmbedderConfig(
            figure_class="historical-map",
            caption_template="Map of {place}",
            embed_strategy="external"  # Use external files
        )
        
        # 6. Embed maps into EPUB
        log_info("Embedding maps into EPUB...")
        embedder = EpubMapEmbedder(config=config)
        embedder.embed_maps(book, ai_results, map_images)
        
        # 7. Save enhanced EPUB
        log_info(f"Saving enhanced EPUB to: {output_path}")
        epub.write_epub(output_path, book)
        
        log_info("✅ Successfully created EPUB with embedded maps!")
        
        # Print statistics
        cache_stats = orchestrator.get_stats()
        print(f"\n📊 Statistics:")
        print(f"  - Paragraphs processed: {len(paragraphs)}")
        print(f"  - Places identified: {total_places}")
        print(f"  - Maps embedded: {len(map_images)}")
        print(f"  - Cache usage: {cache_stats['cache']['usage_percent']:.1f}%")
        
    except Exception as e:
        log_error(f"Failed to process EPUB: {e}")
        raise


def demonstrate_custom_configuration():
    """Demonstrate different embedder configurations."""
    print("\n" + "=" * 60)
    print("Embedder Configuration Examples")
    print("=" * 60)
    
    # Example 1: Custom styling
    config1 = EmbedderConfig(
        figure_class="ancient-map",
        figure_style="margin: 2em auto; text-align: center; max-width: 80%;",
        caption_template="Historical map of {place}",
        max_image_width="600px"
    )
    print("\n1. Custom Styling Configuration:")
    print(f"   - Class: {config1.figure_class}")
    print(f"   - Style: {config1.figure_style}")
    print(f"   - Caption: {config1.caption_template}")
    
    # Example 2: Inline images (Base64)
    config2 = EmbedderConfig(
        embed_strategy="inline",
        caption_template="📍 {place}"
    )
    print("\n2. Inline Image Configuration:")
    print(f"   - Strategy: {config2.embed_strategy}")
    print(f"   - Images embedded as Base64 data URIs")
    
    # Example 3: Minimal configuration
    config3 = EmbedderConfig()
    print("\n3. Default Configuration:")
    print(f"   - All settings use sensible defaults")


def demonstrate_selective_embedding():
    """Demonstrate how to selectively embed maps."""
    print("\n" + "=" * 60)
    print("Selective Embedding Example")
    print("=" * 60)
    
    # Mock data for demonstration
    ai_results = [
        [{"place": "Rome", "zoom": 12}, {"place": "Venice", "zoom": 13}],
        [{"place": "Constantinople", "zoom": 11}],
        [{"place": "Athens", "zoom": 12}]
    ]
    
    # Filter to only embed certain places
    important_places = {"Rome", "Constantinople"}
    
    filtered_results = []
    for chunk_places in ai_results:
        filtered_chunk = [p for p in chunk_places if p["place"] in important_places]
        filtered_results.append(filtered_chunk)
    
    print(f"\nOriginal places: {[p['place'] for chunk in ai_results for p in chunk]}")
    print(f"Filtered places: {[p['place'] for chunk in filtered_results for p in chunk]}")
    
    # Would then pass filtered_results to embedder.embed_maps()


def main():
    """Run examples."""
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
    
    print("🗺️  Historical ePub Map Enhancer - Embedder Examples")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_custom_configuration()
    demonstrate_selective_embedding()
    
    # Process a real EPUB if provided
    if len(sys.argv) > 2:
        input_epub = sys.argv[1]
        output_epub = sys.argv[2]
        
        if os.path.exists(input_epub):
            print(f"\n📚 Processing EPUB: {input_epub}")
            process_epub_with_maps(input_epub, output_epub)
        else:
            print(f"\n❌ Input file not found: {input_epub}")
    else:
        print("\n💡 To process an EPUB, run:")
        print("   python embedder_example.py input.epub output.epub")


if __name__ == "__main__":
    main()
