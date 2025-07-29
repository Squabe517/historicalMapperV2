#!/usr/bin/env python3
"""
Historical ePub Map Enhancer - Main Integration Script

This script processes EPUB files to automatically embed historical maps
using DOM-based parsing for 100% accurate placement.

Usage:
    python main.py input.epub output.epub [options]

Options:
    --chunk-size SIZE      Maximum chunk size for AI processing (default: 1000)
    --embed-strategy TYPE  'external' or 'inline' (default: external)
    --cache-dir PATH       Directory for map cache (default: .cache_maps)
    --log-level LEVEL      Logging level (default: INFO)
    --dry-run             Process without saving output
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import all required modules
from src.config.config_module import load_config, validate_config, get_config
from src.config.logger_module import initialize_logger, log_info, log_error, log_warning

from src.parser.epub_parser import EpubParser
from src.parser.text_chunker import TextChunker, ChunkedParagraph
from src.parser.document_parser import ParagraphElement

from src.ai.openai_client import OpenAIClient
from src.ai.openai_client import OpenAIError

from src.mapping.mapping_client import GoogleMapsClient
from src.mapping.mapping_cache import ImageCacheManager
from src.mapping.mapping_errors import GeocodingError, MapFetchError, CacheError

from src.embedder.embedder_config import EmbedderConfig
from src.embedder.embedder_strategy import ExternalImageStrategy, InlineImageStrategy
from src.embedder.embedder_errors import ImageEmbedError


class HistoricalMapEnhancer:
    """
    Main application class for processing EPUB files with DOM-based map embedding.
    
    This class orchestrates the entire pipeline:
    1. Parse EPUB with DOM structure preservation
    2. Chunk paragraphs for AI processing
    3. Extract geographical places
    4. Fetch map images
    5. Embed maps directly after their source paragraphs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Historical Map Enhancer.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        
        # Initialize components
        self.parser = EpubParser()
        self.chunker = TextChunker()
        self.ai_client = OpenAIClient()
        self.maps_client = GoogleMapsClient()
        self.cache_manager = ImageCacheManager(
            cache_dir=self.config.get('cache_dir'),
            ttl_seconds=self.config.get('cache_ttl', 86400),
            max_cache_size_mb=self.config.get('cache_size_mb', 100)
        )
        
        # Configure embedder
        embedder_config = EmbedderConfig(
            embed_strategy=self.config.get('embed_strategy', 'external'),
            figure_class=self.config.get('figure_class', 'historical-map'),
            caption_template=self.config.get('caption_template', 'Map of {place}'),
            max_image_width=self.config.get('max_image_width', '100%')
        )
        
        # Select embedding strategy
        if embedder_config.embed_strategy == 'inline':
            self.embed_strategy = InlineImageStrategy()
        else:
            self.embed_strategy = ExternalImageStrategy()
        
        self.embedder_config = embedder_config
        
        # Statistics tracking
        self.stats = {
            'start_time': None,
            'paragraphs': 0,
            'chunks': 0,
            'places': 0,
            'maps_fetched': 0,
            'maps_cached': 0,
            'maps_embedded': 0,
            'errors': 0
        }
    
    def process_epub(self, input_path: str, output_path: str, dry_run: bool = False) -> bool:
        """
        Process an EPUB file to add historical maps.
        
        Args:
            input_path: Path to input EPUB file
            output_path: Path to save enhanced EPUB
            dry_run: If True, process without saving output
            
        Returns:
            True if successful, False otherwise
        """
        self.stats['start_time'] = time.time()
        
        try:
            # Step 1: Load and parse EPUB with DOM
            log_info(f"Loading EPUB: {input_path}")
            self.parser.load_file(input_path)
            
            metadata = self.parser.get_metadata()
            log_info(f"Processing: {metadata['title']} by {', '.join(metadata['authors'])}")
            
            # Extract paragraphs with DOM structure
            paragraph_elements = self.parser.extract_paragraphs_with_dom()
            self.stats['paragraphs'] = len(paragraph_elements)
            log_info(f"Found {len(paragraph_elements)} paragraphs")
            
            if not paragraph_elements:
                log_warning("No paragraphs found in EPUB")
                return True
            
            # Step 2: Chunk paragraphs while maintaining DOM references
            chunk_size = self.config.get('chunk_size', 1000)
            chunked_paragraphs = self._chunk_paragraphs(paragraph_elements, chunk_size)
            
            # Step 3: Extract places from chunks
            places_by_paragraph = self._extract_places(chunked_paragraphs)
            
            # Collect all places
            all_places = []
            for places in places_by_paragraph.values():
                all_places.extend(places)
            
            self.stats['places'] = len(all_places)
            
            if not all_places:
                log_info("No geographical places found in text")
                if not dry_run:
                    self.parser.save_document(output_path)
                return True
            
            # Step 4: Fetch maps for all places
            map_images = self._fetch_maps(all_places)
            
            if not map_images:
                log_warning("No maps could be fetched")
                if not dry_run:
                    self.parser.save_document(output_path)
                return True
            
            # Step 5: Embed maps in DOM
            self._embed_maps(places_by_paragraph, map_images)
            
            # Step 6: Save enhanced EPUB
            if not dry_run:
                log_info(f"Saving enhanced EPUB to: {output_path}")
                self.parser.save_document(output_path)
                log_info("✅ Successfully created EPUB with embedded maps!")
            else:
                log_info("Dry run complete - no output saved")
            
            # Print statistics
            self._print_statistics()
            
            return True
            
        except Exception as e:
            log_error(f"Failed to process EPUB: {e}")
            self.stats['errors'] += 1
            return False
    
    def _chunk_paragraphs(self, paragraph_elements: List[ParagraphElement], 
                         chunk_size: int) -> List[ChunkedParagraph]:
        """Chunk paragraphs while maintaining DOM references."""
        log_info(f"Chunking paragraphs (max size: {chunk_size} chars)...")
        
        chunked_paragraphs = self.chunker.chunk_paragraph_elements(
            paragraph_elements, 
            chunk_size=chunk_size
        )
        
        self.stats['chunks'] = len(chunked_paragraphs)
        
        # Get chunking statistics
        chunk_stats = self.chunker.get_chunk_stats(chunked_paragraphs)
        log_info(f"Created {chunk_stats['total_chunks']} chunks, "
                f"{chunk_stats['split_paragraphs']} paragraphs were split")
        
        return chunked_paragraphs
    
    def _extract_places(self, chunked_paragraphs: List[ChunkedParagraph]) -> Dict[ParagraphElement, List[Dict[str, Any]]]:
        """Extract geographical places from chunks."""
        log_info("Analyzing text for geographical places...")
        
        # Process chunks through AI
        chunk_results = []
        total_chunks = len(chunked_paragraphs)
        
        for idx, chunked in enumerate(chunked_paragraphs):
            try:
                # Show progress for large documents
                if total_chunks > 50 and idx % 10 == 0:
                    log_info(f"Processing chunk {idx + 1}/{total_chunks}...")
                
                places = self.ai_client.analyze_chunk(chunked.text)
                chunk_results.append(places)
                
                if places:
                    para_idx = chunked.paragraph_element.metadata['index']
                    log_info(f"Found {len(places)} places in paragraph {para_idx}")
                    
            except OpenAIError as e:
                log_error(f"AI analysis failed for chunk {idx}: {e}")
                chunk_results.append([])
                self.stats['errors'] += 1
            except Exception as e:
                log_error(f"Unexpected error analyzing chunk {idx}: {e}")
                chunk_results.append([])
                self.stats['errors'] += 1
        
        # Aggregate results by paragraph
        places_by_paragraph = self.chunker.aggregate_results_by_paragraph(
            chunked_paragraphs, chunk_results
        )
        
        # Log summary
        paragraphs_with_places = sum(1 for places in places_by_paragraph.values() if places)
        log_info(f"Found places in {paragraphs_with_places} paragraphs")
        
        return places_by_paragraph
    
    def _fetch_maps(self, places: List[Dict[str, Any]]) -> Dict[str, bytes]:
        """Fetch map images for all places."""
        log_info(f"Fetching maps for {len(places)} places...")
        
        map_images = {}
        cache_hits = 0
        
        for idx, place_info in enumerate(places):
            place_name = place_info['place']
            zoom = place_info.get('zoom', 12)
            
            try:
                # Check cache first
                cached_bytes = self.cache_manager.get_cached_bytes(
                    place_name, zoom, "600x400"
                )
                
                if cached_bytes:
                    cache_key = self.cache_manager._generate_cache_key(
                        place_name, zoom, "600x400"
                    )
                    map_images[cache_key] = cached_bytes
                    cache_hits += 1
                    continue
                
                # Geocode place
                coords = self.maps_client.geocode_place(place_name)
                
                # Fetch map
                map_bytes = self.maps_client.fetch_map_bytes(
                    coords['lat'], coords['lng'], zoom
                )
                
                # Cache the image
                cache_key = self.cache_manager.cache_bytes(
                    place_name, zoom, "600x400", map_bytes
                )
                
                map_images[cache_key] = map_bytes
                self.stats['maps_fetched'] += 1
                
                # Log progress for large batches
                if len(places) > 20 and (idx + 1) % 10 == 0:
                    log_info(f"Fetched {idx + 1}/{len(places)} maps...")
                    
            except (GeocodingError, MapFetchError) as e:
                log_warning(f"Failed to fetch map for '{place_name}': {e}")
                self.stats['errors'] += 1
            except Exception as e:
                log_error(f"Unexpected error fetching map for '{place_name}': {e}")
                self.stats['errors'] += 1
        
        self.stats['maps_cached'] = cache_hits
        log_info(f"Retrieved {len(map_images)} maps ({cache_hits} from cache)")
        
        return map_images
    
    def _embed_maps(self, places_by_paragraph: Dict[ParagraphElement, List[Dict[str, Any]]], 
                   map_images: Dict[str, bytes]):
        """Embed maps directly after their source paragraphs."""
        log_info("Embedding maps into EPUB...")
        
        embedded_count = 0
        
        for para_elem, places in places_by_paragraph.items():
            if not places:
                continue
            
            # Process each place for this paragraph
            for place_info in places:
                # Find the map image
                cache_key = self._find_cache_key(place_info, map_images)
                if not cache_key:
                    continue
                
                try:
                    # Add image to EPUB
                    image_href = self.embed_strategy.embed_image(
                        self.parser.book,
                        cache_key,
                        map_images[cache_key]
                    )
                    
                    # Create figure element
                    figure = self.embed_strategy.create_figure_element(
                        image_href,
                        place_info['place'],
                        self.embedder_config,
                        para_elem.metadata['file_path']
                    )
                    
                    # Insert after paragraph
                    self.parser.update_dom_element(para_elem, figure)
                    embedded_count += 1
                    
                    para_idx = para_elem.metadata['index']
                    log_info(f"Embedded map for '{place_info['place']}' "
                            f"after paragraph {para_idx}")
                    
                except ImageEmbedError as e:
                    log_error(f"Failed to embed map for '{place_info['place']}': {e}")
                    self.stats['errors'] += 1
                except Exception as e:
                    log_error(f"Unexpected error embedding map for "
                             f"'{place_info['place']}': {e}")
                    self.stats['errors'] += 1
        
        self.stats['maps_embedded'] = embedded_count
        log_info(f"Embedded {embedded_count} maps")
    
    def _find_cache_key(self, place_info: Dict[str, Any], 
                       map_images: Dict[str, bytes]) -> Optional[str]:
        """Find the cache key for a place in map images."""
        import re
        
        place_name = place_info['place']
        safe_place = re.sub(r'[^a-zA-Z0-9\-.]', '_', place_name)[:20]
        
        # Try exact match
        for cache_key in map_images:
            if cache_key.startswith(safe_place + "_"):
                return cache_key
        
        # Try fuzzy match
        place_words = safe_place.lower().split('_')
        place_words = [w for w in place_words if len(w) > 2]
        
        if place_words:
            for cache_key in map_images:
                cache_key_lower = cache_key.lower()
                if all(word in cache_key_lower for word in place_words):
                    return cache_key
        
        return None
    
    def _print_statistics(self):
        """Print processing statistics."""
        elapsed = time.time() - self.stats['start_time']
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Get rate limit status
        rate_status = self.maps_client.get_rate_limit_status()
        
        print("\n" + "=" * 60)
        print("📊 Processing Statistics")
        print("=" * 60)
        
        print(f"\n📖 Document Analysis:")
        print(f"  - Paragraphs processed: {self.stats['paragraphs']:,}")
        print(f"  - Text chunks created: {self.stats['chunks']:,}")
        print(f"  - Unique places found: {self.stats['places']:,}")
        
        print(f"\n🗺️  Map Processing:")
        print(f"  - Maps fetched from API: {self.stats['maps_fetched']:,}")
        print(f"  - Maps loaded from cache: {self.stats['maps_cached']:,}")
        print(f"  - Maps embedded in EPUB: {self.stats['maps_embedded']:,}")
        
        print(f"\n💾 Cache Status:")
        print(f"  - Cache size: {cache_stats['total_size_mb']:.1f} MB "
              f"({cache_stats['usage_percent']:.1f}% of {cache_stats['max_size_mb']} MB)")
        print(f"  - Cached files: {cache_stats['total_files']:,}")
        
        print(f"\n⚡ Performance:")
        print(f"  - Processing time: {elapsed:.1f} seconds")
        print(f"  - Rate limit tokens: {rate_status['available_tokens']:.1f}")
        
        if self.stats['errors'] > 0:
            print(f"\n⚠️  Errors encountered: {self.stats['errors']}")
        
        print("\n" + "=" * 60)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Historical ePub Map Enhancer - Automatically embed maps in EPUB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.epub enhanced_book.epub
  %(prog)s book.epub enhanced_book.epub --embed-strategy inline
  %(prog)s book.epub enhanced_book.epub --chunk-size 1500 --cache-dir ./maps
  %(prog)s book.epub enhanced_book.epub --dry-run --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument('input', help='Input EPUB file path')
    parser.add_argument('output', help='Output EPUB file path')
    
    # Optional arguments
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Maximum chunk size for AI processing (default: 1000)')
    
    parser.add_argument('--embed-strategy', choices=['external', 'inline'],
                       default='external',
                       help='Image embedding strategy (default: external)')
    
    parser.add_argument('--cache-dir', type=str,
                       help='Directory for map cache (default: .cache_maps)')
    
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Process without saving output')
    
    parser.add_argument('--figure-class', type=str, default='historical-map',
                       help='CSS class for map figures (default: historical-map)')
    
    parser.add_argument('--caption-template', type=str, default='Map of {place}',
                       help='Caption template for maps (default: "Map of {place}")')
    
    parser.add_argument('--max-image-width', type=str, default='100%%',
                       help='Maximum image width CSS (default: 100%%)')
    
    return parser.parse_args()


def main():
    """Main entry point for the Historical Map Enhancer."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize logging
    initialize_logger(log_level=args.log_level)
    
    # Load configuration from environment
    load_config()
    
    # Validate required API keys
    try:
        validate_config(['OPENAI_API_KEY', 'GOOGLE_MAPS_API_KEY'])
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure the following environment variables are set:")
        print("  - OPENAI_API_KEY")
        print("  - GOOGLE_MAPS_API_KEY")
        print("\nYou can set them in a .env file or as environment variables.")
        return 1
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"\n❌ Input file not found: {args.input}")
        return 1
    
    # Print header
    print("\n" + "=" * 60)
    print("🗺️  Historical ePub Map Enhancer")
    print("=" * 60)
    print(f"\n📖 Input:  {args.input}")
    print(f"📖 Output: {args.output}")
    print(f"⚙️  Strategy: {args.embed_strategy}")
    print(f"📏 Chunk size: {args.chunk_size}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No output will be saved")
    
    print("\nStarting processing...\n")
    
    # Build configuration
    config = {
        'chunk_size': args.chunk_size,
        'embed_strategy': args.embed_strategy,
        'figure_class': args.figure_class,
        'caption_template': args.caption_template,
        'max_image_width': args.max_image_width
    }
    
    if args.cache_dir:
        config['cache_dir'] = args.cache_dir
    
    # Create enhancer and process EPUB
    enhancer = HistoricalMapEnhancer(config)
    success = enhancer.process_epub(args.input, args.output, args.dry_run)
    
    if success:
        if not args.dry_run:
            print(f"\n✅ Success! Enhanced EPUB saved to: {args.output}")
        return 0
    else:
        print(f"\n❌ Processing failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())