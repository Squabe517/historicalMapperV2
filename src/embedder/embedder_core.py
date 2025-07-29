"""
Core embedder functionality for inserting maps into EPUB files.

Coordinates the embedding process by integrating outputs from AI
and mapping modules to insert map images at appropriate locations
in EPUB documents.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from lxml import etree
import ebooklib
from ebooklib import epub

from ..config.logger_module import log_info, log_warning, log_error
from .embedder_config import EmbedderConfig
from .embedder_strategy import ImageEmbedStrategy, ExternalImageStrategy, InlineImageStrategy
from .embedder_errors import (
    EmbedderError, 
    ParagraphNotFoundError, 
    InvalidEpubStructureError,
    ImageEmbedError
)


class EpubMapEmbedder:
    """
    Embeds maps into EPUB based on AI and mapping results.
    
    Direct integration approach - no intermediate data structures needed.
    Works directly with outputs from AI and mapping modules.
    """
    
    def __init__(self, 
                 config: EmbedderConfig = None,
                 strategy: ImageEmbedStrategy = None):
        """
        Initialize the embedder.
        
        Args:
            config: Embedder configuration (uses defaults if None)
            strategy: Image embedding strategy (auto-selected based on config if None)
        """
        self.config = config or EmbedderConfig()
        
        # Select strategy based on config if not provided
        if strategy is None:
            if self.config.embed_strategy == "inline":
                self.strategy = InlineImageStrategy()
            else:
                self.strategy = ExternalImageStrategy()
        else:
            self.strategy = strategy
            
        self._paragraph_cache = {}  # Cache for paragraph lookups
        self._chunk_to_para_map = {}  # Maps chunk indices to paragraph ranges
        self._epub_structure = {}  # Store EPUB structure info
        
        log_info(f"EpubMapEmbedder initialized with {self.config.embed_strategy} strategy")
    
    def embed_maps(self,
                   book: epub.EpubBook,
                   ai_results: List[List[Dict[str, Any]]],  # From AI module
                   map_images: Dict[str, bytes],            # From mapping module
                   chunk_info: Optional[List[Tuple[int, int]]] = None) -> epub.EpubBook:
        """
        Main entry point for embedding maps.
        
        Args:
            book: EPUB book to modify
            ai_results: AI module output - List of lists, each containing
                       dicts with 'place' and 'zoom'
            map_images: Mapping module output - cache_key -> image bytes
            chunk_info: Optional list of (start_para, end_para) tuples for each chunk
            
        Returns:
            Modified EPUB book with embedded maps
            
        Raises:
            InvalidEpubStructureError: If EPUB structure is invalid
        """
        # Validate EPUB structure
        self.validate_epub_structure(book)
        
        # Analyze EPUB structure for path calculation
        self._analyze_epub_structure(book)
        
        # Build paragraph index
        self._build_paragraph_index(book)
        
        # Build chunk-to-paragraph mapping if provided
        if chunk_info:
            self._build_chunk_mapping(chunk_info)
        
        # Process each place
        embedded_count = 0
        skipped_count = 0
        
        for chunk_idx, chunk_places in enumerate(ai_results):
            for place_info in chunk_places:
                try:
                    # Find matching image
                    cache_key = self._find_cache_key(place_info, map_images)
                    if not cache_key:
                        log_warning(f"No map image found for {place_info['place']}")
                        skipped_count += 1
                        continue
                    
                    # Calculate paragraph index
                    paragraph_idx = self._chunk_to_paragraph_index(chunk_idx, place_info)
                    
                    # Embed the map
                    self._embed_single_map(
                        book, 
                        paragraph_idx,
                        place_info['place'],
                        map_images[cache_key],
                        cache_key
                    )
                    embedded_count += 1
                    
                except (ParagraphNotFoundError, ImageEmbedError) as e:
                    log_error(f"Failed to embed map for {place_info['place']}: {e}")
                    skipped_count += 1
                    continue
                except Exception as e:
                    log_error(f"Unexpected error embedding map for {place_info['place']}: {e}")
                    skipped_count += 1
                    continue
        
        log_info(f"Embedding complete: {embedded_count} maps embedded, {skipped_count} skipped")
        return book
    
    def _analyze_epub_structure(self, book: epub.EpubBook) -> None:
        """
        Analyze EPUB structure to determine correct image paths.
        
        This examines where XHTML files are located to determine:
        - Common directory patterns
        - How to construct relative paths to images
        """
        # Get all document items
        doc_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        
        # Analyze paths
        paths = []
        for item in doc_items:
            if hasattr(item, 'file_name') and item.file_name:
                paths.append(item.file_name)
        
        if not paths:
            log_warning("No document paths found for structure analysis")
            return
        
        # Determine common patterns
        self._epub_structure['total_docs'] = len(paths)
        
        # Check if documents are in subdirectories
        docs_in_subdirs = sum(1 for p in paths if '/' in p)
        self._epub_structure['docs_in_subdirs'] = docs_in_subdirs
        
        # Find common directory patterns
        directories = set()
        for path in paths:
            if '/' in path:
                dir_path = path.rsplit('/', 1)[0]
                directories.add(dir_path)
        
        self._epub_structure['directories'] = list(directories)
        
        # Determine if most docs are in subdirectories
        self._epub_structure['majority_in_subdirs'] = docs_in_subdirs > len(paths) / 2
        
        log_info(f"EPUB structure: {len(paths)} documents, "
                f"{docs_in_subdirs} in subdirectories, "
                f"directories: {directories}")
    
    def _calculate_image_path(self, xhtml_path: str, image_href: str) -> str:
        """
        Calculate the correct relative path from XHTML to image.
        
        Args:
            xhtml_path: Path to the XHTML file (e.g., "text/chapter1.xhtml")
            image_href: Path to the image (e.g., "images/map.png")
            
        Returns:
            Correct relative path (e.g., "../images/map.png" or "images/map.png")
        """
        # Handle different path separators
        xhtml_path = xhtml_path.replace('\\', '/')
        image_href = image_href.replace('\\', '/')
        
        # Count directory levels in XHTML path
        xhtml_parts = xhtml_path.split('/')
        xhtml_depth = len(xhtml_parts) - 1  # -1 for the filename
        
        # Count directory levels in image path
        image_parts = image_href.split('/')
        image_depth = len(image_parts) - 1  # -1 for the filename
        
        # If XHTML is in a subdirectory, we need to go up
        if xhtml_depth > 0:
            # Go up the required number of levels
            up_levels = '../' * xhtml_depth
            return up_levels + image_href
        else:
            # XHTML is in root, use direct path
            return image_href
    
    def _build_paragraph_index(self, book: epub.EpubBook) -> None:
        """Build index of paragraphs across all XHTML documents."""
        self._paragraph_cache = {}
        para_count = 0
        
        # Get all XHTML documents in spine order
        spine_items = []
        for spine_item in book.spine:
            if isinstance(spine_item, tuple):
                item_id = spine_item[0]
                item = book.get_item_with_id(item_id)
                if item:
                    spine_items.append(item)
            else:
                item = book.get_item_with_id(spine_item)
                if item:
                    spine_items.append(item)
        
        # Process documents in reading order
        for item in spine_items:
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
                
            try:
                # Parse XHTML content
                parser = etree.XMLParser(recover=True, encoding='utf-8')
                tree = etree.fromstring(item.content, parser=parser)
                
                # Handle different namespace scenarios
                # Try with namespace first
                nsmap = {'html': 'http://www.w3.org/1999/xhtml'}
                paragraphs = tree.xpath('//html:p', namespaces=nsmap)
                
                # If no results, try without namespace
                if not paragraphs:
                    paragraphs = tree.xpath('//p')
                
                for para in paragraphs:
                    # Store paragraph info
                    self._paragraph_cache[para_count] = {
                        'item': item,
                        'element': para,
                        'tree': tree,
                        'text': self._get_element_text(para)
                    }
                    para_count += 1
                    
                log_info(f"Indexed {len(paragraphs)} paragraphs from {item.file_name}")
                    
            except etree.XMLSyntaxError as e:
                log_warning(f"Skipping malformed document {item.file_name}: {e}")
            except Exception as e:
                log_error(f"Error processing document {item.file_name}: {e}")
        
        log_info(f"Total paragraphs indexed: {para_count}")
    
    def _get_element_text(self, element) -> str:
        """Extract all text from an element and its children."""
        if element is None:
            return ""
        
        text_parts = []
        if element.text:
            text_parts.append(element.text)
        
        for child in element:
            text_parts.append(self._get_element_text(child))
            if child.tail:
                text_parts.append(child.tail)
        
        return ' '.join(text_parts).strip()
    
    def _build_chunk_mapping(self, chunk_info: List[Tuple[int, int]]) -> None:
        """Build mapping from chunk indices to paragraph ranges."""
        self._chunk_to_para_map = {}
        for chunk_idx, (start_para, end_para) in enumerate(chunk_info):
            self._chunk_to_para_map[chunk_idx] = (start_para, end_para)
        log_info(f"Built chunk mapping for {len(chunk_info)} chunks")
    
    def _find_cache_key(self, place_info: Dict[str, Any], 
                       map_images: Dict[str, bytes]) -> Optional[str]:
        """Find cache key for a place in map_images."""
        place_name = place_info['place']
        
        # Normalize place name for comparison
        # Same logic as used in mapping module's cache key generation
        safe_place = re.sub(r'[^a-zA-Z0-9\-.]', '_', place_name)[:20]
        
        # Look for exact match first
        for cache_key in map_images:
            if cache_key.startswith(safe_place + "_"):
                return cache_key
        
        # Try with partial matching (in case of slight differences)
        place_words = safe_place.lower().split('_')
        place_words = [w for w in place_words if len(w) > 2]  # Skip short words
        
        if place_words:
            for cache_key in map_images:
                cache_key_lower = cache_key.lower()
                if all(word in cache_key_lower for word in place_words):
                    log_info(f"Fuzzy matched '{place_name}' to cache key '{cache_key}'")
                    return cache_key
        
        return None
    
    def _chunk_to_paragraph_index(self, chunk_idx: int, 
                                 place_info: Dict[str, Any]) -> int:
        """
        Convert chunk index to paragraph index.
        
        Uses chunk mapping if available, otherwise estimates based on
        typical paragraph distribution.
        """
        # If we have explicit chunk mapping, use it
        if self._chunk_to_para_map and chunk_idx in self._chunk_to_para_map:
            start_para, end_para = self._chunk_to_para_map[chunk_idx]
            # Return the end of the chunk range (place at end of chunk)
            return min(end_para, len(self._paragraph_cache) - 1)
        
        # Otherwise, estimate based on typical distribution
        # Assuming chunks are roughly equal and contain ~5-10 paragraphs each
        paragraphs_per_chunk = max(1, len(self._paragraph_cache) // max(1, len(self._chunk_to_para_map) or 10))
        estimated_idx = min(
            chunk_idx * paragraphs_per_chunk + paragraphs_per_chunk - 1,
            len(self._paragraph_cache) - 1
        )
        
        log_info(f"Estimated paragraph index {estimated_idx} for chunk {chunk_idx}")
        return estimated_idx
    
    def _embed_single_map(self, 
                         book: epub.EpubBook,
                         paragraph_idx: int,
                         place: str,
                         image_bytes: bytes,
                         cache_key: str) -> None:
        """Embed a single map after the specified paragraph."""
        
        # Get paragraph info
        if paragraph_idx not in self._paragraph_cache:
            raise ParagraphNotFoundError(f"Paragraph {paragraph_idx} not found")
        
        para_info = self._paragraph_cache[paragraph_idx]
        
        # Add image to EPUB using strategy
        image_href = self.strategy.embed_image(book, cache_key, image_bytes)
        
        # Create figure element using strategy
        figure = self.strategy.create_figure_element(
            image_href, place, self.config, para_info['item'].file_name
        )
        
        # Insert after paragraph
        para_element = para_info['element']
        parent = para_element.getparent()
        
        if parent is None:
            raise ImageEmbedError(f"Paragraph {paragraph_idx} has no parent element")
        
        # Find paragraph position in parent
        para_index = list(parent).index(para_element)
        
        # Insert figure after paragraph
        parent.insert(para_index + 1, figure)
        
        # Update item content with modified tree
        # Serialize with proper encoding and XML declaration
        updated_content = etree.tostring(
            para_info['tree'], 
            pretty_print=True,
            encoding='utf-8',
            xml_declaration=True
        )
        
        para_info['item'].content = updated_content
        
        log_info(f"Embedded map for '{place}' after paragraph {paragraph_idx} in {para_info['item'].file_name}")
    
    def validate_epub_structure(self, book: epub.EpubBook) -> None:
        """Validate EPUB has expected structure for embedding."""
        # Check for any XHTML documents
        doc_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        if not doc_items:
            raise InvalidEpubStructureError("No XHTML documents found in EPUB")
        
        # Check spine exists
        if not hasattr(book, 'spine') or not book.spine:
            raise InvalidEpubStructureError("EPUB has no spine (reading order)")
        
        # Check for at least one paragraph
        para_found = False
        for item in doc_items:
            try:
                parser = etree.XMLParser(recover=True, encoding='utf-8')
                tree = etree.fromstring(item.content, parser=parser)
                
                # Try with namespace
                if tree.xpath('//html:p', namespaces={'html': 'http://www.w3.org/1999/xhtml'}):
                    para_found = True
                    break
                # Try without namespace
                elif tree.xpath('//p'):
                    para_found = True
                    break
            except:
                continue
        
        if not para_found:
            raise InvalidEpubStructureError("No paragraphs found in EPUB")
        
        log_info("EPUB structure validation passed")