"""
Image embedding strategies for the embedder module.

Provides different strategies for embedding images into EPUB files,
including external file references and inline Base64 encoding.
"""

from abc import ABC, abstractmethod
import base64
from lxml import etree
import ebooklib
from ebooklib import epub

from ..config.logger_module import log_info, log_error
from .embedder_config import EmbedderConfig
from .embedder_errors import ImageEmbedError


class ImageEmbedStrategy(ABC):
    """Abstract base class for image embedding strategies."""
    
    @abstractmethod
    def embed_image(self, book: epub.EpubBook, 
                   cache_key: str, 
                   image_bytes: bytes) -> str:
        """
        Add image to EPUB and return its href.
        
        Args:
            book: EPUB book object
            cache_key: Filename from mapping module (e.g., "Rome_abc123.png")
            image_bytes: Raw image data
            
        Returns:
            EPUB href for the image (e.g., "images/Rome_abc123.png")
            
        Raises:
            ImageEmbedError: If embedding fails
        """
        pass
    
    @abstractmethod
    def create_figure_element(self, image_href: str, 
                            place: str,
                            config: EmbedderConfig,
                            xhtml_path: str) -> etree.Element:
        """
        Create XHTML figure element with image and caption.
        
        Args:
            image_href: Href to the embedded image
            place: Place name for alt text and caption
            config: Embedder configuration
            xhtml_path: Path to the XHTML file (for relative path calculation)
            
        Returns:
            lxml Element representing the complete figure
        """
        pass


class ExternalImageStrategy(ImageEmbedStrategy):
    """Embeds images as separate files in EPUB package."""
    
    def embed_image(self, book: epub.EpubBook, 
                   cache_key: str, 
                   image_bytes: bytes) -> str:
        """Add image as external file in EPUB."""
        try:
            # Determine media type from extension
            media_type = "image/png" if cache_key.endswith(".png") else "image/jpeg"
            
            # Create epub.EpubImage
            img = epub.EpubImage()
            img.file_name = f"images/{cache_key}"
            img.media_type = media_type
            img.content = image_bytes
            
            # Add to book
            book.add_item(img)
            
            log_info(f"Added image to EPUB: {img.file_name}")
            return img.file_name
            
        except Exception as e:
            log_error(f"Failed to embed image {cache_key}: {e}")
            raise ImageEmbedError(f"Failed to embed image {cache_key}: {str(e)}")
    
    def create_figure_element(self, image_href: str, 
                            place: str,
                            config: EmbedderConfig,
                            xhtml_path: str) -> etree.Element:
        """Create XHTML figure with external image reference."""
        try:
            # Create namespace map for XHTML
            nsmap = {None: "http://www.w3.org/1999/xhtml"}
            
            # Build figure element
            figure = etree.Element("figure", {"class": config.figure_class}, nsmap=nsmap)
            if config.figure_style:
                figure.set("style", config.figure_style)
            
            # Calculate relative path based on XHTML location
            img_src = image_href
            if xhtml_path and '/' in xhtml_path:
                # XHTML is in a subdirectory, need to go up
                depth = xhtml_path.count('/')
                img_src = '../' * depth + image_href
            
            # Add image
            img = etree.SubElement(figure, "img", {
                "src": img_src,
                "alt": f"Map of {place}",
                "style": f"max-width: {config.max_image_width};"
            })
            
            # Add caption with italic styling
            caption_text = config.caption_template.format(place=place)
            figcaption = etree.SubElement(figure, "figcaption", {
                "style": "font-style: italic; text-align: center; font-size: 0.9em; color: #666; margin-top: 0.5em; padding: 0 1em;"
            })
            figcaption.text = caption_text
            
            return figure
            
        except Exception as e:
            log_error(f"Failed to create figure element for {place}: {e}")
            raise ImageEmbedError(f"Failed to create figure element: {str(e)}")


class InlineImageStrategy(ImageEmbedStrategy):
    """Embeds images as Base64 data URIs directly in XHTML."""
    
    def embed_image(self, book: epub.EpubBook, 
                   cache_key: str, 
                   image_bytes: bytes) -> str:
        """Convert image to Base64 data URI."""
        try:
            # Determine MIME type
            mime_type = "image/png" if cache_key.endswith(".png") else "image/jpeg"
            
            # Create data URI
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            data_uri = f"data:{mime_type};base64,{base64_data}"
            
            log_info(f"Created data URI for {cache_key} ({len(base64_data)} chars)")
            return data_uri
            
        except Exception as e:
            log_error(f"Failed to create data URI for {cache_key}: {e}")
            raise ImageEmbedError(f"Failed to create data URI: {str(e)}")
    
    def create_figure_element(self, image_href: str, 
                            place: str,
                            config: EmbedderConfig,
                            xhtml_path: str) -> etree.Element:
        """Create XHTML figure with inline Base64 image."""
        try:
            # Create namespace map for XHTML
            nsmap = {None: "http://www.w3.org/1999/xhtml"}
            
            # Build figure element
            figure = etree.Element("figure", {"class": config.figure_class}, nsmap=nsmap)
            if config.figure_style:
                figure.set("style", config.figure_style)
            
            # Add image with data URI
            img = etree.SubElement(figure, "img", {
                "src": image_href,  # This is the data URI
                "alt": f"Map of {place}",
                "style": f"max-width: {config.max_image_width};"
            })
            
            # Add caption with italic styling
            caption_text = config.caption_template.format(place=place)
            figcaption = etree.SubElement(figure, "figcaption", {
                "style": "font-style: italic; text-align: center; font-size: 0.9em; color: #666; margin-top: 0.5em; padding: 0 1em;"
            })
            figcaption.text = caption_text
            
            return figure
            
        except Exception as e:
            log_error(f"Failed to create inline figure element for {place}: {e}")
            raise ImageEmbedError(f"Failed to create inline figure element: {str(e)}")