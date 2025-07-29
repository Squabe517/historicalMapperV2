"""
Configuration for the embedder module.

Defines settings for how maps are embedded into EPUB files,
including styling, captions, and embedding strategy.
"""

from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    """Configuration for map embedding in EPUB files."""
    
    figure_class: str = "historical-map"
    figure_style: str = "margin: 1em 0; text-align: center;"
    caption_template: str = "Map of {place}"
    image_format: str = "png"
    max_image_width: str = "100%"
    embed_strategy: str = "external"  # "external" or "inline"
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.embed_strategy not in ["external", "inline"]:
            raise ValueError(f"Invalid embed_strategy: {self.embed_strategy}. Must be 'external' or 'inline'")
        
        if not self.caption_template:
            raise ValueError("caption_template cannot be empty")
        
        if "{place}" not in self.caption_template:
            raise ValueError("caption_template must contain {place} placeholder")
