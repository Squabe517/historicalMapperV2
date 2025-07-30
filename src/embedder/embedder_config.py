"""
Configuration for the embedder module.

Defines settings for how maps are embedded into EPUB files,
including styling, captions, and embedding strategy.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbedderConfig:
    """Configuration for map embedding in EPUB files."""
    
    # CSS class for figure elements
    figure_class: str = "historical-map"
    
    # Default figure style (overridden by strategies for side-by-side display)
    figure_style: str = "margin: 1em 0; text-align: center;"
    
    # Width for individual figures when displaying side-by-side
    figure_width: str = "45%"  # Allows 2 maps per row with 5% gap between
    
    # Caption template with {place} placeholder
    caption_template: str = "Map of {place}"
    
    # Image format
    image_format: str = "png"
    
    # Maximum image width within figure
    max_image_width: str = "100%"
    
    # Embedding strategy: "external" (separate files) or "inline" (base64)
    embed_strategy: str = "external"
    
    # Container settings for grouping maps
    container_class: str = "map-container"
    maps_per_row: int = 2
    
    # Map image dimensions (for fetching from API)
    map_size: str = "600x400"
    
    # Optional custom styles
    container_style: Optional[str] = None
    image_style: Optional[str] = None
    caption_style: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.embed_strategy not in ["external", "inline"]:
            raise ValueError(
                f"Invalid embed_strategy: {self.embed_strategy}. "
                f"Must be 'external' or 'inline'"
            )
        
        if not self.caption_template:
            raise ValueError("caption_template cannot be empty")
        
        if "{place}" not in self.caption_template:
            raise ValueError("caption_template must contain {place} placeholder")
        
        # Validate figure width
        if self.figure_width:
            # Remove % sign if present for validation
            width_str = self.figure_width.rstrip('%')
            try:
                width_num = float(width_str)
                if not 0 < width_num <= 100:
                    raise ValueError(
                        f"figure_width must be between 0 and 100%, got {self.figure_width}"
                    )
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(
                        f"Invalid figure_width: {self.figure_width}. "
                        f"Must be a percentage (e.g., '45%')"
                    )
                raise
        
        # Validate maps per row
        if not 1 <= self.maps_per_row <= 4:
            raise ValueError(
                f"maps_per_row must be between 1 and 4, got {self.maps_per_row}"
            )
        
        # Calculate optimal figure width based on maps per row if not set
        if not self.figure_width and self.maps_per_row > 1:
            # Account for margins between figures
            margin_percent = 2.5  # per side
            total_margin = margin_percent * 2 * self.maps_per_row
            available_width = 100 - total_margin
            self.figure_width = f"{available_width / self.maps_per_row:.1f}%"
    
    def get_container_style(self) -> str:
        """Get the container style, using default if not custom style provided."""
        if self.container_style:
            return self.container_style
        
        return (
            "margin: 1em 0; "
            "clear: both; "
            "overflow: hidden; "
            "text-align: center;"
        )
    
    def get_figure_style(self) -> str:
        """Get the figure style for side-by-side display."""
        return (
            f"display: inline-block; "
            f"width: {self.figure_width}; "
            f"margin: 0.5em 2.5%; "  # 5% total gap between figures
            f"text-align: center; "
            f"vertical-align: top; "
            f"box-sizing: border-box;"
        )
    
    def get_image_style(self) -> str:
        """Get the image style, using default if not custom style provided."""
        if self.image_style:
            return self.image_style
        
        return (
            f"max-width: {self.max_image_width}; "
            "height: auto; "
            "display: block; "
            "margin: 0 auto; "
            "border: 1px solid #ddd; "
            "border-radius: 4px; "
            "box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
        )
    
    def get_caption_style(self) -> str:
        """Get the caption style, using default if not custom style provided."""
        if self.caption_style:
            return self.caption_style
        
        return (
            "font-size: 0.85em; "
            "color: #666; "
            "font-style: italic; "
            "margin-top: 0.5em; "
            "line-height: 1.3; "
            "font-family: Georgia, serif;"
        )