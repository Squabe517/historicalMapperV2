# Embedder Module

The embedder module is responsible for inserting map images into EPUB files at appropriate locations based on AI-identified places and mapping results.

## Features

- **Direct Integration**: Works directly with outputs from AI and mapping modules
- **Multiple Strategies**: Supports external file embedding and inline Base64 data URIs
- **Smart Matching**: Fuzzy matching algorithm to link place names with cache keys
- **Error Recovery**: Graceful handling of missing images or invalid paragraphs
- **EPUB Integrity**: Maintains proper XHTML structure and namespaces

## Components

### EmbedderConfig
Configuration dataclass with validation:
- `figure_class`: CSS class for figure elements
- `figure_style`: Inline CSS for figures
- `caption_template`: Template with `{place}` placeholder
- `embed_strategy`: "external" or "inline"

### Image Embedding Strategies

#### ExternalImageStrategy
- Adds images as separate files in the EPUB package
- Creates proper EPUB image items with correct MIME types
- Uses relative paths in XHTML (`../images/...`)

#### InlineImageStrategy
- Embeds images as Base64 data URIs
- No external files needed
- Larger EPUB size but self-contained

### EpubMapEmbedder
Main orchestrator that:
1. Validates EPUB structure
2. Builds paragraph index across all XHTML documents
3. Matches AI-identified places with map images
4. Embeds figures after appropriate paragraphs
5. Updates EPUB content

## Usage

```python
from src.embedder import EpubMapEmbedder, EmbedderConfig

# Configure embedder
config = EmbedderConfig(
    figure_class="historical-map",
    caption_template="Map of {place}",
    embed_strategy="external"
)

# Create embedder
embedder = EpubMapEmbedder(config=config)

# Embed maps
embedder.embed_maps(
    book=epub_book,           # From parser
    ai_results=ai_results,    # From AI module
    map_images=map_images     # From mapping module
)

# Save enhanced EPUB
epub.write_epub("output.epub", book)
```

## Integration with Pipeline

```python
# 1. Parse EPUB
parser = EpubParser()
parser.load_file("book.epub")
paragraphs = parser.extract_text()
book = parser.book

# 2. Chunk text
chunks = TextChunker().chunk_text(paragraphs)

# 3. Extract places
ai_results = OpenAIClient().batch_analyze_chunks(chunks)

# 4. Get maps
places = [place for chunk in ai_results for place in chunk]
map_images = MappingOrchestrator().batch_get_maps(places)

# 5. Embed maps
embedder = EpubMapEmbedder()
embedder.embed_maps(book, ai_results, map_images)

# 6. Save
epub.write_epub("book_with_maps.epub", book)
```

## Error Handling

- **InvalidEpubStructureError**: EPUB has no documents or paragraphs
- **ParagraphNotFoundError**: Target paragraph index is invalid
- **ImageEmbedError**: Failed to embed image or create figure
- **Missing Images**: Logged and skipped, processing continues

## Cache Key Matching

The module uses a two-stage matching algorithm:
1. **Exact prefix match**: Looks for cache keys starting with normalized place name
2. **Fuzzy match**: Splits place name into words and finds keys containing all words

This handles variations in place name formatting between AI output and cache key generation.

## XHTML Handling

- Supports both namespaced and non-namespaced XHTML
- Preserves XML declarations and encoding
- Uses lxml's recovery parser for malformed content
- Maintains document structure when inserting figures

## Testing

Comprehensive test suite covers:
- Configuration validation
- Strategy implementations
- Paragraph indexing
- Cache key matching
- Full embedding workflow
- Error scenarios

Run tests:
```bash
pytest src/embedder/test_embedder.py -v
```

## Future Enhancements

1. **Smart Placement**: Use NLP to find best paragraph for each place
2. **Duplicate Detection**: Avoid embedding same map multiple times
3. **Image Optimization**: Resize/compress images before embedding
4. **CSS Injection**: Automatically add required CSS rules
5. **Progress Callbacks**: For UI integration
