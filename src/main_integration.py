"""
Main entry point for Historical ePub Map Enhancer.

Demonstrates integration of config and logger modules with proper
startup sequence and error handling.
"""

import sys
import argparse
from pathlib import Path

from src.config.config_module import load_config, get_config, validate_config, ConfigError
from src.config.logger_module import initialize_logger, log_info, log_warning, log_error


def setup_application() -> bool:
    """
    Initialize application configuration and logging.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Load environment configuration
        load_config()
        
        # Initialize logging with config values
        log_level = get_config("LOG_LEVEL", "INFO")
        log_file = get_config("LOG_FILE", "logs/app.log")
        initialize_logger(log_level=log_level, log_file=log_file)
        
        log_info("Application startup initiated")
        
        # Validate required configuration keys
        required_keys = ["OPENAI_API_KEY", "GOOGLE_MAPS_API_KEY"]
        validate_config(required_keys)
        
        log_info("Application setup completed successfully")
        return True
        
    except ConfigError as e:
        # Configuration error - log and exit gracefully
        log_error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        print("Please check your .env file and ensure all required keys are set.", file=sys.stderr)
        return False
        
    except Exception as e:
        # Unexpected error during setup
        log_error(f"Unexpected error during setup: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Historical ePub Map Enhancer - Automatically embed maps into historical ePub books"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input ePub file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output ePub file path"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text chunk size for AI processing (default: 1000)"
    )
    
    parser.add_argument(
        "--zoom",
        type=int,
        default=12,
        help="Map zoom level (default: 12)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def validate_file_arguments(input_file: str, output_file: str) -> bool:
    """
    Validate input and output file arguments.
    
    Args:
        input_file: Path to input ePub file
        output_file: Path to output ePub file
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Check input file exists
    if not input_path.exists():
        log_error(f"Input file does not exist: {input_file}")
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False
    
    # Check input file is ePub
    if input_path.suffix.lower() != '.epub':
        log_error(f"Input file is not an ePub file: {input_file}")
        print(f"Error: Input file must be an ePub file (.epub extension).", file=sys.stderr)
        return False
    
    # Check output directory exists or can be created
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error(f"Cannot create output directory: {e}")
        print(f"Error: Cannot create output directory: {e}", file=sys.stderr)
        return False
    
    # Check output file extension
    if output_path.suffix.lower() != '.epub':
        log_warning(f"Output file does not have .epub extension: {output_file}")
        print(f"Warning: Output file should have .epub extension.", file=sys.stderr)
    
    return True


def process_epub(input_file: str, output_file: str, chunk_size: int, zoom: int) -> bool:
    """
    Process ePub file to add maps (placeholder for actual implementation).
    
    Args:
        input_file: Path to input ePub file
        output_file: Path to output ePub file
        chunk_size: Text chunk size for AI processing
        zoom: Map zoom level
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    log_info(f"Starting ePub processing: {input_file} -> {output_file}")
    log_info(f"Parameters: chunk_size={chunk_size}, zoom={zoom}")
    
    # TODO: This is where the actual ePub processing pipeline would go:
    # 1. Parse ePub and extract text (parser module)
    # 2. Chunk text for AI consumption (parser module)
    # 3. Extract place names using OpenAI (ai module)
    # 4. Normalize and filter places (ai module)
    # 5. Geocode places and fetch map images (mapping module)
    # 6. Embed images into ePub (embedder module)
    
    log_info("ePub processing completed successfully (placeholder)")
    print(f"Enhanced ePub saved to: {output_file}")
    
    return True


def main() -> int:
    """
    Main application entry point.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Setup application (config and logging)
    if not setup_application():
        return 1
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate file arguments
    if not validate_file_arguments(args.input, args.output):
        return 1
    
    # Set verbose logging if requested
    if args.verbose:
        log_info("Verbose mode enabled")
    
    # Process the ePub file
    try:
        success = process_epub(
            input_file=args.input,
            output_file=args.output,
            chunk_size=args.chunk_size,
            zoom=args.zoom
        )
        
        if success:
            log_info("Application completed successfully")
            return 0
        else:
            log_error("Application completed with errors")
            return 1
            
    except KeyboardInterrupt:
        log_warning("Application interrupted by user")
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 1
        
    except Exception as e:
        log_error(f"Unexpected error during processing: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
