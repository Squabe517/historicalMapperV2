"""
OpenAI client for extracting place names from historical text chunks.

Uses GPT-4 with structured responses and retry logic for robust
place name extraction from ePub content.
"""

import json
import logging
import time
from typing import List, Dict, Any

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.config_module import get_config


class OpenAIError(Exception):
    """Custom exception for OpenAI API failures."""
    pass


class Place(BaseModel):
    """Model for a geographical place with zoom level."""
    place: str
    zoom: int


class PlacesList(BaseModel):
    """Model for a list of places."""
    places: List[Place]


class OpenAIClient:
    """Client for extracting place names using OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI client with API key and model.
        
        Args:
            api_key: OpenAI API key (defaults to config)
            model: Model name (defaults to gpt-4o-2024-08-06 for structured outputs)
        """
        self.api_key = api_key or get_config("OPENAI_API_KEY")
        # Use a model that supports structured outputs
        self.model = model or get_config("OPENAI_MODEL", "gpt-4o-2024-08-06")
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise OpenAIError("OpenAI API key not provided")
        
        # Configure OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        self.logger.info(f"Initialized OpenAI client with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),  # Will catch OpenAI exceptions
        reraise=True
    )
    def analyze_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        """
        Extract place names from a single text chunk using OpenAI.
        
        Args:
            chunk: Text content to analyze
            
        Returns:
            List of dictionaries with 'place' and 'zoom' fields
            
        Raises:
            OpenAIError: If API call fails after retries
        """
        if not chunk or not chunk.strip():
            return []
        
        start_time = time.time()
        
        try:
            # Use structured outputs with Pydantic models
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a historical geography expert. Extract all geographical "
                            "place names (cities, countries, regions, landmarks) from text. "
                            "Use modern names where possible. For each place, provide an appropriate "
                            "Google Maps zoom level: "
                            "- Continents: 3-4"
                            "- Countries: 5-6"
                            "- States/Regions: 6-8"
                            "- Cities: 10-12"
                            "- Neighborhoods/Districts: 13-15"
                            "- Specific landmarks/buildings: 16-18"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Identify all historical or modern place names mentioned in the following text. "
                            f"For each place, determine the appropriate Google Maps zoom level based on its type. "
                            f"Use modern place names where known:\n\n"
                            f'"{chunk}"'
                        )
                    }
                ],
                temperature=0,
                response_format=PlacesList,
            )
            
            # Extract the parsed structured output
            parsed_output = response.choices[0].message.parsed
            
            # Convert Pydantic models to dictionaries
            places = [
                {"place": place.place, "zoom": place.zoom}
                for place in parsed_output.places
            ]
            
            elapsed = time.time() - start_time
            self.logger.debug(
                f"Analyzed chunk ({len(chunk)} chars) in {elapsed:.2f}s, "
                f"found {len(places)} places: {places[:3]}{'...' if len(places) > 3 else ''}"
            )
            
            return places
            
        except Exception as e:
            # Check if it's a rate limit or connection error for retry
            error_str = str(e).lower()
            if any(term in error_str for term in ['rate limit', 'connection', 'timeout', 'service unavailable']):
                self.logger.warning(f"OpenAI API error (will retry): {e}")
                raise
            else:
                self.logger.error(f"OpenAI API call failed: {e}")
                raise OpenAIError(f"OpenAI API call failed: {e}")
    
    def batch_analyze_chunks(self, chunks: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Process multiple text chunks to extract place names.
        
        Args:
            chunks: List of text chunks to analyze
            
        Returns:
            List of place lists, one per input chunk. Each place is a dict with 'place' and 'zoom'
            
        Raises:
            OpenAIError: If any chunk analysis fails after retries
        """
        if not chunks:
            self.logger.info("No chunks to analyze")
            return []
        
        self.logger.info(f"Sending {len(chunks)} chunks to OpenAI for analysis")
        
        # Process chunks sequentially
        results = []
        errors = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Log progress for longer batches
                if len(chunks) > 10 and i % 10 == 0:
                    self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                result = self.analyze_chunk(chunk)
                results.append(result)
                
            except Exception as e:
                error_msg = f"Failed to analyze chunk {i}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                results.append([])  # Empty result for failed chunk
        
        # Report summary
        total_places = sum(len(result) for result in results)
        successful_chunks = len([r for r in results if r])
        
        self.logger.info(
            f"Completed analysis: {successful_chunks}/{len(chunks)} chunks successful, "
            f"{total_places} total places found"
        )
        
        if errors:
            self.logger.warning(f"Encountered {len(errors)} errors during batch processing")
        
        return results
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring.
        
        Returns:
            Dictionary with usage information
        """
        return {
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "supports_structured_outputs": True
        }
    
    def extract_place_names_only(self, chunk: str) -> List[str]:
        """
        Legacy method to extract just place names without zoom levels.
        
        Args:
            chunk: Text content to analyze
            
        Returns:
            List of place name strings
        """
        places_with_zoom = self.analyze_chunk(chunk)
        return [place["place"] for place in places_with_zoom]