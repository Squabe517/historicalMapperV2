"""
OpenAI client for extracting place names from historical text chunks.

Uses GPT-4 with structured responses and retry logic for robust
place name extraction from ePub content.
"""

import json
import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.config_module import get_config


class OpenAIError(Exception):
    """Custom exception for OpenAI API failures."""
    pass


class OpenAIClient:
    """Client for extracting place names using OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI client with API key and model.
        
        Args:
            api_key: OpenAI API key (defaults to config)
            model: Model name (defaults to gpt-4o)
        """
        self.api_key = api_key or get_config("OPENAI_API_KEY")
        self.model = model or get_config("OPENAI_MODEL", "gpt-4o")
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise OpenAIError("OpenAI API key not provided")
        
        # Configure OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Rate limiting
        self.max_concurrent = int(get_config("OPENAI_MAX_CONCURRENT", "5"))
        self.semaphore = Semaphore(self.max_concurrent)
        
        self.logger.info(f"Initialized OpenAI client with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,))  # Will catch OpenAI exceptions
    )
    def analyze_chunk(self, chunk: str) -> List[str]:
        """
        Extract place names from a single text chunk using OpenAI.
        
        Args:
            chunk: Text content to analyze
            
        Returns:
            List of unique place names found
            
        Raises:
            OpenAIError: If API call fails after retries
        """
        if not chunk or not chunk.strip():
            return []
        
        start_time = time.time()
        
        try:
            with self.semaphore:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a historical geography expert. Extract all geographical "
                                "place names (cities, countries, regions, landmarks) from text. "
                                "Use modern names where possible. Return only a JSON array of strings."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Identify all historical or modern place names mentioned in the following text. "
                                f"Respond with a JSON array of unique place names (modern equivalents where known):\n\n"
                                f'"{chunk}"'
                            )
                        }
                    ],
                    temperature=0,
                )
            
            # Extract and parse response
            content = response.choices[0].message.content.strip()
            
            try:
                # Handle both direct array and object with array property
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    places = parsed
                elif isinstance(parsed, dict):
                    # Try common property names
                    places = (parsed.get("places") or 
                             parsed.get("place_names") or 
                             parsed.get("locations"))
                    
                    # If no standard property found, try first value if it's a list
                    if places is None and parsed:
                        first_value = list(parsed.values())[0]
                        if isinstance(first_value, list):
                            places = first_value
                        else:
                            places = []
                    elif places is None:
                        places = []
                else:
                    places = []
                
                # Ensure all items are strings
                places = [str(place).strip() for place in places if place]
                places = [place for place in places if place]  # Remove empty strings
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse OpenAI response as JSON: {content[:100]}...")
                raise OpenAIError(f"Invalid JSON response from OpenAI: {e}")
            
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
    
    def batch_analyze_chunks(self, chunks: List[str]) -> List[List[str]]:
        """
        Process multiple text chunks to extract place names.
        
        Args:
            chunks: List of text chunks to analyze
            
        Returns:
            List of place name lists, one per input chunk
            
        Raises:
            OpenAIError: If any chunk analysis fails after retries
        """
        if not chunks:
            self.logger.info("No chunks to analyze")
            return []
        
        self.logger.info(f"Sending {len(chunks)} chunks to OpenAI for analysis")
        
        # Process chunks with controlled concurrency
        results = [None] * len(chunks)  # Preserve order
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.analyze_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    error_msg = f"Failed to analyze chunk {index}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    results[index] = []  # Empty result for failed chunk
        
        # Report summary
        total_places = sum(len(result) for result in results)
        successful_chunks = len([r for r in results if r is not None])
        
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
            "max_concurrent": self.max_concurrent,
            "api_key_set": bool(self.api_key)
        }