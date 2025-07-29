"""
Token bucket rate limiter for API requests.

Implements a single-threaded token bucket algorithm with exponential backoff
to prevent API quota exhaustion and handle rate limiting gracefully.
"""

import time
from typing import Optional

from ..config.logger_module import log_info
from .mapping_errors import RateLimitError


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API requests (single-threaded).
    
    The token bucket algorithm allows for burst traffic while maintaining
    an average rate over time. Tokens are added at a constant rate, and
    requests consume tokens. If no tokens are available, requests wait.
    """

    def __init__(self, 
                 rate_per_second: float = 5.0,
                 burst_capacity: int = 10,
                 retry_attempts: int = 3,
                 backoff_factor: float = 2.0):
        """
        Initialize the rate limiter.
        
        Args:
            rate_per_second: Tokens added per second (average rate)
            burst_capacity: Maximum tokens in bucket (burst allowance)
            retry_attempts: Max retries on rate limit
            backoff_factor: Exponential backoff multiplier
        """
        self.rate_per_second = rate_per_second
        self.burst_capacity = burst_capacity
        self.retry_attempts = retry_attempts
        self.backoff_factor = backoff_factor
        
        # Initialize bucket with full capacity
        self.tokens = float(burst_capacity)
        self.last_update = time.time()
        
        log_info(
            f"RateLimiter initialized: {rate_per_second}/sec, "
            f"burst capacity: {burst_capacity}"
        )

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate_per_second
        self.tokens = min(self.tokens + tokens_to_add, self.burst_capacity)
        self.last_update = current_time

    def acquire(self, tokens_needed: int = 1) -> bool:
        """
        Attempt to acquire tokens.
        
        Args:
            tokens_needed: Number of tokens required
            
        Returns:
            True if successful, False if rate limited
        """
        self._refill_tokens()
        
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False

    def wait_for_token(self, tokens_needed: int = 1) -> None:
        """
        Block until tokens are available, with exponential backoff.
        
        Args:
            tokens_needed: Number of tokens required
            
        Raises:
            RateLimitError: After max retry attempts
        """
        wait_time = 0.1  # Initial wait time in seconds
        
        for attempt in range(self.retry_attempts):
            if self.acquire(tokens_needed):
                return
            
            # Calculate time needed to accumulate required tokens
            self._refill_tokens()
            tokens_deficit = tokens_needed - self.tokens
            
            if tokens_deficit > 0:
                # Time to wait for required tokens
                time_to_wait = tokens_deficit / self.rate_per_second
                actual_wait = max(wait_time, time_to_wait)
                
                log_info(
                    f"Rate limited. Waiting {actual_wait:.2f}s "
                    f"(attempt {attempt + 1}/{self.retry_attempts})"
                )
                
                time.sleep(actual_wait)
                wait_time *= self.backoff_factor
            else:
                # Tokens available after refill
                if self.acquire(tokens_needed):
                    return
        
        raise RateLimitError(
            f"Failed to acquire {tokens_needed} token(s) after "
            f"{self.retry_attempts} attempts"
        )

    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        self._refill_tokens()
        return self.tokens

    def get_wait_time(self, tokens_needed: int = 1) -> float:
        """
        Calculate wait time for tokens without blocking.
        
        Args:
            tokens_needed: Number of tokens required
            
        Returns:
            Estimated wait time in seconds (0 if tokens available)
        """
        self._refill_tokens()
        
        if self.tokens >= tokens_needed:
            return 0.0
        
        tokens_deficit = tokens_needed - self.tokens
        return tokens_deficit / self.rate_per_second