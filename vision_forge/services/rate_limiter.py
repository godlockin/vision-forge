"""Rate limiting and traffic control for API requests."""

import asyncio
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Priority(str, Enum):
    """Request priority levels."""
    HIGH = "high"       # Compliance, PM decisions
    NORMAL = "normal"   # Visual expert, Prompt engineer
    LOW = "low"         # Interior designer, Knowledge admin


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    rpm_limit: int = 60           # Requests per minute
    tpm_limit: int = 500000       # Tokens per minute
    burst_size: int = 10          # Token bucket capacity


class TokenBucket:
    """
    Token bucket rate limiter.

    Implements both RPM (requests per minute) and TPM (tokens per minute) limits.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize token bucket.

        Args:
            config: Rate limit configuration
        """
        self.rpm_limit = config.rpm_limit
        self.tpm_limit = config.tpm_limit
        self.burst_size = config.burst_size

        # Current tokens
        self.request_tokens = float(config.burst_size)
        self.token_tokens = float(config.tpm_limit)

        # Last refill timestamps
        self.last_request_refill = time.time()
        self.last_token_refill = time.time()

        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of token tokens to acquire

        Returns:
            True if acquired, False if insufficient tokens
        """
        async with self._lock:
            self._refill()

            if self.request_tokens >= 1 and self.token_tokens >= tokens:
                self.request_tokens -= 1
                self.token_tokens -= tokens
                return True
            return False

    async def wait_for_token(
        self,
        tokens: int = 1,
        timeout: float = 30.0
    ) -> bool:
        """
        Wait until tokens are available or timeout.

        Args:
            tokens: Number of token tokens needed
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.acquire(tokens):
                return True
            await asyncio.sleep(0.1)

        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()

        # Refill request tokens (RPM)
        elapsed_minutes = (now - self.last_request_refill) / 60.0
        self.request_tokens = min(
            self.burst_size,
            self.request_tokens + elapsed_minutes * self.rpm_limit
        )
        self.last_request_refill = now

        # Refill token tokens (TPM)
        elapsed_minutes = (now - self.last_token_refill) / 60.0
        self.token_tokens = min(
            self.tpm_limit,
            self.token_tokens + elapsed_minutes * self.tpm_limit
        )
        self.last_token_refill = now

    def get_available_tokens(self) -> Dict[str, float]:
        """
        Get current available tokens.

        Returns:
            Dictionary with request and token tokens
        """
        self._refill()
        return {
            "request_tokens": self.request_tokens,
            "token_tokens": self.token_tokens
        }


class RateLimiter:
    """
    Global rate limiter with multiple provider support.

    Manages rate limiting across different API providers
    (Azure OpenAI, Vertex AI, etc.) with independent limits.
    """

    def __init__(self):
        """Initialize rate limiter."""
        self.limiters: Dict[str, TokenBucket] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._priority_queues: Dict[Priority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in Priority
        }

    def add_limiter(self, provider: str, config: RateLimitConfig):
        """
        Add rate limiter for a provider.

        Args:
            provider: Provider name
            config: Rate limit configuration
        """
        self.limiters[provider] = TokenBucket(config)
        self._locks[provider] = asyncio.Lock()

    async def acquire(
        self,
        provider: str,
        tokens: int = 1,
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """
        Acquire rate limit tokens for a provider.

        Args:
            provider: Provider name
            tokens: Number of token tokens
            priority: Request priority

        Returns:
            True if acquired, False if rate limited
        """
        if provider not in self.limiters:
            return True  # No limit configured

        limiter = self.limiters[provider]
        return await limiter.wait_for_token(tokens)

    def get_limiter(self, provider: str) -> Optional[TokenBucket]:
        """
        Get limiter for a provider.

        Args:
            provider: Provider name

        Returns:
            TokenBucket or None if not configured
        """
        return self.limiters.get(provider)

    @classmethod
    def from_config(cls, config: Dict[str, Dict[str, int]]) -> "RateLimiter":
        """
        Create rate limiter from configuration dictionary.

        Args:
            config: Dictionary with provider configurations

        Returns:
            Configured RateLimiter instance
        """
        limiter = cls()
        for provider, limits in config.items():
            limiter.add_limiter(provider, RateLimitConfig(**limits))
        return limiter

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all limiters.

        Returns:
            Dictionary with stats per provider
        """
        stats = {}
        for provider, limiter in self.limiters.items():
            tokens = limiter.get_available_tokens()
            stats[provider] = {
                "request_tokens": tokens["request_tokens"],
                "token_tokens": tokens["token_tokens"],
                "rpm_limit": limiter.rpm_limit,
                "tpm_limit": limiter.tpm_limit
            }
        return stats


class PriorityScheduler:
    """
    Scheduler for prioritized request handling.

    Ensures high-priority requests (compliance, PM decisions)
    are processed before lower-priority ones.
    """

    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize scheduler.

        Args:
            rate_limiter: Rate limiter instance
        """
        self.rate_limiter = rate_limiter
        self._queues: Dict[Priority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in Priority
        }
        self._running = False

    async def submit(
        self,
        request: Any,
        priority: Priority = Priority.NORMAL
    ):
        """
        Submit request to queue.

        Args:
            request: Request object
            priority: Request priority
        """
        await self._queues[priority].put(request)

    async def run(self, handler):
        """
        Run scheduler loop.

        Args:
            handler: Async function to handle requests
        """
        self._running = True

        while self._running:
            # Process high priority first
            for priority in [Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                if not self._queues[priority].empty():
                    request = await self._queues[priority].get()
                    try:
                        await handler(request)
                    except Exception as e:
                        print(f"Request failed: {e}")
                    break
            else:
                await asyncio.sleep(0.1)

    def stop(self):
        """Stop scheduler loop."""
        self._running = False
