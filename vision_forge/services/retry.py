"""Retry logic with exponential backoff."""

import asyncio
import random
from functools import wraps
from typing import Callable, Any, Optional, Tuple, Type
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    jitter: float = 0.1
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class RetryError(Exception):
    """Raised when all retry attempts fail."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration

    Returns:
        Decorated function

    Example:
        @with_retry(RetryConfig(max_attempts=5, initial_delay_ms=500))
        async def flaky_api_call():
            ...
    """

    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    # Don't delay on last attempt
                    if attempt == config.max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay_ms = config.initial_delay_ms * (config.backoff_multiplier ** attempt)
                    delay_ms = min(delay_ms, config.max_delay_ms)

                    # Add jitter to prevent thundering herd
                    jitter_range = delay_ms * config.jitter
                    delay_ms += random.uniform(-jitter_range, jitter_range)

                    await asyncio.sleep(delay_ms / 1000)

            raise RetryError(
                f"Failed after {config.max_attempts} attempts",
                last_exception=last_exception
            )

        return wrapper
    return decorator


class RetryHandler:
    """
    Programmatic retry handler for more control.

    Use when decorator pattern is not suitable.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self._last_exception: Optional[Exception] = None
        self._attempt_count = 0

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            RetryError: If all attempts fail
        """
        self._attempt_count = 0
        self._last_exception = None

        for attempt in range(self.config.max_attempts):
            self._attempt_count = attempt + 1
            try:
                return await func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                self._last_exception = e

                if attempt == self.config.max_attempts - 1:
                    break

                await self._wait(attempt)

        raise RetryError(
            f"Failed after {self.config.max_attempts} attempts",
            last_exception=self._last_exception
        )

    async def _wait(self, attempt: int):
        """
        Wait with exponential backoff.

        Args:
            attempt: Current attempt number
        """
        delay_ms = self.config.initial_delay_ms * (self.config.backoff_multiplier ** attempt)
        delay_ms = min(delay_ms, self.config.max_delay_ms)

        jitter_range = delay_ms * self.config.jitter
        delay_ms += random.uniform(-jitter_range, jitter_range)

        await asyncio.sleep(delay_ms / 1000)

    @property
    def last_exception(self) -> Optional[Exception]:
        """Get last exception from execution."""
        return self._last_exception

    @property
    def attempt_count(self) -> int:
        """Get number of attempts made."""
        return self._attempt_count


def retry_with_fallback(
    primary: Callable,
    fallback: Callable,
    config: Optional[RetryConfig] = None
):
    """
    Decorator that tries primary function, then fallback on failure.

    Args:
        primary: Primary async function
        fallback: Fallback async function
        config: Retry configuration for primary

    Returns:
        Decorated function that uses fallback on failure
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await with_retry(config)(primary)(*args, **kwargs)
            except RetryError:
                return await fallback(*args, **kwargs)
        return wrapper
    return decorator
