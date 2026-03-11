"""Base service interfaces for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


@dataclass
class ServiceResponse:
    """Response from text generation service."""
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0
    finish_reason: Optional[str] = None

    def __post_init__(self):
        if not self.usage:
            self.usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }


@dataclass
class ImageGenerationResponse:
    """Response from image generation service."""
    image_url: Optional[str] = None
    image_data: Optional[bytes] = None
    prompt_used: str = ""
    model: str = ""
    latency_ms: int = 0
    negative_prompt: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class TaskType(str, Enum):
    """Type of task for model routing."""
    TEXT_REASONING = "text_reasoning"
    VISUAL_ANALYSIS = "visual_analysis"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    SUMMARY = "summary"
    COMPLIANCE_CHECK = "compliance_check"
    EMBEDDING = "embedding"


class BaseService(ABC):
    """
    Abstract base class for LLM services.

    All LLM provider implementations should inherit from this.
    """

    def __init__(self, provider_name: str):
        """
        Initialize base service.

        Args:
            provider_name: Name of the provider (e.g., "azure_openai", "vertex_ai")
        """
        self.provider_name = provider_name
        self._request_count = 0
        self._total_tokens = 0

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ServiceResponse:
        """
        Generate text response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ServiceResponse with generated text
        """
        pass

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> ImageGenerationResponse:
        """
        Generate image.

        Args:
            prompt: Text prompt for image
            negative_prompt: What to exclude from image
            width: Image width
            height: Image height
            **kwargs: Additional provider-specific parameters

        Returns:
            ImageGenerationResponse with image data
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "provider": self.provider_name,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens
        }

    def _increment_stats(self, tokens: int):
        """Update internal statistics."""
        self._request_count += 1
        self._total_tokens += tokens
