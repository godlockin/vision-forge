"""Azure OpenAI service implementation."""

import asyncio
import time
from typing import Optional, Dict, Any, List
from .base import BaseService, ServiceResponse, ImageGenerationResponse


class AzureOpenAIService(BaseService):
    """
    Azure OpenAI service implementation.

    Supports GPT-4o, GPT-4o-mini, o1-preview, o1-mini models.
    """

    # Allowed models
    ALLOWED_MODELS = {
        "gpt-4o",
        "gpt-4o-mini",
        "o1-preview",
        "o1-mini"
    }

    def __init__(
        self,
        api_key: str,
        api_base: str,
        api_version: str = "2024-12-01-preview",
        model: str = "gpt-4o"
    ):
        """
        Initialize Azure OpenAI service.

        Args:
            api_key: Azure OpenAI API key
            api_base: Azure OpenAI endpoint URL
            api_version: API version
            model: Default model to use
        """
        super().__init__("azure_openai")

        if model not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Model {model} not allowed. "
                f"Allowed models: {self.ALLOWED_MODELS}"
            )

        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.api_version = api_version
        self.model = model
        self._client = None

    @property
    def client(self):
        """Get or create async OpenAI client."""
        if self._client is None:
            from openai import AsyncAzureOpenAI
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.api_base,
                api_version=self.api_version
            )
        return self._client

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ServiceResponse:
        """
        Generate text using Azure OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            ServiceResponse with generated text
        """
        start_time = time.time()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # o1 models don't support system prompt or temperature
        if self.model.startswith("o1"):
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        latency_ms = int((time.time() - start_time) * 1000)

        # Update stats
        self._increment_stats(response.usage.total_tokens)

        return ServiceResponse(
            content=response.choices[0].message.content or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            model=self.model,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason
        )

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> ImageGenerationResponse:
        """
        Generate image using DALL-E 3 via Azure OpenAI.

        Args:
            prompt: Text prompt for image
            negative_prompt: Not supported by DALL-E 3
            width: Image width (DALL-E 3 only supports 1024x1024)
            height: Image height
            **kwargs: Additional parameters

        Returns:
            ImageGenerationResponse
        """
        start_time = time.time()

        # DALL-E 3 only supports 1024x1024, 1792x1024, or 1024x1792
        size = self._get_dalle_size(width, height)

        response = await self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size=size,
            **kwargs
        )

        latency_ms = int((time.time() - start_time) * 1000)

        image_url = response.data[0].url if response.data else None

        return ImageGenerationResponse(
            image_url=image_url,
            image_data=None,  # Download separately if needed
            prompt_used=prompt,
            model="dall-e-3",
            latency_ms=latency_ms,
            negative_prompt=negative_prompt,
            parameters={"size": size}
        )

    def _get_dalle_size(self, width: int, height: int) -> str:
        """
        Get DALL-E 3 supported size.

        Args:
            width: Desired width
            height: Desired height

        Returns:
            DALL-E 3 size string
        """
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"

    def set_model(self, model: str):
        """
        Change the model to use.

        Args:
            model: New model name
        """
        if model not in self.ALLOWED_MODELS:
            raise ValueError(f"Model {model} not allowed")
        self.model = model
