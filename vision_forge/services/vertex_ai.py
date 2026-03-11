"""Google Vertex AI service implementation."""

import asyncio
import time
from typing import Optional, Dict, Any
from .base import BaseService, ServiceResponse, ImageGenerationResponse


class VertexAIService(BaseService):
    """
    Google Vertex AI service implementation.

    Supports Gemini 2.5/3.0 Pro/Flash for text and Imagen-3 for images.
    """

    # Allowed text models
    ALLOWED_TEXT_MODELS = {
        "gemini-2.0-pro",
        "gemini-2.0-flash",
        "gemini-3.0-pro",
        "gemini-3.0-flash"
    }

    # Allowed image models
    ALLOWED_IMAGE_MODELS = {
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001"
    }

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        credentials_path: Optional[str] = None,
        default_text_model: str = "gemini-2.0-flash",
        default_image_model: str = "imagen-3.0-generate-001"
    ):
        """
        Initialize Vertex AI service.

        Args:
            project_id: GCP project ID
            location: Vertex AI location
            credentials_path: Path to service account credentials
            default_text_model: Default text generation model
            default_image_model: Default image generation model
        """
        super().__init__("vertex_ai")

        self.project_id = project_id
        self.location = location
        self.credentials_path = credentials_path
        self.default_text_model = default_text_model
        self.default_image_model = default_image_model

        self._text_client = None
        self._image_client = None
        self._initialized = False

    def _initialize(self):
        """Initialize Vertex AI clients."""
        if self._initialized:
            return

        import vertexai
        vertexai.init(
            project=self.project_id,
            location=self.location,
            credentials=self._load_credentials()
        )

        self._initialized = True

    def _load_credentials(self):
        """Load credentials from file if specified."""
        if self.credentials_path:
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
        return None

    def _get_text_client(self):
        """Get text generation client."""
        self._initialize()
        if self._text_client is None:
            from vertexai.generative_models import GenerativeModel
            self._text_client = GenerativeModel(
                model_name=self.default_text_model,
                project=self.project_id,
                location=self.location
            )
        return self._text_client

    def _get_image_client(self):
        """Get image generation client."""
        self._initialize()
        if self._image_client is None:
            from vertexai.preview.vision_models import ImageGenerationModel
            self._image_client = ImageGenerationModel.from_pretrained(
                self.default_image_model
            )
        return self._image_client

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        **kwargs
    ) -> ServiceResponse:
        """
        Generate text using Gemini models.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters

        Returns:
            ServiceResponse with generated text
        """
        import vertexai
        from vertexai.generative_models import (
            GenerativeModel,
            GenerationConfig,
            SafetySetting,
            HarmCategory,
            HarmBlockThreshold
        )

        start_time = time.time()

        vertexai.init(project=self.project_id, location=self.location)
        model = GenerativeModel(
            model_name=self.default_text_model,
            system_instruction=[system_prompt] if system_prompt else None
        )

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )

        # Set safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        response = await model.generate_content_async(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract usage (Gemini doesn't provide exact token counts)
        usage = {"total_tokens": len(response.text) // 4} if response.text else {}

        self._increment_stats(usage.get("total_tokens", 0))

        return ServiceResponse(
            content=response.text if response.text else "",
            usage=usage,
            model=self.default_text_model,
            latency_ms=latency_ms,
            finish_reason="stop"
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
        Generate image using Imagen-3.

        Args:
            prompt: Text prompt for image
            negative_prompt: What to exclude from image
            width: Image width
            height: Image height
            **kwargs: Additional parameters

        Returns:
            ImageGenerationResponse with image data
        """
        from vertexai.preview.vision_models import ImageGenerationModel

        start_time = time.time()

        vertexai.init(project=self.project_id, location=self.location)
        model = ImageGenerationModel.from_pretrained(self.default_image_model)

        # Generate image
        response = await model.generate_image_async(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=self._get_aspect_ratio(width, height),
            **kwargs
        )

        latency_ms = int((time.time() - start_time) * 1000)

        self._increment_stats(1)

        return ImageGenerationResponse(
            image_url="",
            image_data=response.image_bytes if response else None,
            prompt_used=prompt,
            model=self.default_image_model,
            latency_ms=latency_ms,
            negative_prompt=negative_prompt,
            parameters={"aspect_ratio": self._get_aspect_ratio(width, height)}
        )

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """
        Get Imagen aspect ratio string.

        Args:
            width: Image width
            height: Image height

        Returns:
            Aspect ratio string
        """
        ratio = width / height
        if abs(ratio - 1.0) < 0.1:
            return "1:1"
        elif ratio > 1:
            return "16:9"
        else:
            return "9:16"

    def set_text_model(self, model: str):
        """
        Change the text generation model.

        Args:
            model: New model name
        """
        if model not in self.ALLOWED_TEXT_MODELS:
            raise ValueError(f"Model {model} not allowed")
        self.default_text_model = model
        self._text_client = None  # Reset client

    def set_image_model(self, model: str):
        """
        Change the image generation model.

        Args:
            model: New model name
        """
        if model not in self.ALLOWED_IMAGE_MODELS:
            raise ValueError(f"Model {model} not allowed")
        self.default_image_model = model
        self._image_client = None  # Reset client
