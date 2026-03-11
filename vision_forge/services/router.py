"""Model router for intelligent provider selection."""

from typing import Dict, Any, Optional, Tuple, Set
from enum import Enum

from .base import BaseService, ServiceResponse, ImageGenerationResponse, TaskType


class ModelRouter:
    """
    Smart model router based on task type and cost optimization.

    Routes requests to appropriate models based on:
    - Task complexity
    - Cost optimization
    - Provider availability
    - Quality requirements
    """

    # Allowed text models by provider
    ALLOWED_TEXT_MODELS: Dict[str, Set[str]] = {
        "azure": {"gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"},
        "vertex": {"gemini-2.0-pro", "gemini-2.0-flash", "gemini-3.0-pro", "gemini-3.0-flash"}
    }

    # Allowed image models by provider
    ALLOWED_IMAGE_MODELS: Dict[str, Set[str]] = {
        "vertex": {"imagen-3.0-generate-001", "imagen-3.0-fast-generate-001"}
    }

    # Model cost estimates (per 1K tokens)
    MODEL_COSTS: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "o1-preview": {"input": 0.015, "output": 0.060},
        "o1-mini": {"input": 0.003, "output": 0.012},
        "gemini-2.0-pro": {"input": 0.0025, "output": 0.0075},
        "gemini-2.0-flash": {"input": 0.00075, "output": 0.003},
    }

    def __init__(self, services: Dict[str, BaseService]):
        """
        Initialize model router.

        Args:
            services: Dictionary of provider name to service instance
        """
        self.services = services
        self._request_log: list = []

    @classmethod
    def from_env(cls) -> "ModelRouter":
        """
        Create router from environment configuration.

        Returns:
            ModelRouter instance with configured services

        Note:
            Requires environment variables:
            - OPENAI_API_KEY, OPENAI_API_BASE (for Azure OpenAI)
            - VERTEX_PROJECT_ID, VERTEX_LOCATION (for Vertex AI)
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()

        services = {}

        # Azure OpenAI
        if os.getenv("OPENAI_API_KEY"):
            from .azure_openai import AzureOpenAIService
            services["azure"] = AzureOpenAIService(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_API_BASE"),
                api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
            )

        # Vertex AI
        if os.getenv("VERTEX_PROJECT_ID"):
            from .vertex_ai import VertexAIService
            services["vertex"] = VertexAIService(
                project_id=os.getenv("VERTEX_PROJECT_ID"),
                location=os.getenv("VERTEX_LOCATION", "us-central1"),
                credentials_path=os.getenv("VERTEX_CREDENTIALS_PATH")
            )

        return cls(services)

    def get_best_model(self, task_type: TaskType) -> Tuple[str, BaseService]:
        """
        Get the best model for a task type.

        Args:
            task_type: Type of task

        Returns:
            Tuple of (model_name, service)

        Raises:
            ValueError: If no suitable service available
        """
        if task_type == TaskType.IMAGE_GENERATION:
            return self._get_image_model()

        if task_type == TaskType.SUMMARY:
            # Use cheapest model for summaries
            return self._get_cheapest_model()

        if task_type == TaskType.COMPLIANCE_CHECK:
            # Use most reliable model for compliance
            return self._get_most_reliable_model()

        if task_type == TaskType.EMBEDDING:
            # Use embedding-specific model
            return self._get_embedding_model()

        # Default: use GPT-4o or Gemini Pro for general reasoning
        return self._get_default_model()

    def _get_image_model(self) -> Tuple[str, BaseService]:
        """Get best image generation model."""
        if "vertex" in self.services:
            return "imagen-3.0-generate-001", self.services["vertex"]
        raise ValueError("No image generation service available")

    def _get_cheapest_model(self) -> Tuple[str, BaseService]:
        """Get cheapest available model."""
        # Prefer gpt-4o-mini or gemini-flash
        if "azure" in self.services:
            return "gpt-4o-mini", self.services["azure"]
        if "vertex" in self.services:
            return "gemini-2.0-flash", self.services["vertex"]
        raise ValueError("No cheap model available")

    def _get_most_reliable_model(self) -> Tuple[str, BaseService]:
        """Get most reliable model for critical tasks."""
        if "azure" in self.services:
            return "gpt-4o", self.services["azure"]
        if "vertex" in self.services:
            return "gemini-2.0-pro", self.services["vertex"]
        raise ValueError("No reliable model available")

    def _get_embedding_model(self) -> Tuple[str, BaseService]:
        """Get embedding model."""
        if "azure" in self.services:
            return "text-embedding-3-small", self.services["azure"]
        if "vertex" in self.services:
            return "text-embedding-005", self.services["vertex"]
        raise ValueError("No embedding model available")

    def _get_default_model(self) -> Tuple[str, BaseService]:
        """Get default model for general tasks."""
        if "azure" in self.services:
            return "gpt-4o", self.services["azure"]
        if "vertex" in self.services:
            return "gemini-2.0-pro", self.services["vertex"]
        raise ValueError("No service available")

    async def route_request(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> ServiceResponse | ImageGenerationResponse:
        """
        Route request to appropriate model.

        Args:
            task_type: Type of task
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Service response

        Raises:
            ValueError: If no suitable service
        """
        model_name, service = self.get_best_model(task_type)

        if task_type == TaskType.IMAGE_GENERATION:
            return await service.generate_image(prompt, **kwargs)
        else:
            return await service.generate_text(prompt, **kwargs)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if model not in self.MODEL_COSTS:
            return 0.0

        costs = self.MODEL_COSTS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    def get_available_models(self) -> Dict[str, list]:
        """
        Get list of available models by provider.

        Returns:
            Dictionary of provider to model list
        """
        return {
            provider: list(models)
            for provider, models in {
                "azure": self.ALLOWED_TEXT_MODELS.get("azure", set()),
                "vertex": self.ALLOWED_TEXT_MODELS.get("vertex", set())
            }.items()
            if provider in self.services
        }
