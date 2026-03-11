"""Vision Forge - Intelligent Vision Reviewer.

An expert system for AI image evaluation and optimization.
"""

from .core.models import (
    Archetype,
    LoadStrategy,
    TaskStatus,
    Persona,
    Capability,
    IOSpec,
    DecisionStyle,
    MemoryConfig,
    ExpertConfig,
    Task,
    Job,
)
from .core.config_loader import load_expert_config, load_all_static_experts
from .experts.expert import Expert
from .experts.expert_registry import ExpertRegistry
from .memory.blackboard import SharedBlackboard, BlackboardEvent, EventType
from .memory.manager import MemoryManager
from .services.base import BaseService, ServiceResponse, ImageGenerationResponse, TaskType
from .services.router import ModelRouter

__version__ = "0.1.0"
__author__ = "Vision Forge Team"

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Core models
    "Archetype",
    "LoadStrategy",
    "TaskStatus",
    "Persona",
    "Capability",
    "IOSpec",
    "DecisionStyle",
    "MemoryConfig",
    "ExpertConfig",
    "Task",
    "Job",

    # Config loader
    "load_expert_config",
    "load_all_static_experts",

    # Experts
    "Expert",
    "ExpertRegistry",

    # Memory
    "SharedBlackboard",
    "BlackboardEvent",
    "EventType",
    "MemoryManager",

    # Services
    "BaseService",
    "ServiceResponse",
    "ImageGenerationResponse",
    "TaskType",
    "ModelRouter",
]
