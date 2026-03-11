"""Expert registry for managing expert instances."""

from typing import Dict, List, Type, Optional
from pathlib import Path

from ..core.models import ExpertConfig
from ..core.config_loader import load_expert_config, load_all_static_experts
from .expert import Expert


class ExpertRegistry:
    """
    Registry for expert classes and instances.

    Manages:
    - Expert class registration
    - Expert configuration storage
    - Expert instance creation
    """

    def __init__(self):
        """Initialize registry."""
        self._expert_classes: Dict[str, Type[Expert]] = {}
        self._active_instances: Dict[str, Expert] = {}
        self._expert_configs: Dict[str, ExpertConfig] = {}

    def register_class(self, expert_id: str, expert_class: Type[Expert]):
        """
        Register an expert class.

        Args:
            expert_id: Expert ID to register for
            expert_class: Expert class type
        """
        self._expert_classes[expert_id] = expert_class

    def register_config(self, config: ExpertConfig):
        """
        Register an expert configuration.

        Args:
            config: Expert configuration
        """
        self._expert_configs[config.id] = config

    def create_instance(
        self,
        expert_id: str,
        blackboard,
        model_router,
        as_executor: bool = True
    ) -> Optional[Expert]:
        """
        Create expert instance.

        Args:
            expert_id: Expert ID to create
            blackboard: Shared blackboard instance
            model_router: Model router instance
            as_executor: Whether to create as executor (vs critic)

        Returns:
            Expert instance or None if not found
        """
        if expert_id not in self._expert_classes:
            return None

        expert_class = self._expert_classes[expert_id]
        config = self._expert_configs.get(expert_id)

        if not config:
            return None

        expert = expert_class(config, blackboard, model_router)

        if as_executor:
            expert.as_executor()
        else:
            expert.as_critic()

        self._active_instances[expert_id] = expert
        return expert

    def get_instance(self, expert_id: str) -> Optional[Expert]:
        """
        Get active expert instance.

        Args:
            expert_id: Expert ID

        Returns:
            Expert instance or None
        """
        return self._active_instances.get(expert_id)

    def get_all_configs(self) -> List[ExpertConfig]:
        """
        Get all registered configurations.

        Returns:
            List of ExpertConfig objects
        """
        return list(self._expert_configs.values())

    def get_configs_by_archetype(self, archetype: str) -> List[ExpertConfig]:
        """
        Get configs filtered by archetype.

        Args:
            archetype: "static" or "dynamic"

        Returns:
            List of matching ExpertConfig objects
        """
        return [
            c for c in self._expert_configs.values()
            if c.archetype.value == archetype
        ]

    def get_configs_by_load_strategy(self, strategy: str) -> List[ExpertConfig]:
        """
        Get configs filtered by load strategy.

        Args:
            strategy: "always", "on_demand", or "dynamic"

        Returns:
            List of matching ExpertConfig objects
        """
        return [
            c for c in self._expert_configs.values()
            if c.load_strategy.value == strategy
        ]

    @classmethod
    def from_directory(cls, directory: str = "experts/static") -> "ExpertRegistry":
        """
        Create registry from YAML configs in directory.

        Args:
            directory: Directory containing expert YAML files

        Returns:
            Configured ExpertRegistry instance
        """
        registry = cls()

        # Load all static expert configs
        configs = load_all_static_experts(directory)
        for config in configs:
            registry.register_config(config)

        return registry

    def load_additional_configs(self, directory: str):
        """
        Load additional configs from directory.

        Args:
            directory: Directory containing YAML files
        """
        configs = load_all_static_experts(directory)
        for config in configs:
            if config.id not in self._expert_configs:
                self.register_config(config)

    def get_expert_ids(self) -> List[str]:
        """
        Get list of registered expert IDs.

        Returns:
            List of expert IDs
        """
        return list(self._expert_configs.keys())

    def has_expert(self, expert_id: str) -> bool:
        """
        Check if expert is registered.

        Args:
            expert_id: Expert ID

        Returns:
            True if registered
        """
        return expert_id in self._expert_configs

    def remove_instance(self, expert_id: str):
        """
        Remove expert instance from registry.

        Args:
            expert_id: Expert ID to remove
        """
        if expert_id in self._active_instances:
            del self._active_instances[expert_id]

    def clear_instances(self):
        """Clear all expert instances."""
        self._active_instances.clear()

    def get_stats(self) -> Dict[str, int]:
        """
        Get registry statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "registered_classes": len(self._expert_classes),
            "registered_configs": len(self._expert_configs),
            "active_instances": len(self._active_instances),
            "static_experts": len(self.get_configs_by_archetype("static")),
            "dynamic_experts": len(self.get_configs_by_archetype("dynamic")),
            "always_load": len(self.get_configs_by_load_strategy("always")),
            "on_demand": len(self.get_configs_by_load_strategy("on_demand")),
        }
