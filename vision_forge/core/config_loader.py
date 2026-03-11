"""Configuration loader for expert system."""

import yaml
from pathlib import Path
from typing import List, Optional

from .models import ExpertConfig


def load_expert_config(path: str) -> ExpertConfig:
    """
    Load expert configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        ExpertConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If no expert config found in file
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Expert config not found: {path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        # Handle multi-document YAML
        docs = list(yaml.safe_load_all(f))
        for doc in docs:
            if doc and 'expert' in doc:
                return ExpertConfig(**doc['expert'])

    raise ValueError(f"No expert config found in {path}")


def load_all_static_experts(directory: str = "experts/static") -> List[ExpertConfig]:
    """
    Load all static expert configurations from directory.

    Args:
        directory: Directory containing expert YAML configs

    Returns:
        List of ExpertConfig objects
    """
    experts = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return experts

    for path in sorted(dir_path.glob("*.yml")):
        try:
            config = load_expert_config(str(path))
            experts.append(config)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    return experts


def save_expert_config(config: ExpertConfig, path: str) -> None:
    """
    Save expert configuration to YAML file.

    Args:
        config: ExpertConfig object
        path: Path to save YAML file
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            {'expert': config.model_dump()},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )
