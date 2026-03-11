"""Pytest fixtures for Vision Forge tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from vision_forge.core.models import (
    ExpertConfig, Persona, DecisionStyle, MemoryConfig, Archetype, LoadStrategy
)


@pytest.fixture
def mock_blackboard():
    """Create mock blackboard."""
    blackboard = MagicMock()
    blackboard.append = AsyncMock()
    blackboard.get_events = MagicMock(return_value=[])
    blackboard.get_current_round = MagicMock(return_value=0)
    return blackboard


@pytest.fixture
def mock_router():
    """Create mock model router."""
    router = MagicMock()
    router.route_request = AsyncMock(return_value=MagicMock(
        content="Mock response",
        usage={"total_tokens": 10}
    ))
    return router


@pytest.fixture
def sample_expert_config():
    """Create sample expert config."""
    return ExpertConfig(
        id="test_expert",
        role="Test Expert",
        archetype=Archetype.STATIC,
        load_strategy=LoadStrategy.ALWAYS,
        persona=Persona(
            description="Test description",
            background="Test background",
            personality="Test personality"
        ),
        thinking_framework=["Think step 1", "Think step 2"],
        strengths=["Strength 1", "Strength 2"],
        decision_style=DecisionStyle(
            risk_tolerance="medium",
            consensus_need="low"
        ),
        memory=MemoryConfig(scope="session", retention="persistent")
    )


@pytest.fixture
def pm_expert_config():
    """Create Project Manager expert config."""
    return ExpertConfig(
        id="project_manager_01",
        role="项目经理",
        archetype=Archetype.STATIC,
        load_strategy=LoadStrategy.ALWAYS,
        persona=Persona(
            description="经验丰富、果断的项目管理者",
            background="15 年科技项目管理经验",
            personality="务实、果断、以结果为导向"
        ),
        decision_style=DecisionStyle(
            risk_tolerance="medium",
            consensus_need="low"
        ),
        memory=MemoryConfig(scope="session", retention="persistent")
    )


@pytest.fixture
def compliance_expert_config():
    """Create Compliance expert config."""
    return ExpertConfig(
        id="compliance_legal_01",
        role="合规与法律专家",
        archetype=Archetype.STATIC,
        load_strategy=LoadStrategy.ALWAYS,
        persona=Persona(
            description="严谨、零妥协的法律与合规守门人",
            background="顶级律所合伙人，20 年知识产权、内容合规经验",
            personality="谨慎、原则性强、不留灰色地带"
        ),
        decision_style=DecisionStyle(
            risk_tolerance="very_low",
            consensus_need="none"
        ),
        memory=MemoryConfig(scope="persistent", retention="permanent"),
        veto_power={"enabled": True, "overrideable": False}
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes (minimal PNG)."""
    # Minimal valid PNG (1x1 pixel)
    return bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82
    ])
