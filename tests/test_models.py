"""Tests for core models."""

import pytest
from vision_forge.core.models import (
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


class TestArchetype:
    """Test Archetype enum."""

    def test_archetype_values(self):
        assert Archetype.STATIC.value == "static"
        assert Archetype.DYNAMIC.value == "dynamic"


class TestLoadStrategy:
    """Test LoadStrategy enum."""

    def test_load_strategy_values(self):
        assert LoadStrategy.ALWAYS.value == "always"
        assert LoadStrategy.ON_DEMAND.value == "on_demand"
        assert LoadStrategy.DYNAMIC.value == "dynamic"


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.PROCESSING.value == "processing"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestPersona:
    """Test Persona model."""

    def test_persona_creation(self):
        persona = Persona(
            description="Test description",
            background="Test background",
            personality="Test personality"
        )
        assert persona.description == "Test description"
        assert persona.background == "Test background"
        assert persona.personality == "Test personality"


class TestCapability:
    """Test Capability model."""

    def test_capability_creation(self):
        cap = Capability(name="test_capability")
        assert cap.name == "test_capability"
        assert cap.params == {}

    def test_capability_with_params(self):
        cap = Capability(
            name="test_capability",
            params={"param1": "value1", "param2": 42}
        )
        assert cap.params == {"param1": "value1", "param2": 42}


class TestExpertConfig:
    """Test ExpertConfig model."""

    def test_expert_config_creation(self, sample_expert_config):
        assert sample_expert_config.id == "test_expert"
        assert sample_expert_config.role == "Test Expert"
        assert sample_expert_config.archetype == Archetype.STATIC
        assert sample_expert_config.load_strategy == LoadStrategy.ALWAYS

    def test_expert_config_default_values(self):
        config = ExpertConfig(
            id="minimal_expert",
            role="Minimal Expert",
            archetype=Archetype.STATIC,
            persona=Persona(
                description="Desc",
                background="Back",
                personality="Pers"
            )
        )
        assert config.load_strategy == LoadStrategy.ALWAYS
        assert config.thinking_framework == []
        assert config.strengths == []
        assert config.weaknesses == []
        assert config.blind_spots == []
        assert config.superhuman_insights == []
        assert config.capabilities == []

    def test_expert_config_serialization(self, sample_expert_config):
        # Test model_dump
        data = sample_expert_config.model_dump()
        assert data["id"] == "test_expert"
        assert data["role"] == "Test Expert"

        # Test model_validate
        new_config = ExpertConfig(**data)
        assert new_config.id == sample_expert_config.id


class TestTask:
    """Test Task model."""

    def test_task_creation(self):
        import time
        task = Task(
            id="task_001",
            type="image_generation",
            status=TaskStatus.PENDING,
            data={"prompt": "Generate an image"},
            created_at=int(time.time()),
            updated_at=int(time.time())
        )
        assert task.id == "task_001"
        assert task.type == "image_generation"
        assert task.status == TaskStatus.PENDING

    def test_task_default_status(self):
        import time
        task = Task(
            id="task_002",
            type="test",
            created_at=int(time.time()),
            updated_at=int(time.time())
        )
        assert task.status == TaskStatus.PENDING


class TestJob:
    """Test Job model."""

    def test_job_creation(self):
        import time
        job = Job(
            id="job_001",
            status=TaskStatus.PROCESSING,
            action="generate_image",
            data={"prompt": "Test prompt"},
            created_at=int(time.time()),
            updated_at=int(time.time())
        )
        assert job.id == "job_001"
        assert job.status == TaskStatus.PROCESSING
        assert job.action == "generate_image"
