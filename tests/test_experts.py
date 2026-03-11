"""Tests for expert system components."""

import pytest
from unittest.mock import MagicMock

from vision_forge.experts.expert import Expert
from vision_forge.experts.expert_registry import ExpertRegistry
from vision_forge.core.models import (
    ExpertConfig, Persona, DecisionStyle, MemoryConfig,
    Archetype, LoadStrategy
)


class TestExpert:
    """Test Expert base class."""

    @pytest.fixture
    def expert_instance(self, sample_expert_config, mock_blackboard, mock_router):
        """Create a concrete Expert instance for testing."""
        class ConcreteExpert(Expert):
            async def process(self, task_data):
                return {"result": "processed"}

        return ConcreteExpert(sample_expert_config, mock_blackboard, mock_router)

    def test_expert_properties(self, expert_instance):
        assert expert_instance.id == "test_expert"
        assert expert_instance.role == "Test Expert"

    def test_confidence_setting(self, expert_instance):
        expert_instance.set_confidence(0.85)
        assert expert_instance.get_confidence() == 0.85

    def test_confidence_bounds(self, expert_instance):
        expert_instance.set_confidence(1.5)
        assert expert_instance.get_confidence() == 1.0

        expert_instance.set_confidence(-0.5)
        assert expert_instance.get_confidence() == 0.0

    def test_executor_mode(self, expert_instance):
        expert_instance.as_executor()
        assert expert_instance.is_executor is True

    def test_critic_mode(self, expert_instance):
        expert_instance.as_critic()
        assert expert_instance.is_executor is False

    @pytest.mark.asyncio
    async def test_publish_opinion(self, expert_instance, mock_blackboard):
        await expert_instance.publish_opinion(
            task_id="task_1",
            opinion="Test opinion",
            score=0.9
        )

        assert mock_blackboard.append.called

    @pytest.mark.asyncio
    async def test_record_action(self, expert_instance):
        expert_instance.record_action("test_action", {"result": "success"})

        history = expert_instance.get_history()
        assert len(history) == 1
        assert history[0]["action"] == "test_action"

    def test_clear_history(self, expert_instance):
        expert_instance.record_action("action1", {})
        expert_instance.record_action("action2", {})

        expert_instance.clear_history()
        assert len(expert_instance.get_history()) == 0

    def test_repr(self, expert_instance):
        expert_instance.as_executor()
        expert_instance.set_confidence(0.8)

        repr_str = repr(expert_instance)
        assert "test_expert" in repr_str
        assert "executor" in repr_str


class TestExpertRegistry:
    """Test ExpertRegistry class."""

    def test_register_class_and_config(self):
        registry = ExpertRegistry()

        class TestExpert(Expert):
            async def process(self, task_data):
                return {}

        registry.register_class("test_expert", TestExpert)
        registry.register_config(ExpertConfig(
            id="test_expert",
            role="Test Expert",
            archetype=Archetype.STATIC,
            persona=Persona(description="D", background="B", personality="P")
        ))

        assert "test_expert" in registry._expert_classes
        assert "test_expert" in registry._expert_configs

    def test_create_instance(self, mock_blackboard, mock_router):
        registry = ExpertRegistry()

        class TestExpert(Expert):
            async def process(self, task_data):
                return {}

        config = ExpertConfig(
            id="test_expert",
            role="Test Expert",
            archetype=Archetype.STATIC,
            persona=Persona(description="D", background="B", personality="P")
        )

        registry.register_class("test_expert", TestExpert)
        registry.register_config(config)

        expert = registry.create_instance(
            "test_expert",
            mock_blackboard,
            mock_router,
            as_executor=True
        )

        assert expert is not None
        assert expert.is_executor is True

    def test_create_nonexistent_expert(self, mock_blackboard, mock_router):
        registry = ExpertRegistry()

        expert = registry.create_instance(
            "nonexistent",
            mock_blackboard,
            mock_router
        )

        assert expert is None

    def test_get_instance(self, mock_blackboard, mock_router):
        registry = ExpertRegistry()

        class TestExpert(Expert):
            async def process(self, task_data):
                return {}

        config = ExpertConfig(
            id="test_expert",
            role="Test Expert",
            archetype=Archetype.STATIC,
            persona=Persona(description="D", background="B", personality="P")
        )

        registry.register_class("test_expert", TestExpert)
        registry.register_config(config)

        registry.create_instance("test_expert", mock_blackboard, mock_router)

        instance = registry.get_instance("test_expert")
        assert instance is not None

    def test_get_all_configs(self):
        registry = ExpertRegistry()

        for i in range(3):
            config = ExpertConfig(
                id=f"expert_{i}",
                role=f"Expert {i}",
                archetype=Archetype.STATIC,
                persona=Persona(description="D", background="B", personality="P")
            )
            registry.register_config(config)

        configs = registry.get_all_configs()
        assert len(configs) == 3

    def test_get_configs_by_archetype(self):
        registry = ExpertRegistry()

        registry.register_config(ExpertConfig(
            id="static_expert",
            role="Static Expert",
            archetype=Archetype.STATIC,
            persona=Persona(description="D", background="B", personality="P")
        ))

        registry.register_config(ExpertConfig(
            id="dynamic_expert",
            role="Dynamic Expert",
            archetype=Archetype.DYNAMIC,
            persona=Persona(description="D", background="B", personality="P")
        ))

        static_configs = registry.get_configs_by_archetype("static")
        dynamic_configs = registry.get_configs_by_archetype("dynamic")

        assert len(static_configs) == 1
        assert len(dynamic_configs) == 1

    def test_get_stats(self, mock_blackboard, mock_router):
        registry = ExpertRegistry.from_directory("experts/static")
        stats = registry.get_stats()

        assert "registered_classes" in stats
        assert "registered_configs" in stats
        assert "active_instances" in stats

    def test_has_expert(self):
        registry = ExpertRegistry()
        registry.register_config(ExpertConfig(
            id="existing_expert",
            role="Existing Expert",
            archetype=Archetype.STATIC,
            persona=Persona(description="D", background="B", personality="P")
        ))

        assert registry.has_expert("existing_expert")
        assert not registry.has_expert("nonexistent_expert")


class TestPMExpert:
    """Test ProjectManagerExpert implementation."""

    @pytest.mark.asyncio
    async def test_pm_expert_process(self, pm_expert_config, mock_blackboard, mock_router):
        from vision_forge.experts.pm import ProjectManagerExpert

        expert = ProjectManagerExpert(pm_expert_config, mock_blackboard, mock_router)

        result = await expert.process({
            "task_id": "task_1",
            "requirements": "Test requirements"
        })

        assert "decision" in result
        assert "rationale" in result

    @pytest.mark.asyncio
    async def test_deadlock_detection(self, pm_expert_config, mock_blackboard, mock_router):
        from vision_forge.experts.pm import ProjectManagerExpert

        expert = ProjectManagerExpert(pm_expert_config, mock_blackboard, mock_router)

        # Create deadlock scenario
        opinions = [{"opinion": "A"}, {"opinion": "B"}, {"opinion": "C"}, {"opinion": "D"}, {"opinion": "E"}]

        assert expert._is_deadlock(opinions) is True

        # Non-deadlock scenario
        opinions = [{"opinion": "A"}, {"opinion": "A"}, {"opinion": "A"}]
        assert expert._is_deadlock(opinions) is False


class TestComplianceExpert:
    """Test ComplianceExpert implementation."""

    @pytest.mark.asyncio
    async def test_compliance_expert_veto_power(
        self, compliance_expert_config, mock_blackboard, mock_router
    ):
        from vision_forge.experts.compliance import ComplianceExpert

        expert = ComplianceExpert(compliance_expert_config, mock_blackboard, mock_router)

        assert expert.has_veto_power() is True

    def test_can_autofallback(self, compliance_expert_config, mock_blackboard, mock_router):
        from vision_forge.experts.compliance import ComplianceExpert

        expert = ComplianceExpert(compliance_expert_config, mock_blackboard, mock_router)

        # Medium severity should allow fallback
        assert expert.can_autofallback("copyright_ip") is True
        assert expert.can_autofallback("brand_trademark") is True

        # High/Critical should not allow fallback
        assert expert.can_autofallback("political_sensitive") is False
        assert expert.can_autofallback("violence_gore") is False

    def test_get_severity(self, compliance_expert_config, mock_blackboard, mock_router):
        from vision_forge.experts.compliance import ComplianceExpert, ViolationSeverity

        expert = ComplianceExpert(compliance_expert_config, mock_blackboard, mock_router)

        assert expert.get_severity("copyright_ip") == ViolationSeverity.MEDIUM
        assert expert.get_severity("political_sensitive") == ViolationSeverity.CRITICAL
