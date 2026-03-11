"""Expert collaboration tests for Vision Forge.

Tests for expert modes, opinion scoring, dynamic expert creation, and inter-expert communication.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from vision_forge.core.models import (
    ExpertConfig, Persona, DecisionStyle, MemoryConfig,
    Archetype, LoadStrategy, Capability, IOSpec
)
from vision_forge.experts.expert import Expert
from vision_forge.experts.expert_registry import ExpertRegistry
from vision_forge.memory.blackboard import SharedBlackboard
from vision_forge.memory.events import BlackboardEvent, EventType
from vision_forge.services.router import ModelRouter, TaskType
from vision_forge.services.base import ServiceResponse


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def mock_blackboard_real():
    """Create mock blackboard with real async methods."""
    blackboard = MagicMock()
    blackboard.append = AsyncMock(return_value=True)
    blackboard.get_events = MagicMock(return_value=[])
    blackboard.get_current_round = MagicMock(return_value=1)
    blackboard.get_opinions = MagicMock(return_value=[])
    blackboard.get_scores = MagicMock(return_value=[])
    return blackboard


@pytest.fixture
def mock_router_with_responses():
    """Create mock router with configurable task-type responses."""
    router = MagicMock(spec=ModelRouter)

    responses = {
        TaskType.TEXT_REASONING: "Text reasoning response",
        TaskType.COMPLIANCE_CHECK: '{"passed": true, "violations": []}',
        TaskType.VISUAL_ANALYSIS: "Visual analysis result",
        TaskType.IMAGE_GENERATION: "Image generation complete",
    }

    async def route(task_type, *args, **kwargs):
        content = responses.get(task_type, "Default response")
        response = MagicMock(spec=ServiceResponse)
        response.content = content
        response.usage = {"total_tokens": 10}
        return response

    router.route_request = route
    return router


@pytest.fixture
def sample_expert_instance(sample_expert_config, mock_blackboard_real, mock_router_with_responses):
    """Create a concrete Expert instance for testing."""
    class TestExpert(Expert):
        async def process(self, task_data):
            return {"result": "processed", "data": task_data}

    return TestExpert(sample_expert_config, mock_blackboard_real, mock_router_with_responses)


# =============================================================================
# TestExpertModes
# =============================================================================

class TestExpertModes:
    """Test expert executor vs critic modes."""

    def test_executor_mode_switch(self, sample_expert_instance):
        """Test switching expert to executor mode."""
        # Default should be executor
        assert sample_expert_instance.is_executor is True

        # Explicitly set as executor
        sample_expert_instance.as_executor()
        assert sample_expert_instance.is_executor is True

        # Verify mode affects representation
        repr_str = repr(sample_expert_instance)
        assert "executor" in repr_str

    def test_critic_mode_switch(self, sample_expert_instance):
        """Test switching expert to critic mode."""
        sample_expert_instance.as_critic()
        assert sample_expert_instance.is_executor is False

        # Verify mode affects representation
        repr_str = repr(sample_expert_instance)
        assert "critic" in repr_str

    def test_mode_toggle_preserves_state(self, sample_expert_instance):
        """Test that toggling mode preserves other state."""
        # Set initial state
        sample_expert_instance.set_confidence(0.85)
        sample_expert_instance.record_action("test_action", {"result": "success"})

        # Toggle mode
        sample_expert_instance.as_critic()
        assert sample_expert_instance.is_executor is False

        # Verify other state preserved
        assert sample_expert_instance.get_confidence() == 0.85
        assert len(sample_expert_instance.get_history()) == 1

        # Toggle back
        sample_expert_instance.as_executor()
        assert sample_expert_instance.is_executor is True
        assert sample_expert_instance.get_confidence() == 0.85

    @pytest.mark.asyncio
    async def test_executor_vs_critic_same_expert(
        self,
        sample_expert_config,
        mock_blackboard_real,
        mock_router_with_responses
    ):
        """Test same expert behaves differently in executor vs critic modes."""
        class TestExpert(Expert):
            async def process(self, task_data):
                # Different behavior based on mode
                if self.is_executor:
                    return {"action": "execute", "mode": "executor"}
                else:
                    return {"action": "review", "mode": "critic"}

        executor = TestExpert(sample_expert_config, mock_blackboard_real, mock_router_with_responses)
        executor.as_executor()

        critic = TestExpert(sample_expert_config, mock_blackboard_real, mock_router_with_responses)
        critic.as_critic()

        # Process same task
        executor_result = await executor.process({"task_id": "test"})
        critic_result = await critic.process({"task_id": "test"})

        # Verify different outputs
        assert executor_result["mode"] == "executor"
        assert critic_result["mode"] == "critic"
        assert executor_result["action"] == "execute"
        assert critic_result["action"] == "review"


# =============================================================================
# TestOpinionScoring
# =============================================================================

class TestOpinionScoring:
    """Test experts scoring each other's opinions."""

    @pytest.mark.asyncio
    async def test_opinion_publish(
        self,
        sample_expert_instance,
        mock_blackboard_real
    ):
        """Test expert publishing opinion to blackboard."""
        await sample_expert_instance.publish_opinion(
            task_id="task_1",
            opinion="This design follows best practices",
            score=0.85,
            round_number=1
        )

        # Verify blackboard.append was called
        assert mock_blackboard_real.append.called
        call_args = mock_blackboard_real.append.call_args

        # Verify event structure
        event = call_args[0][0]
        assert isinstance(event, BlackboardEvent)
        assert event.type == EventType.OPINION_ADDED
        assert event.expert_id == "test_expert"
        assert event.task_id == "task_1"
        assert event.data["opinion"] == "This design follows best practices"
        assert event.data["score"] == 0.85

    @pytest.mark.asyncio
    async def test_score_publication(
        self,
        sample_expert_instance,
        mock_blackboard_real
    ):
        """Test expert scoring another expert's opinion."""
        await sample_expert_instance.publish_score(
            task_id="task_1",
            target_expert_id="visual_expert_01",
            score=0.9,
            rationale="Strong analysis with good reasoning"
        )

        # Verify blackboard.append was called
        assert mock_blackboard_real.append.called
        call_args = mock_blackboard_real.append.call_args

        # Verify score event structure
        event = call_args[0][0]
        assert event.type == EventType.SCORE_SUBMITTED
        assert event.data["target_expert"] == "visual_expert_01"
        assert event.data["score"] == 0.9
        assert event.data["rationale"] == "Strong analysis with good reasoning"

    @pytest.mark.asyncio
    async def test_opinion_scoring_weighted(
        self,
        mock_blackboard_real
    ):
        """Test weighted scoring of opinions."""
        task_id = "scoring_task"

        # Add opinions with different confidences directly to internal list
        opinions_data = [
            {"expert_id": "expert_a", "opinion": "Approve", "score": 0.9, "confidence": 0.95},
            {"expert_id": "expert_b", "opinion": "Approve", "score": 0.8, "confidence": 0.7},
            {"expert_id": "expert_c", "opinion": "Reject", "score": 0.3, "confidence": 0.5},
        ]

        events = []
        for opinion in opinions_data:
            event = BlackboardEvent(
                type=EventType.OPINION_ADDED,
                expert_id=opinion["expert_id"],
                task_id=task_id,
                data=opinion,
                round_number=1
            )
            events.append(event)

        # Mock get_events to return our events
        mock_blackboard_real.get_events = MagicMock(return_value=events)

        # Calculate weighted scores
        events = mock_blackboard_real.get_events(task_id=task_id)
        weighted_scores = []
        for event in events:
            base_score = event.data.get("score", 0.5)
            confidence = event.data.get("confidence", 0.5) or 0.5
            # Higher confidence = more weight: weight = 0.5 + confidence * 0.5
            weighted = base_score * (0.5 + confidence * 0.5)
            weighted_scores.append(weighted)

        assert len(weighted_scores) == 3, "Should have 3 weighted scores"
        avg_weighted = sum(weighted_scores) / len(weighted_scores)

        # Verify weighted scores are calculated correctly
        # expert_a: 0.9 * (0.5 + 0.95 * 0.5) = 0.9 * 0.975 = 0.8775
        # expert_b: 0.8 * (0.5 + 0.7 * 0.5) = 0.8 * 0.85 = 0.68
        # expert_c: 0.3 * (0.5 + 0.5 * 0.5) = 0.3 * 0.75 = 0.225
        # Average: (0.8775 + 0.68 + 0.225) / 3 = 0.594166...
        expected_avg = (0.9 * 0.975 + 0.8 * 0.85 + 0.3 * 0.75) / 3
        assert abs(avg_weighted - expected_avg) < 0.001, f"Expected ~{expected_avg}, got {avg_weighted}"

    @pytest.mark.asyncio
    async def test_cross_expert_review(
        self,
        sample_expert_config,
        mock_blackboard_real,
        mock_router_with_responses
    ):
        """Test multiple experts reviewing each other."""
        class TestExpert(Expert):
            async def process(self, task_data):
                return {"decision": "approved"}

        # Create two experts
        expert_a = TestExpert(sample_expert_config, mock_blackboard_real, mock_router_with_responses)
        expert_b = TestExpert(sample_expert_config, mock_blackboard_real, mock_router_with_responses)

        task_id = "review_task"

        # Expert A publishes opinion
        await expert_a.publish_opinion(task_id, "Design looks good", score=0.8)

        # Expert B reviews A's opinion
        await expert_b.publish_score(
            task_id,
            target_expert_id=expert_a.id,
            score=0.9,
            rationale="Good eye for detail"
        )

        # Verify both events published
        assert mock_blackboard_real.append.call_count >= 2


# =============================================================================
# TestDynamicExpert
# =============================================================================

class TestDynamicExpert:
    """Test dynamic expert creation and registration."""

    @pytest.mark.asyncio
    async def test_dynamic_expert_creation(
        self,
        mock_router_with_responses,
        tmp_path
    ):
        """Test HR creating new dynamic expert."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator

        # Create template file
        template_dir = tmp_path / "sys_init" / "settings"
        template_dir.mkdir(parents=True, exist_ok=True)
        template_file = template_dir / "dynamic_expert_template.yml"

        with open(template_file, 'w') as f:
            f.write("""---
expert:
  id: "dynamic_{{domain}}"
  archetype: "dynamic"
  load_strategy: "on_demand"
  memory:
    scope: "task"
    retention: "session"
---
""")

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry,
            template_path=str(template_file)
        )

        # Mock the LLM response to return valid JSON
        mock_response = MagicMock(spec=ServiceResponse)
        mock_response.content = '''{
            "role": "UI 设计专家",
            "persona": {
                "description": "UI 设计领域专家",
                "background": "10 年 UI 设计经验",
                "personality": "注重细节"
            },
            "thinking_framework": ["分析需求", "设计方案"],
            "strengths": ["UI 设计", "用户体验"],
            "weaknesses": ["后端知识有限"],
            "blind_spots": ["可能忽视性能"],
            "superhuman_insights": ["识别设计模式"],
            "capabilities": [
                {"name": "ui_analysis", "params": {}},
                {"name": "design_review", "params": {}}
            ],
            "i_o_spec": {
                "input": {"type": "design_requirements", "format": "json"},
                "output": {"type": "design_feedback", "format": "json"}
            },
            "decision_style": {
                "risk_tolerance": "medium",
                "consensus_need": "medium"
            },
            "trigger_conditions": ["UI 设计相关任务"]
        }'''
        mock_router_with_responses.route_request = AsyncMock(return_value=mock_response)

        # Generate expert
        config = await generator.generate_expert(
            domain_description="UI/UX design for mobile applications",
            requirements=["responsive design", "accessibility", "Material Design"]
        )

        # Verify expert created
        assert config is not None
        assert config.archetype == Archetype.DYNAMIC
        assert config.load_strategy == LoadStrategy.DYNAMIC
        assert len(config.capabilities) >= 1
        assert "UI" in config.role or "设计" in config.role

    @pytest.mark.asyncio
    async def test_dynamic_expert_registration(
        self,
        mock_router_with_responses,
        tmp_path
    ):
        """Test dynamic expert registration to registry."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry
        )

        # Create minimal valid config manually
        config = ExpertConfig(
            id="dynamic_test_expert",
            role="测试专家",
            archetype=Archetype.DYNAMIC,
            load_strategy=LoadStrategy.DYNAMIC,
            persona=Persona(
                description="测试领域专家",
                background="多年测试经验",
                personality="严谨细致"
            ),
            thinking_framework=["分析需求", "设计方案"],
            strengths=["测试能力"],
            capabilities=[Capability(name="test_capability", params={})],
            memory=MemoryConfig(scope="session", retention="persistent")
        )

        # Register expert
        result = generator.register_expert(config)

        # Verify registration
        assert result is True
        assert registry.has_expert("dynamic_test_expert")

        # Verify config stored - get_all_configs returns dict values, check by id
        configs = registry.get_all_configs()
        assert any(c.id == "dynamic_test_expert" for c in configs)

    @pytest.mark.asyncio
    async def test_dynamic_expert_persistence(
        self,
        mock_router_with_responses,
        tmp_path
    ):
        """Test dynamic expert saving to disk."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator
        from vision_forge.core.config_loader import save_expert_config

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry
        )

        # Create expert config
        config = ExpertConfig(
            id="dynamic_persist_test",
            role="持久化测试专家",
            archetype=Archetype.DYNAMIC,
            load_strategy=LoadStrategy.DYNAMIC,
            persona=Persona(
                description="持久化测试专家",
                background="测试背景",
                personality="专业"
            ),
            thinking_framework=["思考 1", "思考 2"],
            strengths=["能力 1"],
            capabilities=[Capability(name="persist_test", params={"key": "value"})],
            memory=MemoryConfig(scope="session", retention="persistent")
        )

        # Create output directory
        output_dir = tmp_path / "experts" / "dynamic"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Persist expert
        output_path = generator.persist_expert(config, output_dir=str(output_dir))

        # Verify file created
        import os
        assert os.path.exists(output_path)

        # Verify content can be loaded
        with open(output_path, 'r', encoding='utf-8') as f:
            import yaml
            loaded_data = yaml.safe_load(f)

        assert loaded_data is not None
        assert loaded_data["expert"]["id"] == "dynamic_persist_test"
        assert loaded_data["expert"]["role"] == "持久化测试专家"

    @pytest.mark.asyncio
    async def test_generate_and_register_workflow(
        self,
        mock_router_with_responses,
        tmp_path
    ):
        """Test complete generate-and-register workflow."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator

        # Create template
        template_dir = tmp_path / "sys_init" / "settings"
        template_dir.mkdir(parents=True, exist_ok=True)
        template_file = template_dir / "dynamic_expert_template.yml"

        with open(template_file, 'w') as f:
            f.write("""---
expert:
  id: "dynamic_expert"
  archetype: "dynamic"
  load_strategy: "dynamic"
  memory:
    scope: "task"
    retention: "session"
---
""")

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry,
            template_path=str(template_file)
        )

        # Mock all LLM calls
        async def mock_route(task_type, *args, **kwargs):
            response = MagicMock(spec=ServiceResponse)

            if "Generate" in str(args[0])[:20]:
                # Generation call
                response.content = '''{
                    "role": "测试专家",
                    "persona": {
                        "description": "测试专家",
                        "background": "测试背景",
                        "personality": "专业"
                    },
                    "thinking_framework": ["分析", "执行"],
                    "strengths": ["测试"],
                    "weaknesses": [],
                    "blind_spots": [],
                    "superhuman_insights": [],
                    "capabilities": [{"name": "test_cap", "params": {}}],
                    "i_o_spec": {
                        "input": {"type": "text"},
                        "output": {"type": "text"}
                    },
                    "decision_style": {
                        "risk_tolerance": "medium",
                        "consensus_need": "medium"
                    },
                    "trigger_conditions": []
                }'''
            else:
                # Feasibility scoring call
                response.content = '{"score": 0.85, "rationale": "Good fit"}'

            return response

        mock_router_with_responses.route_request = mock_route

        # Run complete workflow
        output_dir = tmp_path / "experts" / "dynamic"
        output_dir.mkdir(parents=True, exist_ok=True)

        config = await generator.generate_and_register(
            domain_description="测试领域",
            requirements=["测试能力"],
            task_requirements={"requirements": ["测试"]},
            minimum_feasibility=0.6
        )

        # Verify expert was created and registered
        assert config is not None
        assert registry.has_expert(config.id)


# =============================================================================
# TestFeasibilityScoring
# =============================================================================

class TestFeasibilityScoring:
    """Test expert feasibility scoring."""

    @pytest.mark.asyncio
    async def test_feasibility_score_calculation(
        self,
        mock_router_with_responses
    ):
        """Test feasibility score returned from LLM."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry
        )

        # Create test expert config
        config = ExpertConfig(
            id="test_expert",
            role="测试专家",
            archetype=Archetype.DYNAMIC,
            load_strategy=LoadStrategy.DYNAMIC,
            persona=Persona(
                description="测试专家",
                background="测试背景",
                personality="专业"
            ),
            thinking_framework=["思考"],
            strengths=["能力"],
            capabilities=[Capability(name="test", params={})]
        )

        # Mock response with score
        mock_response = MagicMock(spec=ServiceResponse)
        mock_response.content = '{"score": 0.85, "rationale": "Good match"}'
        mock_router_with_responses.route_request = AsyncMock(return_value=mock_response)

        # Score feasibility
        score = await generator.score_feasibility(
            config,
            {"requirements": ["测试"], "gap": {"description": "需要测试能力"}}
        )

        # Verify score in valid range
        assert 0.0 <= score <= 1.0
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_feasibility_rating_strings(
        self,
        mock_router_with_responses
    ):
        """Test feasibility rating string conversion."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry
        )

        # Test rating thresholds
        assert generator.get_feasibility_rating(0.9) == "EXCELLENT"
        assert generator.get_feasibility_rating(0.7) == "GOOD"
        assert generator.get_feasibility_rating(0.5) == "ACCEPTABLE"
        assert generator.get_feasibility_rating(0.3) == "POOR"

    @pytest.mark.asyncio
    async def test_feasibility_below_threshold(
        self,
        mock_router_with_responses
    ):
        """Test expert not registered when feasibility below threshold."""
        from vision_forge.experts.dynamic_expert_generator import DynamicExpertGenerator

        registry = ExpertRegistry()
        generator = DynamicExpertGenerator(
            model_router=mock_router_with_responses,
            registry=registry
        )

        # Create template
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""---
expert:
  id: "dynamic_expert"
  archetype: "dynamic"
---
""")
            template_path = f.name

        try:
            generator.template_path = template_path

            # Mock generation
            async def mock_route(task_type, *args, **kwargs):
                response = MagicMock(spec=ServiceResponse)
                if "Evaluate" in str(args[0])[:10]:
                    # Feasibility scoring - return low score
                    response.content = '{"score": 0.3, "rationale": "Poor fit"}'
                else:
                    # Generation
                    response.content = '''{
                        "role": "低分专家",
                        "persona": {"description": "D", "background": "B", "personality": "P"},
                        "thinking_framework": [],
                        "strengths": [],
                        "weaknesses": [],
                        "blind_spots": [],
                        "superhuman_insights": [],
                        "capabilities": [],
                        "i_o_spec": {"input": {}, "output": {}},
                        "decision_style": {"risk_tolerance": "medium", "consensus_need": "medium"},
                        "trigger_conditions": []
                    }'''
                return response

            mock_router_with_responses.route_request = mock_route

            # Attempt generate and register with threshold
            config = await generator.generate_and_register(
                domain_description="测试",
                requirements=[],
                task_requirements={},
                minimum_feasibility=0.6
            )

            # Should return None due to low feasibility
            assert config is None

        finally:
            os.unlink(template_path)


# =============================================================================
# TestExpertRegistryIntegration
# =============================================================================

class TestExpertRegistryIntegration:
    """Test ExpertRegistry integration scenarios."""

    def test_registry_load_from_directory(
        self,
        tmp_path
    ):
        """Test loading experts from directory."""
        import yaml

        # Create expert directory
        expert_dir = tmp_path / "experts" / "static"
        expert_dir.mkdir(parents=True, exist_ok=True)

        # Create expert config file
        expert_file = expert_dir / "test_expert.yml"
        with open(expert_file, 'w') as f:
            yaml.dump({
                "expert": {
                    "id": "loaded_expert_01",
                    "role": "加载测试专家",
                    "archetype": "static",
                    "load_strategy": "always",
                    "persona": {
                        "description": "测试专家",
                        "background": "测试背景",
                        "personality": "专业"
                    },
                    "thinking_framework": ["思考"],
                    "strengths": ["能力"],
                    "memory": {
                        "scope": "session",
                        "retention": "persistent"
                    }
                }
            }, f, allow_unicode=True)

        # Load from directory
        registry = ExpertRegistry.from_directory(str(expert_dir))

        # Verify expert loaded
        assert registry.has_expert("loaded_expert_01")

    def test_registry_create_instance_with_mode(
        self,
        mock_blackboard_real,
        mock_router_with_responses
    ):
        """Test creating expert instances with different modes."""
        registry = ExpertRegistry()

        # Register expert
        config = ExpertConfig(
            id="mode_test_expert",
            role="模式测试专家",
            archetype=Archetype.STATIC,
            load_strategy=LoadStrategy.ALWAYS,
            persona=Persona(
                description="测试专家",
                background="测试",
                personality="专业"
            )
        )
        registry.register_config(config)

        # Define expert class
        class ModeTestExpert(Expert):
            async def process(self, task_data):
                return {"mode": "executor" if self.is_executor else "critic"}

        registry.register_class("mode_test_expert", ModeTestExpert)

        # Create as executor
        executor = registry.create_instance(
            "mode_test_expert",
            mock_blackboard_real,
            mock_router_with_responses,
            as_executor=True
        )
        assert executor is not None
        assert executor.is_executor is True

        # Create as critic
        critic = registry.create_instance(
            "mode_test_expert",
            mock_blackboard_real,
            mock_router_with_responses,
            as_executor=False
        )
        assert critic is not None
        assert critic.is_executor is False

    def test_registry_stats(
        self,
        mock_blackboard_real,
        mock_router_with_responses
    ):
        """Test registry statistics."""
        registry = ExpertRegistry()

        # Register multiple experts
        for i in range(3):
            config = ExpertConfig(
                id=f"stat_expert_{i}",
                role=f"统计专家{i}",
                archetype=Archetype.STATIC,
                load_strategy=LoadStrategy.ALWAYS,
                persona=Persona(
                    description="测试",
                    background="测试",
                    personality="专业"
                )
            )
            registry.register_config(config)

        # Get stats
        stats = registry.get_stats()

        # Verify stats structure
        assert "registered_classes" in stats
        assert "registered_configs" in stats
        assert "active_instances" in stats
        assert stats["registered_configs"] >= 3

    def test_registry_get_configs_by_archetype(self):
        """Test filtering configs by archetype."""
        registry = ExpertRegistry()

        # Register static experts
        for i in range(2):
            registry.register_config(ExpertConfig(
                id=f"static_{i}",
                role=f"静态专家{i}",
                archetype=Archetype.STATIC,
                load_strategy=LoadStrategy.ALWAYS,
                persona=Persona(description="D", background="B", personality="P")
            ))

        # Register dynamic experts
        for i in range(2):
            registry.register_config(ExpertConfig(
                id=f"dynamic_{i}",
                role=f"动态专家{i}",
                archetype=Archetype.DYNAMIC,
                load_strategy=LoadStrategy.DYNAMIC,
                persona=Persona(description="D", background="B", personality="P")
            ))

        # Filter by archetype
        static_configs = registry.get_configs_by_archetype("static")
        dynamic_configs = registry.get_configs_by_archetype("dynamic")

        assert len(static_configs) == 2
        assert len(dynamic_configs) == 2


# =============================================================================
# TestHRExpertIntegration
# =============================================================================

class TestHRExpertIntegration:
    """Test HR Expert integration with dynamic expert generation."""

    @pytest.mark.asyncio
    async def test_hr_gap_analysis(
        self,
        mock_blackboard_real,
        mock_router_with_responses
    ):
        """Test HR expert performing gap analysis."""
        from vision_forge.experts.hr import HRExpert

        config = ExpertConfig(
            id="hr_expert_01",
            role="资深 HR 专家",
            archetype=Archetype.STATIC,
            load_strategy=LoadStrategy.ON_DEMAND,
            persona=Persona(
                description="资深 HR 专家",
                background="20 年人力资源管理经验",
                personality="善于识人、注重团队匹配"
            ),
            thinking_framework=["分析能力缺口", "评估团队匹配"],
            strengths=["人才招聘", "团队优化"]
        )

        # Mock LLM response for gap analysis
        mock_response = MagicMock(spec=ServiceResponse)
        mock_response.content = '''{
            "coverage_score": 0.6,
            "gaps": [
                {
                    "name": "3D 建模能力",
                    "description": "团队缺乏 3D 建模专家",
                    "priority": "high",
                    "rationale": "任务需要 3D 资产创建"
                }
            ],
            "covered_requirements": ["图像分析", "视觉评估"],
            "missing_requirements": ["3D 建模"]
        }'''
        mock_router_with_responses.route_request = AsyncMock(return_value=mock_response)

        hr_expert = HRExpert(config, mock_blackboard_real, mock_router_with_responses)

        # Perform gap analysis
        result = await hr_expert.gap_analysis(
            requirements=["图像分析", "视觉评估", "3D 建模"],
            current_team=["visual_expert_01"]
        )

        # Verify gap analysis result
        assert "coverage_score" in result
        assert "gaps" in result
        assert len(result["gaps"]) >= 1
        assert result["gaps"][0]["name"] == "3D 建模能力"

    @pytest.mark.asyncio
    async def test_hr_team_optimization(
        self,
        mock_blackboard_real,
        mock_router_with_responses
    ):
        """Test HR expert optimizing team composition."""
        from vision_forge.experts.hr import HRExpert

        config = ExpertConfig(
            id="hr_expert_01",
            role="资深 HR 专家",
            archetype=Archetype.STATIC,
            load_strategy=LoadStrategy.ON_DEMAND,
            persona=Persona(
                description="HR 专家",
                background="20 年 HR 经验",
                personality="善于团队优化"
            )
        )

        # Mock LLM response for team optimization
        mock_response = MagicMock(spec=ServiceResponse)
        mock_response.content = '["visual_expert_01", "prompt_engineer_01"]'
        mock_router_with_responses.route_request = AsyncMock(return_value=mock_response)

        hr_expert = HRExpert(config, mock_blackboard_real, mock_router_with_responses)

        # Optimize team
        optimized_team = await hr_expert.team_optimization(
            available_experts=["visual_expert_01", "prompt_engineer_01", "pm_expert_01"],
            task_requirements={"requirements": ["图像生成", "Prompt 优化"]}
        )

        # Verify team optimization
        assert isinstance(optimized_team, list)
        assert len(optimized_team) >= 1
        # Should select relevant experts
        assert "visual_expert_01" in optimized_team or "prompt_engineer_01" in optimized_team
