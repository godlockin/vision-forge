"""Dynamic Expert Generator for creating new experts on-demand.

This module provides the capability to generate dynamic experts when:
- Current experts lack required capabilities
- Task requires specialized domain knowledge
- HR Expert identifies capability gaps

The generator uses LLM to populate expert templates based on domain descriptions,
scores feasibility, and registers new experts to the registry.
"""

import yaml
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from ..core.models import (
    ExpertConfig,
    Archetype,
    LoadStrategy,
    Persona,
    Capability,
    IOSpec,
    DecisionStyle,
    MemoryConfig
)
from ..experts.expert_registry import ExpertRegistry
from ..experts.expert import Expert
from ..services.router import ModelRouter, TaskType


class DynamicExpertGenerator:
    """
    Dynamic Expert Generator for creating new experts on-demand.

    Responsibilities:
    - Generate expert configs from YAML templates
    - Use LLM to populate expert details based on domain description
    - Score feasibility of new experts
    - Register new experts to registry
    - Persist dynamic expert configs to disk

    Usage:
        generator = DynamicExpertGenerator(model_router, registry)
        config = await generator.generate_expert(
            domain_description="UI/UX design for mobile apps",
            requirements=["responsive layout", "accessibility", " Material Design"]
        )
        feasibility = await generator.score_feasibility(config, task_requirements)
        if feasibility > 0.6:
            generator.register_expert(config)
            generator.persist_expert(config)
    """

    # Default template path
    DEFAULT_TEMPLATE_PATH = "sys_init/settings/dynamic_expert_template.yml"

    # Feasibility score thresholds
    FEASIBILITY_EXCELLENT = 0.8
    FEASIBILITY_GOOD = 0.6
    FEASIBILITY_MINIMUM = 0.4

    def __init__(
        self,
        model_router: ModelRouter,
        registry: ExpertRegistry,
        template_path: str = "sys_init/settings/dynamic_expert_template.yml"
    ):
        """
        Initialize Dynamic Expert Generator.

        Args:
            model_router: Model router for LLM calls
            registry: Expert registry for registering new experts
            template_path: Path to YAML template file
        """
        self.model_router = model_router
        self.registry = registry
        self.template_path = template_path
        self._template: Optional[Dict[str, Any]] = None

    def _load_template(self) -> Dict[str, Any]:
        """
        Load expert template from YAML file.

        Returns:
            Template dictionary

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template is invalid
        """
        if self._template is not None:
            return self._template

        template_file = Path(self.template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Expert template not found: {self.template_path}")

        with open(template_file, 'r', encoding='utf-8') as f:
            # Load all YAML documents in the file
            docs = list(yaml.safe_load_all(f))

            # Find the expert template document
            for doc in docs:
                if doc and 'expert' in doc:
                    self._template = doc['expert']
                    return self._template

        raise ValueError(f"No expert template found in {self.template_path}")

    async def generate_expert(
        self,
        domain_description: str,
        requirements: List[str]
    ) -> ExpertConfig:
        """
        Generate a new expert configuration using LLM.

        Uses the template and LLM to create a complete expert configuration
        based on the domain description and requirements.

        Args:
            domain_description: Description of the domain/field
            requirements: List of required capabilities/skills

        Returns:
            ExpertConfig for the new dynamic expert

        Raises:
            ValueError: If generation fails
        """
        # Load template
        template = self._load_template()

        # Build prompt for LLM to fill template
        prompt = self._build_generation_prompt(domain_description, requirements, template)

        # Call LLM to generate expert config
        response = await self.model_router.route_request(
            TaskType.TEXT_REASONING,
            prompt,
            max_tokens=2500
        )

        # Parse response into ExpertConfig
        config = self._parse_llm_response(response.content, domain_description, requirements)

        return config

    def _build_generation_prompt(
        self,
        domain_description: str,
        requirements: List[str],
        template: Dict[str, Any]
    ) -> str:
        """
        Build prompt for LLM to generate expert config.

        Args:
            domain_description: Domain description
            requirements: List of requirements
            template: Template structure

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert system architect tasked with creating a new dynamic expert.

DOMAIN DESCRIPTION:
{domain_description}

TASK REQUIREMENTS:
{chr(10).join(f'- {req}' for req in requirements)}

Generate a complete expert configuration based on the template structure below.
The expert should be specifically tailored to the domain and requirements.

TEMPLATE STRUCTURE:
{yaml.dump(template, default_flow_style=False, allow_unicode=True)}

Return ONLY a valid JSON object with the following structure. Do not include any explanation or commentary:

{{
    "role": "Clear role name in Chinese (e.g., UI 排版专家，3D 资产专家)",
    "persona": {{
        "description": "One sentence role description based on domain",
        "background": "Background story with years of experience, achievements, etc.",
        "personality": "2-3 key personality traits"
    }},
    "thinking_framework": [
        "Thinking principle 1",
        "Thinking principle 2",
        "Thinking principle 3",
        "Thinking principle 4"
    ],
    "strengths": [
        "Core strength 1",
        "Core strength 2",
        "Core strength 3",
        "Core strength 4"
    ],
    "weaknesses": [
        "Potential weakness 1",
        "Potential weakness 2"
    ],
    "blind_spots": [
        "Possible oversight 1",
        "Possible oversight 2"
    ],
    "superhuman_insights": [
        "Unique insight 1",
        "Unique insight 2",
        "Unique insight 3"
    ],
    "capabilities": [
        {{"name": "capability_name_1", "params": {{"key": "value"}}}},
        {{"name": "capability_name_2", "params": {{"key": "value"}}}},
        {{"name": "capability_name_3", "params": {{"key": "value"}}}}
    ],
    "i_o_spec": {{
        "input": {{"type": "domain-specific input type", "format": "json"}},
        "output": {{"type": "domain-specific output type", "format": "json"}}
    }},
    "decision_style": {{
        "risk_tolerance": "very_low|low|medium|high",
        "consensus_need": "none|low|medium|high"
    }},
    "trigger_conditions": [
        "Trigger condition 1",
        "Trigger condition 2"
    ]
}}

Make sure:
1. The role name is in Chinese
2. All fields are relevant to the domain: {domain_description}
3. Capabilities match the requirements provided
4. Thinking framework reflects domain-specific expertise
5. Return valid JSON only"""

    def _parse_llm_response(
        self,
        response_content: str,
        domain_description: str,
        requirements: List[str]
    ) -> ExpertConfig:
        """
        Parse LLM response into ExpertConfig.

        Args:
            response_content: LLM response text
            domain_description: Original domain description
            requirements: Original requirements list

        Returns:
            ExpertConfig instance

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response_content)
            data = json.loads(json_str)

            # Generate unique ID
            expert_id = self._generate_expert_id(domain_description)

            # Parse persona
            persona_data = data.get("persona", {})
            persona = Persona(
                description=persona_data.get("description", f"{domain_description}领域的专家"),
                background=persona_data.get("background", f"在{domain_description}领域拥有多年专业经验"),
                personality=persona_data.get("personality", "严谨、专业、注重细节")
            )

            # Parse capabilities
            capabilities = []
            for cap_data in data.get("capabilities", []):
                capabilities.append(Capability(
                    name=cap_data.get("name", ""),
                    params=cap_data.get("params", {})
                ))

            # Parse I/O spec
            io_data = data.get("i_o_spec", {})
            i_o_spec = None
            if io_data:
                i_o_spec = IOSpec(
                    input=io_data.get("input", {"type": "text", "format": "json"}),
                    output=io_data.get("output", {"type": "text", "format": "json"})
                )

            # Parse decision style
            decision_data = data.get("decision_style", {})
            decision_style = None
            if decision_data:
                decision_style = DecisionStyle(
                    risk_tolerance=decision_data.get("risk_tolerance", "medium"),
                    consensus_need=decision_data.get("consensus_need", "medium")
                )

            # Create ExpertConfig
            config = ExpertConfig(
                id=expert_id,
                role=data.get("role", f"{domain_description}专家"),
                archetype=Archetype.DYNAMIC,
                load_strategy=LoadStrategy.DYNAMIC,
                persona=persona,
                thinking_framework=data.get("thinking_framework", []),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                blind_spots=data.get("blind_spots", []),
                superhuman_insights=data.get("superhuman_insights", []),
                capabilities=capabilities,
                i_o_spec=i_o_spec,
                decision_style=decision_style,
                memory=MemoryConfig(scope="task", retention="session"),
                trigger_conditions=data.get("trigger_conditions", [])
            )

            return config

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: create minimal valid config
            return self._create_fallback_config(domain_description, requirements, str(e))

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON string from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            JSON string
        """
        # Try to find JSON object in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx >= 0 and end_idx > start_idx:
            return response[start_idx:end_idx]

        # If no JSON found, return response as-is (may be pure JSON)
        return response.strip()

    def _generate_expert_id(self, domain_description: str) -> str:
        """
        Generate unique expert ID from domain description.

        Args:
            domain_description: Domain description

        Returns:
            Unique ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Create slug from domain
        slug = domain_description.lower()
        slug = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '_', slug)
        slug = slug[:30]  # Limit length

        return f"dynamic_expert_{slug}_{timestamp}"

    def _create_fallback_config(
        self,
        domain_description: str,
        requirements: List[str],
        error_reason: str
    ) -> ExpertConfig:
        """
        Create fallback expert config when LLM parsing fails.

        Args:
            domain_description: Domain description
            requirements: Requirements list
            error_reason: Error that caused fallback

        Returns:
            Minimal valid ExpertConfig
        """
        expert_id = self._generate_expert_id(domain_description)

        return ExpertConfig(
            id=expert_id,
            role=f"{domain_description}专家",
            archetype=Archetype.DYNAMIC,
            load_strategy=LoadStrategy.DYNAMIC,
            persona=Persona(
                description=f"专注于{domain_description}领域的专家",
                background=f"在{domain_description}领域拥有丰富经验，能够处理相关专业任务",
                personality="专业、严谨、注重细节"
            ),
            thinking_framework=[
                "分析任务需求和约束条件",
                "应用领域专业知识进行评估",
                "综合考虑多方面因素给出建议",
                "验证方案的可行性和合理性"
            ],
            strengths=[
                f"在{domain_description}领域有深厚积累",
                "能够快速理解复杂问题",
                "提供专业且可执行的建议"
            ],
            weaknesses=[
                "可能缺乏跨领域知识整合",
                "对非专业领域理解有限"
            ],
            blind_spots=[
                "可能过度依赖领域经验",
                "忽视创新解决方案"
            ],
            superhuman_insights=[
                f"能够识别{domain_description}领域的关键模式",
                "预判潜在问题和风险"
            ],
            capabilities=[
                Capability(name="domain_analysis", params={"domain": domain_description}),
                Capability(name="requirement_evaluation", params={})
            ],
            i_o_spec=IOSpec(
                input={"type": "task_requirements", "format": "json"},
                output={"type": "expert_analysis", "format": "json"}
            ),
            decision_style=DecisionStyle(
                risk_tolerance="medium",
                consensus_need="medium"
            ),
            memory=MemoryConfig(scope="task", retention="session"),
            trigger_conditions=[
                f"任务涉及{domain_description}领域",
                "现有专家无法覆盖该领域需求"
            ]
        )

    async def score_feasibility(
        self,
        expert_config: ExpertConfig,
        task_requirements: Dict[str, Any]
    ) -> float:
        """
        Score how feasible a new expert is for given task requirements.

        Evaluates the expert on:
        - Capability match with requirements
        - Role clarity and appropriateness
        - Expected contribution to task

        Args:
            expert_config: Expert configuration to evaluate
            task_requirements: Task requirements dictionary

        Returns:
            Feasibility score between 0.0 and 1.0
        """
        # Build evaluation prompt
        prompt = self._build_feasibility_prompt(expert_config, task_requirements)

        # Call LLM for scoring
        response = await self.model_router.route_request(
            TaskType.TEXT_REASONING,
            prompt,
            max_tokens=500
        )

        # Parse score from response
        score = self._parse_feasibility_score(response.content)

        return score

    def _build_feasibility_prompt(
        self,
        expert_config: ExpertConfig,
        task_requirements: Dict[str, Any]
    ) -> str:
        """
        Build prompt for feasibility scoring.

        Args:
            expert_config: Expert config to evaluate
            task_requirements: Task requirements

        Returns:
            Formatted prompt
        """
        requirements_list = task_requirements.get("requirements", [])
        gap_info = task_requirements.get("gap", {}).get("description", "N/A")

        return f"""Evaluate the feasibility of this expert for the task.

EXPERT PROFILE:
- Role: {expert_config.role}
- Archetype: {expert_config.archetype.value}
- Strengths: {', '.join(expert_config.strengths[:3])}
- Capabilities: {', '.join(c.name for c in expert_config.capabilities[:3])}
- Thinking Framework: {', '.join(expert_config.thinking_framework[:2])}

TASK REQUIREMENTS:
{chr(10).join(f'- {req}' for req in requirements_list)}

GAP TO FILL:
{gap_info}

Score the expert's feasibility on these criteria (each 0-10):
1. Capability Match: How well do capabilities match requirements?
2. Role Clarity: Is the role clearly defined and appropriate?
3. Expected Contribution: How much value will this expert add?

Calculate total score (sum of 3 criteria, max 30).
- Score >= 24: Excellent fit (return 0.9-1.0)
- Score 18-23: Good fit (return 0.7-0.9)
- Score 12-17: Partial fit (return 0.4-0.7)
- Score < 12: Poor fit (return 0.0-0.4)

Return ONLY a JSON object: {{"score": float, "rationale": "brief explanation"}}"""

    def _parse_feasibility_score(self, response: str) -> float:
        """
        Parse feasibility score from LLM response.

        Args:
            response: LLM response text

        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response)
            data = json.loads(json_str)
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try to extract any float from response
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            # Normalize if score seems to be on 0-10 scale
            if score > 1.0:
                score = score / 10.0
            return max(0.0, min(1.0, score))

        # Default to neutral score
        return 0.5

    def register_expert(
        self,
        expert_config: ExpertConfig,
        expert_class: Optional[Type[Expert]] = None
    ) -> bool:
        """
        Register a new expert to the registry.

        Args:
            expert_config: Expert configuration to register
            expert_class: Optional expert class to register

        Returns:
            True if registration successful
        """
        try:
            # Register configuration
            self.registry.register_config(expert_config)

            # Register class if provided
            if expert_class is not None:
                self.registry.register_class(expert_config.id, expert_class)

            return True
        except Exception as e:
            print(f"Failed to register expert {expert_config.id}: {e}")
            return False

    def persist_expert(
        self,
        expert_config: ExpertConfig,
        output_dir: str = "experts/dynamic"
    ) -> str:
        """
        Persist expert configuration to YAML file.

        Args:
            expert_config: Expert configuration to save
            output_dir: Directory to save expert file

        Returns:
            Path to saved file
        """
        from ..core.config_loader import save_expert_config

        # Create output path
        output_path = Path(output_dir) / f"{expert_config.id}.yml"

        # Save config
        save_expert_config(expert_config, str(output_path))

        return str(output_path)

    async def validate_expert(self, expert_config: ExpertConfig) -> bool:
        """
        Validate expert configuration structure.

        Checks:
        - All required fields are present
        - Values are within expected ranges
        - Capabilities are properly formatted

        Args:
            expert_config: Expert configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not expert_config.id:
                return False
            if not expert_config.role:
                return False
            if not expert_config.persona:
                return False

            # Check persona fields
            persona = expert_config.persona
            if not persona.description or not persona.background or not persona.personality:
                return False

            # Check capabilities format
            for cap in expert_config.capabilities:
                if not cap.name:
                    return False

            # Check decision style if present
            if expert_config.decision_style:
                valid_risk = ["very_low", "low", "medium", "high"]
                valid_consensus = ["none", "low", "medium", "high"]
                if expert_config.decision_style.risk_tolerance not in valid_risk:
                    return False
                if expert_config.decision_style.consensus_need not in valid_consensus:
                    return False

            # Check memory config if present
            if expert_config.memory:
                valid_scope = ["task", "session", "persistent"]
                valid_retention = ["task", "session", "persistent", "permanent"]
                if expert_config.memory.scope not in valid_scope:
                    return False
                if expert_config.memory.retention not in valid_retention:
                    return False

            return True

        except Exception:
            return False

    def get_feasibility_rating(self, score: float) -> str:
        """
        Get human-readable feasibility rating from score.

        Args:
            score: Feasibility score (0.0-1.0)

        Returns:
            Rating string
        """
        if score >= self.FEASIBILITY_EXCELLENT:
            return "EXCELLENT"
        elif score >= self.FEASIBILITY_GOOD:
            return "GOOD"
        elif score >= self.FEASIBILITY_MINIMUM:
            return "ACCEPTABLE"
        else:
            return "POOR"

    async def generate_and_register(
        self,
        domain_description: str,
        requirements: List[str],
        task_requirements: Dict[str, Any],
        minimum_feasibility: float = 0.6
    ) -> Optional[ExpertConfig]:
        """
        Complete workflow: generate expert, score feasibility, and register if合格.

        This is a convenience method that combines generate_expert, score_feasibility,
        and register_expert into a single workflow.

        Args:
            domain_description: Domain description
            requirements: List of requirements
            task_requirements: Task requirements for feasibility scoring
            minimum_feasibility: Minimum score to proceed with registration

        Returns:
            ExpertConfig if registration successful, None otherwise
        """
        # Generate expert
        config = await self.generate_expert(domain_description, requirements)

        # Score feasibility
        feasibility = await self.score_feasibility(config, task_requirements)

        # Check if feasible enough
        if feasibility < minimum_feasibility:
            print(
                f"Expert feasibility {feasibility:.2f} below threshold "
                f"{minimum_feasibility:.2f}, not registering"
            )
            return None

        # Validate config
        if not await self.validate_expert(config):
            print("Expert config validation failed")
            return None

        # Register expert
        if not self.register_expert(config):
            print("Failed to register expert")
            return None

        # Persist to disk
        try:
            self.persist_expert(config)
        except Exception as e:
            print(f"Warning: Failed to persist expert: {e}")

        return config
