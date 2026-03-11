"""HR Expert for capability gap analysis and dynamic expert recruitment."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from ..experts.expert import Expert
from ..core.models import ExpertConfig, Archetype, LoadStrategy, Persona, DecisionStyle, MemoryConfig, Capability
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType
from ..experts.expert_registry import ExpertRegistry

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from ..experts.dynamic_expert_generator import DynamicExpertGenerator


class HRExpert(Expert):
    """
    Senior HR Expert for capability gap analysis and dynamic expert recruitment.

    Responsibilities:
    - Identify capability gaps in current team
    - Create dynamic expert profiles using DynamicExpertGenerator
    - Score expert feasibility
    - Optimize team composition
    - Register new experts when gaps are identified

    This expert has on_demand load strategy - only loaded when needed.
    It provides recommendations but does not have veto power.
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter,
        expert_generator: Optional["DynamicExpertGenerator"] = None
    ):
        """
        Initialize HR Expert.

        Args:
            config: Expert configuration
            blackboard: Shared blackboard instance
            model_router: Model router for LLM calls
            expert_generator: Optional DynamicExpertGenerator instance.
                          If not provided, will be created lazily when needed.
        """
        super().__init__(config, blackboard, model_router)
        self._expert_generator = expert_generator
        self._registry = None

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process HR analysis task.

        Main entry point for HR expert operations:
        - Analyze task requirements from task_data
        - Check current team capabilities
        - Identify gaps
        - Recommend new experts if needed (using DynamicExpertGenerator)
        - Return recommendations with feasibility scores

        Args:
            task_data: Task data including requirements and current_team

        Returns:
            Recommendations with feasibility scores and expert suggestions
        """
        requirements = task_data.get("requirements", [])
        current_team = task_data.get("current_team", [])

        # Perform gap analysis
        gap_result = await self.gap_analysis(requirements, current_team)

        # If gaps exist, use DynamicExpertGenerator to create and score experts
        recommendations = []
        if gap_result.get("gaps"):
            generator = self._get_expert_generator()

            for gap in gap_result["gaps"]:
                # Use generator to create expert profile
                expert_profile = await generator.generate_expert(
                    domain_description=gap["description"],
                    requirements=requirements
                )

                # Score feasibility using generator
                feasibility = await generator.score_feasibility(
                    expert_profile,
                    {"requirements": requirements, "gap": gap}
                )

                recommendations.append({
                    "gap": gap,
                    "expert_profile": expert_profile,
                    "feasibility_score": feasibility,
                    "recommendation": "hire" if feasibility > 0.6 else "consider"
                })

        # Optimize team composition
        available_experts = task_data.get("available_experts", current_team)
        optimized_team = await self.team_optimization(
            available_experts,
            {"requirements": requirements}
        )

        return {
            "gap_analysis": gap_result,
            "recommendations": recommendations,
            "optimized_team": optimized_team,
            "total_new_experts_needed": len([r for r in recommendations if r["recommendation"] == "hire"])
        }

    async def gap_analysis(
        self,
        requirements: List[str],
        current_team: List[str]
    ) -> Dict[str, Any]:
        """
        Identify capability gaps between requirements and current team.

        Args:
            requirements: List of required capabilities/skills
            current_team: List of current expert IDs or capabilities

        Returns:
            Gap analysis result with identified gaps and coverage score
        """
        # Build prompt for LLM-powered gap analysis
        prompt = f"""Analyze the capability gaps between task requirements and current team.

Task Requirements:
{chr(10).join(f'- {req}' for req in requirements)}

Current Team Capabilities:
{chr(10).join(f'- {expert}' for expert in current_team)}

Identify specific capability gaps. For each gap, provide:
1. Gap name
2. Description of what's missing
3. Priority level (critical/high/medium/low)
4. Why this gap matters for the task

Return JSON with:
{{
    "coverage_score": float (0.0-1.0),
    "gaps": [
        {{"name": str, "description": str, "priority": str, "rationale": str}}
    ],
    "covered_requirements": [str],
    "missing_requirements": [str]
}}"""

        response = await self.call_llm(prompt, max_tokens=1500)

        # Parse response
        return self._parse_gap_analysis_response(response)

    async def create_expert_profile(self, domain_description: str) -> ExpertConfig:
        """
        Create a dynamic expert profile based on domain requirements.

        Args:
            domain_description: Description of the domain/field the expert should cover

        Returns:
            ExpertConfig for the new dynamic expert
        """
        prompt = f"""Create an expert profile for a specialist in the following domain:

{domain_description}

The expert should have:
- Clear role name in Chinese
- Relevant background and expertise
- Appropriate thinking framework
- Key strengths for this domain
- Potential weaknesses and blind spots
- Required capabilities with parameters

Return JSON with expert configuration:
{{
    "id": "dynamic_expert_{domain_slug}",
    "role": "role name in Chinese",
    "archetype": "dynamic",
    "load_strategy": "on_demand",
    "persona": {{
        "description": "...",
        "background": "...",
        "personality": "..."
    }},
    "thinking_framework": ["...", "..."],
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "blind_spots": ["...", "..."],
    "capabilities": [
        {{"name": "...", "params": {{}}}}
    ],
    "i_o_spec": {{
        "input": {{"type": "...", "format": "..."}},
        "output": {{"type": "...", "format": "..."}}
    }},
    "decision_style": {{
        "risk_tolerance": "medium",
        "consensus_need": "medium"
    }}
}}"""

        response = await self.call_llm(prompt, max_tokens=2000)

        # Parse and create ExpertConfig
        return self._parse_expert_profile_response(response, domain_description)

    async def feasibility_score(
        self,
        expert_config: ExpertConfig,
        task_requirements: Dict
    ) -> float:
        """
        Score how well an expert matches task requirements.

        Args:
            expert_config: Expert configuration to evaluate
            task_requirements: Task requirements dictionary

        Returns:
            Feasibility score (0.0-1.0)
        """
        # Build prompt for feasibility scoring
        prompt = f"""Evaluate how well this expert matches the task requirements.

Expert Profile:
- Role: {expert_config.role}
- Strengths: {', '.join(expert_config.strengths)}
- Capabilities: {[c.name for c in expert_config.capabilities]}

Task Requirements:
{chr(10).join(f'- {req}' for req in task_requirements.get('requirements', []))}

Additional Context:
{task_requirements.get('gap', {}).get('description', 'N/A')}

Score the expert's feasibility for this task on a scale of 0.0 to 1.0:
- 0.0-0.3: Not suitable
- 0.3-0.6: Partially suitable, may need support
- 0.6-0.8: Good fit
- 0.8-1.0: Excellent fit

Return only a JSON number: {{"score": float}}"""

        response = await self.call_llm(prompt, max_tokens=200)

        # Parse score from response
        return self._parse_feasibility_score_response(response)

    async def team_optimization(
        self,
        available_experts: List[str],
        task_requirements: Dict
    ) -> List[str]:
        """
        Optimize team composition for given task requirements.

        Args:
            available_experts: List of available expert IDs
            task_requirements: Task requirements dictionary

        Returns:
            Optimized list of expert IDs
        """
        requirements = task_requirements.get("requirements", [])

        prompt = f"""Select the optimal team composition from available experts.

Available Experts:
{chr(10).join(f'- {expert}' for expert in available_experts)}

Task Requirements:
{chr(10).join(f'- {req}' for req in requirements)}

Consider:
1. Coverage of all required capabilities
2. Team size efficiency (prefer smaller teams)
3. Complementary skills
4. Avoid redundancy

Return JSON array of selected expert IDs in optimal order:
["expert_id_1", "expert_id_2", ...]"""

        response = await self.call_llm(prompt, max_tokens=500)

        # Parse team list from response
        return self._parse_team_optimization_response(response)

    def _parse_gap_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM gap analysis response.

        Args:
            response: LLM response text

        Returns:
            Parsed gap analysis result
        """
        import json
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)

                # Validate structure
                return {
                    "coverage_score": result.get("coverage_score", 0.0),
                    "gaps": result.get("gaps", []),
                    "covered_requirements": result.get("covered_requirements", []),
                    "missing_requirements": result.get("missing_requirements", [])
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: simple analysis
        return {
            "coverage_score": 0.5,
            "gaps": [{"name": "unspecified", "description": "Could not parse gaps", "priority": "medium"}],
            "covered_requirements": [],
            "missing_requirements": []
        }

    def _parse_expert_profile_response(
        self,
        response: str,
        domain_description: str
    ) -> ExpertConfig:
        """
        Parse LLM expert profile response.

        Args:
            response: LLM response text
            domain_description: Original domain description

        Returns:
            ExpertConfig instance
        """
        import json
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)

                # Generate unique ID from domain
                domain_slug = domain_description[:20].lower().replace(" ", "_").replace("-", "_")

                return ExpertConfig(
                    id=data.get("id", f"dynamic_expert_{domain_slug}"),
                    role=data.get("role", "Domain Expert"),
                    archetype=Archetype.DYNAMIC,
                    load_strategy=LoadStrategy.ON_DEMAND,
                    persona=Persona(
                        description=data.get("persona", {}).get("description", "Domain expert"),
                        background=data.get("persona", {}).get("background", "Specialist background"),
                        personality=data.get("persona", {}).get("personality", "Professional and analytical")
                    ),
                    thinking_framework=data.get("thinking_framework", []),
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    blind_spots=data.get("blind_spots", []),
                    capabilities=[
                        Capability(name=c.get("name", ""), params=c.get("params", {}))
                        for c in data.get("capabilities", [])
                    ],
                    i_o_spec=None,
                    decision_style=None,
                    memory=MemoryConfig(scope="session", retention="persistent")
                )
        except (json.JSONDecodeError, ValueError, Exception):
            pass

        # Fallback: create minimal expert config
        domain_slug = domain_description[:20].lower().replace(" ", "_").replace("-", "_")
        return ExpertConfig(
            id=f"dynamic_expert_{domain_slug}",
            role="Domain Expert",
            archetype=Archetype.DYNAMIC,
            load_strategy=LoadStrategy.ON_DEMAND,
            persona=Persona(
                description="Domain expert",
                background="Specialist in the required field",
                personality="Professional and analytical"
            ),
            thinking_framework=["Analyze requirements", "Apply domain expertise"],
            strengths=[f"Expert in {domain_description}"],
            weaknesses=["May lack cross-domain knowledge"],
            blind_spots=["Potential tunnel vision"],
            capabilities=[],
            memory=MemoryConfig(scope="session", retention="persistent")
        )

    def _parse_feasibility_score_response(self, response: str) -> float:
        """
        Parse feasibility score from LLM response.

        Args:
            response: LLM response text

        Returns:
            Score between 0.0 and 1.0
        """
        import re
        try:
            # Try to extract JSON score
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                import json
                data = json.loads(json_str)
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))

            # Try to extract any float from response
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Default to neutral score
        return 0.5

    def _parse_team_optimization_response(self, response: str) -> List[str]:
        """
        Parse optimized team list from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of expert IDs
        """
        import json
        try:
            # Try to extract JSON array
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                if isinstance(result, list):
                    return [str(x) for x in result]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract quoted strings
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', response)
        if matches:
            return matches

        # Return empty if nothing found
        return []

    def _get_task_type(self) -> TaskType:
        """
        Get task type for model routing.

        HR expert handles text reasoning tasks for analysis and planning.

        Returns:
            TaskType.TEXT_REASONING
        """
        return TaskType.TEXT_REASONING

    def has_veto_power(self) -> bool:
        """
        Check if this expert has veto power.

        HR expert only provides recommendations, does not have veto power.

        Returns:
            False
        """
        veto_config = self.config.veto_power
        return veto_config.get("enabled", False) if veto_config else False

    def _get_expert_generator(self) -> "DynamicExpertGenerator":
        """
        Get or create DynamicExpertGenerator instance.

        Lazy initialization to avoid circular imports.

        Returns:
            DynamicExpertGenerator instance
        """
        if self._expert_generator is None:
            from ..experts.expert_registry import ExpertRegistry
            from ..experts.dynamic_expert_generator import DynamicExpertGenerator

            # Create registry if not available
            if self._registry is None:
                self._registry = ExpertRegistry()

            self._expert_generator = DynamicExpertGenerator(
                model_router=self.model_router,
                registry=self._registry
            )

        return self._expert_generator

    async def hire_expert(
        self,
        domain_description: str,
        requirements: List[str],
        task_requirements: Dict[str, Any],
        minimum_feasibility: float = 0.6
    ) -> Optional[ExpertConfig]:
        """
        Hire a new dynamic expert using the generator.

        This is a convenience method that:
        1. Generates expert config from domain description
        2. Scores feasibility
        3. Registers expert if feasibility passes threshold
        4. Persists expert to disk

        Args:
            domain_description: Domain/field description
            requirements: List of required capabilities
            task_requirements: Task requirements for scoring
            minimum_feasibility: Minimum score to proceed with hiring

        Returns:
            ExpertConfig if hired successfully, None otherwise
        """
        generator = self._get_expert_generator()

        config = await generator.generate_and_register(
            domain_description=domain_description,
            requirements=requirements,
            task_requirements=task_requirements,
            minimum_feasibility=minimum_feasibility
        )

        return config

    def set_registry(self, registry: ExpertRegistry):
        """
        Set the expert registry for this HR expert.

        Args:
            registry: ExpertRegistry instance
        """
        self._registry = registry
        if self._expert_generator is not None:
            self._expert_generator.registry = registry
