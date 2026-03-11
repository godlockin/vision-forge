"""Image Editor expert for complex editing decomposition and mask generation."""

import json
from typing import Dict, Any, List

from .expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class ImageEditorExpert(Expert):
    """
    Professional image editor expert for complex editing tasks.

    Specializes in:
    - Complex editing task decomposition into step-by-step workflows
    - Mask generation for targeted local adjustments
    - Artifact removal and seamless inpainting
    - Workflow optimization for efficiency
    - Professional retouching techniques

    This expert breaks down complex editing requests into executable steps
    and generates precise mask descriptions for targeted modifications.
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize image editor expert.

        Args:
            config: Expert configuration from YAML
            blackboard: Shared blackboard instance
            model_router: Model router for LLM access
        """
        super().__init__(config, blackboard, model_router)

    def _get_task_type(self) -> TaskType:
        """Get appropriate task type for model routing."""
        return TaskType.TEXT_REASONING

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image editing task.

        Args:
            task_data: Task data containing:
                - user_request: The editing request from user
                - image_context: Optional context about the source image
                - constraints: Optional editing constraints or requirements

        Returns:
            Editing plan with decomposed steps, mask descriptions, and workflow
        """
        user_request = task_data.get("user_request", "")
        image_context = task_data.get("image_context", "")
        constraints = task_data.get("constraints", [])

        result: Dict[str, Any] = {
            "expert_id": self.id,
            "expert_role": self.role,
            "confidence": self._confidence,
        }

        # Decompose the editing task into steps
        edit_steps = await self.decompose_complex_edit(user_request, image_context)
        result["edit_steps"] = edit_steps

        # Generate mask descriptions if needed
        if self._requires_masking(user_request):
            result["mask_descriptions"] = await self._generate_required_masks(
                user_request, edit_steps
            )

        # Suggest the optimal editing approach
        result["editing_approach"] = await self.suggest_editing_approach(user_request)

        # Add workflow optimization suggestions
        result["workflow_tips"] = await self._optimize_workflow(edit_steps)

        self.record_action("image_editing_plan", result)
        return result

    async def decompose_complex_edit(
        self,
        user_request: str,
        image_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Break down complex editing request into executable steps.

        Args:
            user_request: User's editing request
            image_context: Optional context about the source image

        Returns:
            List of editing steps, each containing:
                - step_number: Sequential step number
                - action: The action to perform
                - tool_category: Category of tool/technique needed
                - parameters: Specific parameters for the action
                - dependencies: Previous steps this depends on
                - estimated_difficulty: Difficulty level (easy/medium/hard)
                - mask_required: Whether this step needs a mask
        """
        context_info = f"\nSource Image Context:\n{image_context}" if image_context else ""

        prompt = f"""Decompose this image editing request into a step-by-step workflow.

User Request:
{user_request}
{context_info}

Break this down into discrete, executable editing steps. For each step, identify:
1. The specific action to perform
2. The tool/technique category (e.g., inpainting, color adjustment, object removal, etc.)
3. Any parameters needed
4. Which previous steps it depends on (if any)
5. Difficulty level (easy/medium/hard)
6. Whether a mask is required for precise execution

Respond with a JSON array of step objects. Each object should have keys:
step_number, action, tool_category, parameters, dependencies, estimated_difficulty, mask_required"""

        response = await self.call_llm(prompt)

        try:
            steps = json.loads(response)
            if isinstance(steps, list):
                return self._validate_editing_steps(steps)
            return self._create_fallback_steps(user_request)
        except (json.JSONDecodeError, ValueError):
            return self._create_fallback_steps(user_request)

    async def generate_mask_description(self, edit_area: str) -> str:
        """
        Generate a precise mask description for targeted editing.

        Args:
            edit_area: Description of the area to be edited

        Returns:
            Detailed mask description that can be used for:
                - Manual mask creation guidance
                - AI-based segmentation prompts
                - Mask refinement instructions
        """
        prompt = f"""Generate a precise mask description for this editing area.

Area to Edit:
{edit_area}

Provide a detailed mask description including:
1. Primary subject/region boundaries
2. Edge handling requirements (soft/hard edges, feathering)
3. Areas to explicitly exclude from the mask
4. Any complex boundary considerations (hair, transparent objects, etc.)
5. Suggested mask refinement techniques

The description should be precise enough for either:
- A human editor to create the mask accurately
- An AI segmentation model to generate the mask from text

Respond with a structured mask description."""

        response = await self.call_llm(prompt)
        return response

    async def suggest_editing_approach(self, target_effect: str) -> Dict[str, Any]:
        """
        Recommend the best editing technique for achieving a desired effect.

        Args:
            target_effect: The visual effect the user wants to achieve

        Returns:
            Dictionary containing:
                - recommended_technique: Primary recommended technique
                - alternative_techniques: Other viable approaches
                - pros_and_cons: Trade-offs of the recommended approach
                - required_tools: Tools/capabilities needed
                - difficulty_rating: Overall difficulty (1-10)
                - estimated_time: Rough time estimate
                - quality_expectation: Expected quality outcome
        """
        prompt = f"""Recommend the best editing approach for achieving this effect.

Target Effect:
{target_effect}

Provide a comprehensive recommendation:
1. Primary Technique: What's the best approach and why?
2. Alternative Techniques: What other methods could work?
3. Pros and Cons: Trade-offs of the recommended approach
4. Required Tools: What tools/capabilities are needed?
5. Difficulty: Rate 1-10 (1=trivial, 10=expert level)
6. Time Estimate: Rough time required
7. Quality Expectation: What quality level can be expected?

Respond in JSON format with keys: recommended_technique, alternative_techniques, pros_and_cons, required_tools, difficulty_rating, estimated_time, quality_expectation"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "recommended_technique": response,
                "alternative_techniques": [],
                "pros_and_cons": "Analysis could not be parsed",
                "required_tools": ["Unknown"],
                "difficulty_rating": 5,
                "estimated_time": "Unknown",
                "quality_expectation": "Unknown"
            }

    def _requires_masking(self, user_request: str) -> bool:
        """Check if the editing request requires masking."""
        mask_keywords = [
            "remove", "replace", "change", "adjust", "modify",
            "local", "specific", "only", "except", "background",
            "foreground", "selective", "targeted"
        ]
        return any(keyword in user_request.lower() for keyword in mask_keywords)

    async def _generate_required_masks(
        self,
        user_request: str,
        edit_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate mask descriptions for steps that require masking."""
        mask_descriptions = []

        for step in edit_steps:
            if step.get("mask_required", False):
                area_description = step.get("parameters", {}).get("target_area", "unspecified")
                mask_desc = await self.generate_mask_description(area_description)
                mask_descriptions.append({
                    "step_number": step["step_number"],
                    "action": step["action"],
                    "mask_description": mask_desc
                })

        return mask_descriptions

    def _validate_editing_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize editing steps."""
        validated = []
        required_keys = ["step_number", "action", "tool_category"]

        for i, step in enumerate(steps):
            # Ensure required keys exist
            for key in required_keys:
                if key not in step:
                    step[key] = "unspecified"

            # Ensure step_number is sequential
            step["step_number"] = i + 1

            # Default values for optional keys
            step.setdefault("parameters", {})
            step.setdefault("dependencies", [])
            step.setdefault("estimated_difficulty", "medium")
            step.setdefault("mask_required", False)

            validated.append(step)

        return validated

    def _create_fallback_steps(self, user_request: str) -> List[Dict[str, Any]]:
        """Create fallback editing steps when parsing fails."""
        return [
            {
                "step_number": 1,
                "action": f"Analyze request: {user_request}",
                "tool_category": "analysis",
                "parameters": {"request": user_request},
                "dependencies": [],
                "estimated_difficulty": "medium",
                "mask_required": False
            },
            {
                "step_number": 2,
                "action": "Execute primary editing operation",
                "tool_category": "editing",
                "parameters": {"operation": "primary"},
                "dependencies": [1],
                "estimated_difficulty": "medium",
                "mask_required": True
            },
            {
                "step_number": 3,
                "action": "Review and refine results",
                "tool_category": "review",
                "parameters": {"check_quality": True},
                "dependencies": [2],
                "estimated_difficulty": "easy",
                "mask_required": False
            }
        ]

    async def _optimize_workflow(
        self,
        edit_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Provide workflow optimization suggestions."""
        steps_description = "\n".join(
            f"{s['step_number']}. {s['action']} ({s['tool_category']})"
            for s in edit_steps
        )

        prompt = f"""Optimize this editing workflow for efficiency.

Current Workflow:
{steps_description}

Provide optimization suggestions:
1. Can any steps be parallelized or combined?
2. Are there any redundant steps?
3. What's the optimal order of operations?
4. Any automation opportunities?
5. Tips for maintaining quality while improving efficiency

Respond with specific, actionable optimization suggestions."""

        response = await self.call_llm(prompt)

        return {
            "original_step_count": len(edit_steps),
            "optimization_suggestions": response
        }

    async def suggest_batch_processing(self, similar_tasks: List[str]) -> Dict[str, Any]:
        """
        Suggest batch processing strategy for similar editing tasks.

        Args:
            similar_tasks: List of similar editing tasks to potentially batch

        Returns:
            Batch processing strategy with grouping and automation suggestions
        """
        tasks_text = "\n".join(f"- {task}" for task in similar_tasks)

        prompt = f"""Analyze these editing tasks for batch processing opportunities.

Tasks:
{tasks_text}

Identify:
1. Common operations that can be batched
2. Tasks that should remain separate due to complexity
3. Optimal batch groupings
4. Automation potential for each batch
5. Quality trade-offs of batch vs individual processing

Respond with a structured batch processing strategy."""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "strategy": response,
                "parsing_note": "Could not parse as JSON"
            }
