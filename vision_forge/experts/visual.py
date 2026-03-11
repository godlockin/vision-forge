"""Visual Expert implementation for image analysis, editing, and quality assessment."""

from typing import Dict, Any, List, Optional
from ..experts.expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType
from ..services.base import ImageGenerationResponse


class VisualExpert(Expert):
    """
    Visual Expert for image analysis, editing, and quality assessment.

    This expert has split behavior - can operate as both executor and critic.
    It is a 常驻 (always loaded) expert with load_strategy: "always".

    Responsibilities:
    - Aesthetic scoring (composition, color, lighting, mood)
    - Defect detection (artifacts, symmetry issues, logic errors)
    - Task decomposition for image editing
    - Actual image editing execution via Vertex AI
    """

    # Aesthetic scoring dimensions
    DEFAULT_SCORING_DIMENSIONS = ["composition", "color", "lighting", "mood"]

    # Defect detection categories
    DEFECT_CATEGORIES = ["artifacts", "symmetry_issues", "logic_errors", "other_defects"]

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize Visual Expert.

        Args:
            config: Expert configuration from YAML
            blackboard: Shared blackboard instance
            model_router: Model router for LLM calls
        """
        super().__init__(config, blackboard, model_router)

        # Check if this expert has split_behavior enabled
        self._split_behavior = config.split_behavior
        self._is_critic_mode = False

        # Extract capabilities from config
        self._capabilities = {cap.name: cap.params for cap in config.capabilities}

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing visual tasks.

        Determines if task is analysis or editing and routes accordingly.

        Args:
            task_data: Task-specific data including:
                - task_type: "analysis" or "editing"
                - image_description: Description of the image
                - user_request: User's request or prompt
                - reference_images: Optional list of image bytes

        Returns:
            Comprehensive results with scores and/or edited images
        """
        task_type = task_data.get("task_type", "analysis")
        image_description = task_data.get("image_description", "")
        user_request = task_data.get("user_request", "")
        reference_images = task_data.get("reference_images", [])

        # Set confidence based on task type and capabilities
        await self._assess_confidence(task_type, task_data)

        if task_type == "analysis":
            return await self._process_analysis(
                image_description=image_description,
                criteria=task_data.get("scoring_criteria"),
                reference_images=reference_images
            )

        elif task_type == "editing":
            return await self._process_editing(
                user_request=user_request,
                image_context=task_data.get("image_context", image_description),
                reference_images=reference_images
            )

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _process_analysis(
        self,
        image_description: str,
        criteria: Optional[List[str]] = None,
        reference_images: Optional[List[bytes]] = None
    ) -> Dict[str, Any]:
        """
        Process image analysis task.

        Args:
            image_description: Description of the image to analyze
            criteria: Optional custom scoring criteria
            reference_images: Optional reference images

        Returns:
            Analysis results with scores and defects
        """
        # Perform aesthetic scoring
        scores = await self.aesthetic_scoring(image_description, criteria)

        # Perform defect detection
        defects = await self.defect_detection(image_description)

        # Calculate overall quality score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0

        # Reduce score based on defects
        defect_penalty = self._calculate_defect_penalty(defects)
        adjusted_score = max(0.0, overall_score - defect_penalty)

        result = {
            "task_type": "analysis",
            "aesthetic_scores": scores,
            "overall_score": overall_score,
            "adjusted_score": adjusted_score,
            "defects": defects,
            "defect_penalty": defect_penalty,
            "expert_id": self.id,
            "confidence": self._confidence,
            "mode": "critic" if self._is_critic_mode else "executor"
        }

        return result

    async def _process_editing(
        self,
        user_request: str,
        image_context: str,
        reference_images: Optional[List[bytes]] = None
    ) -> Dict[str, Any]:
        """
        Process image editing task.

        Args:
            user_request: User's editing request
            image_context: Context about the image
            reference_images: Optional reference images

        Returns:
            Editing results with generated images and operation log
        """
        # Decompose the editing task
        subtasks = await self.decompose_edit_task(user_request, image_context)

        # Execute the image edit
        edit_prompt = self._build_edit_prompt(user_request, image_context)
        edit_result = await self.execute_image_edit(edit_prompt, reference_images)

        # Perform quality check on result
        quality_check = await self._quality_check(edit_result)

        result = {
            "task_type": "editing",
            "subtasks": subtasks,
            "edit_result": {
                "image_url": edit_result.image_url if edit_result else None,
                "image_data": edit_result.image_data if edit_result else None,
                "model": edit_result.model if edit_result else None,
                "latency_ms": edit_result.latency_ms if edit_result else None
            },
            "quality_check": quality_check,
            "expert_id": self.id,
            "confidence": self._confidence,
            "mode": "critic" if self._is_critic_mode else "executor"
        }

        return result

    async def aesthetic_scoring(
        self,
        image_description: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform aesthetic scoring on an image.

        Args:
            image_description: Description of the image to score
            criteria: Optional list of scoring criteria (defaults to composition, color, lighting, mood)

        Returns:
            Dictionary of dimension names to scores (0.0-1.0)
        """
        scoring_dimensions = criteria or self.DEFAULT_SCORING_DIMENSIONS

        # Build prompt for aesthetic scoring
        criteria_text = ", ".join(scoring_dimensions)
        mode_modifier = "As a harsh critic, " if self._is_critic_mode else ""

        prompt = f"""{mode_modifier}Analyze the following image description and provide aesthetic scores.

Image Description:
{image_description}

Score each of the following dimensions from 0.0 (poor) to 1.0 (excellent):
{criteria_text}

Provide your response as a JSON object with dimension names as keys and scores as values.
Example: {{"composition": 0.85, "color": 0.75, "lighting": 0.9, "mood": 0.8}}"""

        # Call LLM through model router
        response_text = await self.call_llm(prompt, max_tokens=500)

        # Parse scores from response
        scores = self._parse_scores_from_response(response_text, scoring_dimensions)

        # Validate scores are in valid range
        for dimension in scoring_dimensions:
            if dimension not in scores:
                # Default score if LLM didn't provide one
                scores[dimension] = 0.5
            else:
                scores[dimension] = max(0.0, min(1.0, scores[dimension]))

        return scores

    async def defect_detection(
        self,
        image_description: str
    ) -> Dict[str, Any]:
        """
        Detect defects in an image.

        Args:
            image_description: Description of the image to analyze

        Returns:
            Dictionary containing:
                - artifacts: List of detected artifacts
                - symmetry_issues: List of symmetry issues
                - logic_errors: List of logic errors
                - other_defects: List of other defects
                - severity: Overall severity (low/medium/high)
        """
        mode_modifier = "Be extremely critical and thorough. " if self._is_critic_mode else ""

        prompt = f"""{mode_modifier}Analyze the following image description for defects and issues.

Image Description:
{image_description}

Check for the following categories of defects:
1. Artifacts: AI generation artifacts, blurring, strange textures, unnatural edges
2. Symmetry Issues: Asymmetrical faces, mismatched objects, uneven lighting
3. Logic Errors: Impossible physics, wrong shadows, inconsistent perspectives
4. Other Defects: Any other visual issues not covered above

Provide your response as a JSON object with the following structure:
{{
    "artifacts": ["list of detected artifacts or empty"],
    "symmetry_issues": ["list of symmetry issues or empty"],
    "logic_errors": ["list of logic errors or empty"],
    "other_defects": ["list of other defects or empty"],
    "severity": "low|medium|high"
}}"""

        response_text = await self.call_llm(prompt, max_tokens=1000)

        # Parse defects from response
        defects = self._parse_defects_from_response(response_text)

        return defects

    async def decompose_edit_task(
        self,
        user_request: str,
        image_context: str
    ) -> List[Dict[str, Any]]:
        """
        Decompose a complex image editing task into subtasks.

        Args:
            user_request: User's editing request
            image_context: Context about the source image

        Returns:
            List of subtasks, each containing:
                - name: Subtask name
                - description: What to do
                - dependencies: List of subtask IDs this depends on
                - estimated_complexity: low/medium/high
        """
        prompt = f"""Decompose the following image editing request into clear, actionable subtasks.

User Request:
{user_request}

Image Context:
{image_context}

Break this down into a sequence of editing steps. Consider:
- What needs to be done first (foundational changes)
- What can be done in parallel
- What depends on previous steps
- The complexity of each step

Provide your response as a JSON array of subtasks:
[
    {{
        "id": "step_1",
        "name": "Brief step name",
        "description": "Detailed description of what to do",
        "dependencies": [],
        "estimated_complexity": "low|medium|high"
    }},
    ...
]"""

        response_text = await self.call_llm(prompt, max_tokens=1500)

        # Parse subtasks from response
        subtasks = self._parse_subtasks_from_response(response_text)

        return subtasks

    async def execute_image_edit(
        self,
        prompt: str,
        reference_images: Optional[List[bytes]] = None
    ) -> Dict[str, Any]:
        """
        Execute image editing via Vertex AI Imagen-3.

        Args:
            prompt: Editing prompt describing the desired changes
            reference_images: Optional list of reference image bytes

        Returns:
            ImageGenerationResponse with edited image data
        """
        # Enhance prompt with expert guidance
        enhanced_prompt = self._enhance_edit_prompt(prompt)

        # Use model router to route to image generation
        try:
            response = await self.model_router.route_request(
                TaskType.IMAGE_GENERATION,
                enhanced_prompt
            )

            # Record action for traceability
            self.record_action("image_edit", {
                "prompt": enhanced_prompt,
                "model": response.model if hasattr(response, 'model') else "unknown",
                "latency_ms": response.latency_ms if hasattr(response, 'latency_ms') else 0
            })

            return response

        except Exception as e:
            # Handle error gracefully
            error_result = {
                "error": str(e),
                "prompt_used": enhanced_prompt,
                "expert_id": self.id
            }

            # Return empty response structure on error
            return ImageGenerationResponse(
                image_url=None,
                image_data=None,
                prompt_used=enhanced_prompt,
                model="imagen-3.0-generate-001",
                latency_ms=0,
                parameters={"error": str(e)}
            )

    def _get_task_type(self) -> TaskType:
        """
        Get the appropriate task type for model routing.

        Returns:
            TaskType.IMAGE_ANALYSIS for analysis tasks
            TaskType.IMAGE_GENERATION for generation tasks
        """
        # Default to visual analysis for the Visual Expert
        return TaskType.VISUAL_ANALYSIS

    async def _assess_confidence(
        self,
        task_type: str,
        task_data: Dict[str, Any]
    ):
        """
        Self-assess confidence for the given task.

        Args:
            task_type: Type of task (analysis/editing)
            task_data: Task-specific data
        """
        # Base confidence from config capabilities
        base_confidence = 0.8  # Default high confidence

        # Adjust based on task complexity
        if task_type == "editing":
            # Check if editing capability is available
            if "image_editing" not in self._capabilities:
                base_confidence -= 0.2

            # Adjust based on number of subtasks
            num_subtasks = len(task_data.get("subtasks", []))
            if num_subtasks > 5:
                base_confidence -= 0.1

        # Adjust based on critic mode
        if self._is_critic_mode:
            base_confidence += 0.05  # Slightly higher confidence as critic

        self.set_confidence(base_confidence)

    def _calculate_defect_penalty(self, defects: Dict[str, Any]) -> float:
        """
        Calculate penalty score based on detected defects.

        Args:
            defects: Dictionary of detected defects

        Returns:
            Penalty value to subtract from overall score (0.0-0.5)
        """
        penalty = 0.0

        # Count total defects
        total_defects = sum(
            len(defects.get(category, []))
            for category in self.DEFECT_CATEGORIES
        )

        # Base penalty per defect
        penalty += total_defects * 0.05

        # Severity multiplier
        severity = defects.get("severity", "low")
        if severity == "medium":
            penalty *= 1.5
        elif severity == "high":
            penalty *= 2.0

        # Cap penalty at 0.5
        return min(0.5, penalty)

    def _build_edit_prompt(
        self,
        user_request: str,
        image_context: str
    ) -> str:
        """
        Build comprehensive edit prompt from user request.

        Args:
            user_request: User's editing request
            image_context: Context about the source image

        Returns:
            Enhanced prompt for image generation
        """
        return f"""Transform the image based on the following request.

Source Image Context:
{image_context}

Editing Request:
{user_request}

Requirements:
- Maintain high quality and consistency
- Preserve key elements from the source unless explicitly requested to change
- Ensure natural-looking results without artifacts
- Follow professional photography and design principles"""

    def _enhance_edit_prompt(self, prompt: str) -> str:
        """
        Enhance edit prompt with expert guidance.

        Args:
            prompt: Original prompt

        Returns:
            Enhanced prompt with additional detail
        """
        return f"""{prompt}

Professional quality requirements:
- High resolution, no artifacts
- Natural lighting and shadows
- Consistent perspective and proportions
- Professional color grading
- Clean edges and smooth transitions"""

    async def _quality_check(
        self,
        edit_result: Any
    ) -> Dict[str, Any]:
        """
        Perform quality check on edited image result.

        Args:
            edit_result: ImageGenerationResponse from execute_image_edit

        Returns:
            Quality check results
        """
        if not edit_result or not edit_result.image_data:
            return {
                "passed": False,
                "issues": ["No image data generated"],
                "severity": "high"
            }

        # Build description from prompt for analysis
        prompt_used = edit_result.prompt_used if hasattr(edit_result, 'prompt_used') else ""

        # Quick defect check
        defects = await self.defect_detection(prompt_used)

        # Determine if passed
        severity = defects.get("severity", "low")
        passed = severity != "high"

        return {
            "passed": passed,
            "severity": severity,
            "defects_found": defects,
            "model": edit_result.model if hasattr(edit_result, 'model') else None,
            "latency_ms": edit_result.latency_ms if hasattr(edit_result, 'latency_ms') else None
        }

    def _parse_scores_from_response(
        self,
        response_text: str,
        dimensions: List[str]
    ) -> Dict[str, float]:
        """
        Parse aesthetic scores from LLM response.

        Args:
            response_text: LLM response text
            dimensions: Expected dimension names

        Returns:
            Parsed scores dictionary
        """
        import json

        scores = {}

        # Try to extract JSON from response
        try:
            # Look for JSON object in response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)

                for dim in dimensions:
                    if dim in parsed:
                        try:
                            scores[dim] = float(parsed[dim])
                        except (ValueError, TypeError):
                            scores[dim] = 0.5
        except (json.JSONDecodeError, KeyError):
            # Fallback: try to extract scores using regex
            import re
            for dim in dimensions:
                pattern = rf'"{dim}"\s*:\s*([\d.]+)'
                match = re.search(pattern, response_text)
                if match:
                    try:
                        scores[dim] = float(match.group(1))
                    except ValueError:
                        scores[dim] = 0.5

        return scores

    def _parse_defects_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse defects from LLM response.

        Args:
            response_text: LLM response text

        Returns:
            Parsed defects dictionary
        """
        import json

        default_defects = {
            "artifacts": [],
            "symmetry_issues": [],
            "logic_errors": [],
            "other_defects": [],
            "severity": "low"
        }

        # Try to extract JSON from response
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)

                for key in default_defects.keys():
                    if key in parsed:
                        value = parsed[key]
                        if isinstance(value, list):
                            default_defects[key] = value
                        elif isinstance(value, str) and key == "severity":
                            if value in ["low", "medium", "high"]:
                                default_defects["severity"] = value
        except (json.JSONDecodeError, KeyError):
            pass

        return default_defects

    def _parse_subtasks_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse subtasks from LLM response.

        Args:
            response_text: LLM response text

        Returns:
            Parsed subtasks list
        """
        import json

        # Try to extract JSON array from response
        try:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)

                if isinstance(parsed, list):
                    subtasks = []
                    for item in parsed:
                        if isinstance(item, dict):
                            subtasks.append({
                                "id": item.get("id", f"step_{len(subtasks)+1}"),
                                "name": item.get("name", "Unknown step"),
                                "description": item.get("description", ""),
                                "dependencies": item.get("dependencies", []),
                                "estimated_complexity": item.get("estimated_complexity", "medium")
                            })
                    return subtasks
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: return single task
        return [{
            "id": "step_1",
            "name": "Execute edit",
            "description": response_text[:500],
            "dependencies": [],
            "estimated_complexity": "medium"
        }]

    def set_critic_mode(self, enabled: bool = True):
        """
        Set the expert to critic mode.

        Args:
            enabled: Whether to enable critic mode
        """
        self._is_critic_mode = enabled
        if enabled:
            self._is_executor = False
        else:
            self._is_executor = True

    def is_critic_mode(self) -> bool:
        """Check if expert is in critic mode."""
        return self._is_critic_mode

    def __repr__(self) -> str:
        """String representation."""
        mode = "critic" if self._is_critic_mode else "executor"
        confidence = f"{self._confidence:.2f}" if self._confidence else "N/A"
        return f"<VisualExpert {self.id}: {self.role} ({mode}, confidence: {confidence})>"
