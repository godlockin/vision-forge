"""Photographer expert for lighting, composition, and realism analysis."""

import json
from typing import Dict, Any, List

from .expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class PhotographerExpert(Expert):
    """
    Professional photographer expert for image analysis.

    Specializes in:
    - Lighting quality assessment (hard/soft light, direction, color temperature)
    - Composition analysis (rule of thirds, leading lines, balance)
    - Depth of field and focus analysis
    - Color grading and tone mapping suggestions
    - Photorealism verification and AI artifact detection

    This expert analyzes images from a professional photography perspective,
    identifying lighting inconsistencies, composition issues, and unrealistic elements.
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize photographer expert.

        Args:
            config: Expert configuration from YAML
            blackboard: Shared blackboard instance
            model_router: Model router for LLM access
        """
        super().__init__(config, blackboard, model_router)

    def _get_task_type(self) -> TaskType:
        """Get appropriate task type for model routing."""
        return TaskType.VISUAL_ANALYSIS

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process photography analysis task.

        Args:
            task_data: Task data containing:
                - image_description: Description of the image to analyze
                - analysis_type: Type of analysis requested
                - specific_concerns: Optional list of specific concerns

        Returns:
            Analysis result with lighting, composition, and realism assessments
        """
        image_description = task_data.get("image_description", "")
        analysis_type = task_data.get("analysis_type", "full")
        specific_concerns = task_data.get("specific_concerns", [])

        result: Dict[str, Any] = {
            "expert_id": self.id,
            "expert_role": self.role,
            "analysis_type": analysis_type,
            "confidence": self._confidence,
        }

        if analysis_type == "lighting" or analysis_type == "full":
            result["lighting"] = await self.analyze_lighting(image_description)

        if analysis_type == "composition" or analysis_type == "full":
            result["composition"] = await self.analyze_composition(image_description)

        if analysis_type == "realism" or analysis_type == "full":
            result["realism"] = await self.assess_realism(image_description)

        if specific_concerns:
            result["specific_analysis"] = await self._analyze_specific_concerns(
                image_description, specific_concerns
            )

        self.record_action("photography_analysis", result)
        return result

    async def analyze_lighting(self, image_description: str) -> Dict[str, Any]:
        """
        Analyze lighting quality, direction, and mood.

        Args:
            image_description: Description of the image to analyze

        Returns:
            Dictionary containing:
                - light_quality: Assessment of light quality (hard/soft)
                - light_direction: Primary and secondary light directions
                - color_temperature: Estimated color temperature in Kelvin
                - shadow_analysis: Shadow consistency and direction
                - mood_assessment: Emotional impact of lighting
                - issues: List of lighting issues detected
                - suggestions: Improvement recommendations
        """
        prompt = f"""Analyze the lighting in the following image description from a professional photographer's perspective.

Image Description:
{image_description}

Provide a comprehensive lighting analysis covering:
1. Light Quality: Is the light hard (sharp shadows) or soft (diffused shadows)?
2. Light Direction: Where is the primary light coming from? Are there secondary lights?
3. Color Temperature: Estimate the color temperature (warm/cool, approximate Kelvin)
4. Shadow Analysis: Are shadows consistent with light sources? Any anomalies?
5. Mood Assessment: What emotional tone does the lighting create?
6. Issues: List any lighting inconsistencies or problems
7. Suggestions: How could the lighting be improved?

Respond in JSON format with keys: light_quality, light_direction, color_temperature, shadow_analysis, mood_assessment, issues, suggestions"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "parsing_note": "Response could not be parsed as JSON"
            }

    async def analyze_composition(self, image_description: str) -> Dict[str, float]:
        """
        Analyze and score image composition.

        Args:
            image_description: Description of the image to analyze

        Returns:
            Dictionary with scores (0.0-1.0) for:
                - rule_of_thirds: Adherence to rule of thirds
                - balance: Visual weight balance
                - leading_lines: Use of leading lines
                - symmetry: Symmetry score
                - depth: Sense of depth and layering
                - framing: Quality of framing
                - overall: Overall composition score
        """
        prompt = f"""Analyze the composition of the following image and provide scores for each aspect.

Image Description:
{image_description}

Score each composition element from 0.0 (poor) to 1.0 (excellent):
1. rule_of_thirds: Are key elements positioned along thirds or at intersections?
2. balance: Is visual weight distributed harmoniously?
3. leading_lines: Are there effective lines guiding the viewer's eye?
4. symmetry: Is symmetry (or intentional asymmetry) well executed?
5. depth: Does the image convey depth through layering, perspective, or atmospheric effects?
6. framing: Is the subject well-framed within the image boundaries?
7. overall: Overall composition quality

Respond ONLY with a JSON object containing these 7 keys with float values between 0.0 and 1.0."""

        response = await self.call_llm(prompt)

        try:
            scores = json.loads(response)
            # Validate scores are in range
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    scores[key] = max(0.0, min(1.0, float(value)))
            return scores
        except (json.JSONDecodeError, ValueError):
            return {
                "rule_of_thirds": 0.5,
                "balance": 0.5,
                "leading_lines": 0.5,
                "symmetry": 0.5,
                "depth": 0.5,
                "framing": 0.5,
                "overall": 0.5,
                "parsing_note": "Could not parse scores, returning defaults"
            }

    async def assess_realism(self, image_description: str) -> Dict[str, Any]:
        """
        Assess photorealism and detect AI artifacts.

        Args:
            image_description: Description of the image to analyze

        Returns:
            Dictionary containing:
                - realism_score: Overall realism score (0.0-1.0)
                - ai_artifacts: List of detected AI generation artifacts
                - unnatural_elements: Elements that appear unnatural
                - physics_consistency: Assessment of physical accuracy
                - texture_quality: Naturalness of textures
                - lighting_consistency: Whether lighting is physically accurate
                - verdict: "realistic", "suspect", or "likely_ai_generated"
        """
        prompt = f"""Analyze the following image description for photorealism and AI generation artifacts.

Image Description:
{image_description}

As an experienced photographer, identify signs that this image may be AI-generated or manipulated:

1. Realism Score (0.0-1.0): How realistic does the image appear?
2. AI Artifacts: List common AI artifacts (distorted hands, impossible reflections, inconsistent details, text gibberish, etc.)
3. Unnatural Elements: Any elements that look artificial or implausible
4. Physics Consistency: Are lighting, shadows, reflections, and perspectives physically accurate?
5. Texture Quality: Do textures appear natural or artificially generated?
6. Lighting Consistency: Does all lighting come from consistent sources?
7. Verdict: Based on analysis, classify as "realistic", "suspect", or "likely_ai_generated"

Respond in JSON format with keys: realism_score, ai_artifacts, unnatural_elements, physics_consistency, texture_quality, lighting_consistency, verdict"""

        response = await self.call_llm(prompt)

        try:
            result = json.loads(response)
            if "realism_score" in result:
                result["realism_score"] = max(0.0, min(1.0, float(result["realism_score"])))
            return result
        except (json.JSONDecodeError, ValueError):
            return {
                "realism_score": 0.5,
                "ai_artifacts": ["Analysis could not be parsed"],
                "unnatural_elements": [],
                "physics_consistency": "Unable to determine",
                "texture_quality": "Unable to determine",
                "lighting_consistency": "Unable to determine",
                "verdict": "suspect"
            }

    async def _analyze_specific_concerns(
        self,
        image_description: str,
        concerns: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze specific photography concerns.

        Args:
            image_description: Image description
            concerns: List of specific concerns to address

        Returns:
            Analysis addressing each concern
        """
        concerns_text = "\n".join(f"- {concern}" for concern in concerns)

        prompt = f"""Address the following specific photography concerns for this image.

Image Description:
{image_description}

Specific Concerns:
{concerns_text}

Provide detailed analysis for each concern from a professional photographer's perspective."""

        response = await self.call_llm(prompt)

        return {
            "concerns_addressed": concerns,
            "analysis": response
        }

    async def suggest_improvements(
        self,
        image_description: str,
        target_use: str = "commercial"
    ) -> Dict[str, Any]:
        """
        Suggest photography improvements based on intended use.

        Args:
            image_description: Description of current image
            target_use: Intended use (commercial, editorial, social media, etc.)

        Returns:
            Improvement suggestions tailored to the target use
        """
        prompt = f"""Provide improvement suggestions for this image based on its intended use.

Image Description:
{image_description}

Intended Use: {target_use}

Consider:
- Technical improvements (lighting, composition, color grading)
- Content adjustments for the target audience
- Commercial viability if applicable
- Common pitfalls to avoid for this use case

Provide actionable, specific suggestions."""

        response = await self.call_llm(prompt)

        return {
            "target_use": target_use,
            "suggestions": response
        }
