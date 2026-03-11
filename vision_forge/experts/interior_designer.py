"""Interior Designer expert for spatial layout and style matching."""

import json
from typing import Dict, Any, List

from .expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class InteriorDesignerExpert(Expert):
    """
    Professional interior designer expert for space planning and style coordination.

    Specializes in:
    - Spatial layout optimization and flow analysis
    - Style matching and coordination across design elements
    - Furniture placement for function and aesthetics
    - Color scheme development and coordination
    - Budget and effect balancing

    This expert analyzes interior spaces from both functional and aesthetic
    perspectives, providing actionable recommendations for improvements.
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize interior designer expert.

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
        Process interior design task.

        Args:
            task_data: Task data containing:
                - room_description: Description of the room/space
                - user_preferences: Optional style preferences
                - constraints: Optional constraints (budget, dimensions, etc.)

        Returns:
            Design analysis with spatial, style, and furniture recommendations
        """
        room_description = task_data.get("room_description", "")
        user_preferences = task_data.get("user_preferences", {})
        constraints = task_data.get("constraints", {})

        result: Dict[str, Any] = {
            "expert_id": self.id,
            "expert_role": self.role,
            "confidence": self._confidence,
        }

        # Analyze spatial layout
        result["spatial_analysis"] = await self.analyze_spatial_layout(room_description)

        # Perform style matching if preferences provided
        if user_preferences:
            result["style_matching"] = await self.style_matching(
                user_preferences, room_description
            )

        # Suggest furniture placement if dimensions provided
        if constraints.get("room_dimensions"):
            furniture_list = constraints.get("furniture", [])
            if furniture_list:
                result["furniture_placement"] = await self.suggest_furniture_placement(
                    constraints["room_dimensions"],
                    furniture_list
                )

        # Add color and material suggestions
        result["color_material"] = await self._suggest_color_material(
            room_description, user_preferences
        )

        self.record_action("interior_design_analysis", result)
        return result

    async def analyze_spatial_layout(self, room_description: str) -> Dict[str, Any]:
        """
        Analyze space utilization, flow, and functionality.

        Args:
            room_description: Description of the room including dimensions,
                              existing elements, and architectural features

        Returns:
            Dictionary containing:
                - space_utilization: Assessment of how well space is used
                - traffic_flow: Analysis of movement patterns
                - functional_zones: Identified functional areas
                - focal_points: Key visual focal points
                - architectural_features: Notable architectural elements
                - issues: Spatial problems or inefficiencies
                - recommendations: Improvement suggestions
        """
        prompt = f"""Analyze this interior space from a professional interior designer's perspective.

Room Description:
{room_description}

Provide a comprehensive spatial analysis covering:
1. Space Utilization: How efficiently is the available space being used? (rate 0.0-1.0)
2. Traffic Flow: Are circulation paths clear and logical? Describe the flow patterns.
3. Functional Zones: What functional areas exist or should exist? (e.g., seating, work, storage)
4. Focal Points: What are the natural or potential focal points?
5. Architectural Features: Notable elements like windows, doors, columns, built-ins
6. Issues: Any spatial problems, inefficiencies, or awkward areas
7. Recommendations: Specific suggestions for improving the spatial layout

Respond in JSON format with keys: space_utilization, traffic_flow, functional_zones, focal_points, architectural_features, issues, recommendations"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "parsing_note": "Response could not be parsed as JSON"
            }

    async def style_matching(
        self,
        user_preference: str,
        room_context: str
    ) -> Dict[str, Any]:
        """
        Match and recommend design styles based on user preferences.

        Args:
            user_preference: User's style preference description
            room_context: Context about the room and its requirements

        Returns:
            Dictionary containing:
                - matched_styles: List of styles that match user preference
                - primary_recommendation: Best matching style with rationale
                - style_elements: Key elements of recommended style
                - color_palette: Suggested colors for the style
                - material_suggestions: Appropriate materials and finishes
                - furniture_styles: Furniture styles that complement
                - dos_and_donts: Style-specific guidelines
        """
        prompt = f"""Match design styles to user preferences and room context.

User Style Preference:
{user_preference}

Room Context:
{room_context}

Provide comprehensive style matching:
1. Matched Styles: List 3-5 design styles that align with the preference (e.g., modern, Scandinavian, industrial, minimalist, traditional, mid-century modern, bohemian, coastal, farmhouse, art deco)
2. Primary Recommendation: Which style is the best fit and why?
3. Style Elements: What are the defining characteristics of the recommended style?
4. Color Palette: What colors work best with this style?
5. Material Suggestions: What materials and finishes are appropriate?
6. Furniture Styles: What furniture styles complement this aesthetic?
7. Dos and Don'ts: Key guidelines for maintaining style consistency

Respond in JSON format with keys: matched_styles, primary_recommendation, style_elements, color_palette, material_suggestions, furniture_styles, dos_and_donts"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "matched_styles": ["Analysis could not be parsed"],
                "primary_recommendation": response,
                "style_elements": [],
                "color_palette": [],
                "material_suggestions": [],
                "furniture_styles": [],
                "dos_and_donts": []
            }

    async def suggest_furniture_placement(
        self,
        room_dims: Dict[str, Any],
        furniture: List[str]
    ) -> Dict[str, Any]:
        """
        Suggest optimal furniture placement for a room.

        Args:
            room_dims: Room dimensions and features, e.g.:
                       {
                         "length": 5.0,
                         "width": 4.0,
                         "unit": "meters",
                         "features": ["window on north wall", "door on east wall"]
                       }
            furniture: List of furniture pieces to place

        Returns:
            Dictionary containing:
                - layout_suggestion: Recommended furniture arrangement
                - placement_details: Specific placement for each piece
                - traffic_flow_analysis: How the layout affects movement
                - visual_balance: Assessment of visual weight distribution
                - alternative_layouts: Other viable arrangements
                - spacing_requirements: Minimum clearances to maintain
        """
        room_info = f"""
Room Dimensions: {room_dims.get('length', 'N/A')} x {room_dims.get('width', 'N/A')} {room_dims.get('unit', 'units')}
Room Features: {', '.join(room_dims.get('features', [])) or 'None specified'}
"""
        furniture_list = "\n".join(f"- {item}" for item in furniture)

        prompt = f"""Suggest optimal furniture placement for this room.

{room_info}

Furniture to Place:
{furniture_list}

Provide detailed placement suggestions:
1. Layout Suggestion: Overall arrangement strategy
2. Placement Details: Specific position for each furniture piece (include approximate positions)
3. Traffic Flow Analysis: How does this layout affect movement through the space?
4. Visual Balance: Is visual weight well-distributed? Where is the focal point?
5. Alternative Layouts: What other arrangements could work?
6. Spacing Requirements: What clearances should be maintained? (e.g., walkways, around furniture)

Respond in JSON format with keys: layout_suggestion, placement_details, traffic_flow_analysis, visual_balance, alternative_layouts, spacing_requirements"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "layout_suggestion": response,
                "placement_details": "Could not parse detailed placements",
                "traffic_flow_analysis": "N/A",
                "visual_balance": "N/A",
                "alternative_layouts": [],
                "spacing_requirements": []
            }

    async def _suggest_color_material(
        self,
        room_description: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate color and material suggestions."""
        style_pref = user_preferences.get("style", "not specified") if user_preferences else "not specified"

        prompt = f"""Suggest color schemes and materials for this interior space.

Room Description:
{room_description}

Style Preference:
{style_pref}

Provide recommendations for:
1. Primary Color Scheme: Main colors for walls, large surfaces
2. Accent Colors: Colors for accessories and highlights
3. Material Palette: Flooring, countertops, fixtures
4. Texture Mix: How to combine different textures
5. Lighting Considerations: How lighting affects color and material choices

Respond in JSON format with keys: primary_colors, accent_colors, materials, textures, lighting_notes"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "suggestions": response,
                "parsing_note": "Could not parse as JSON"
            }

    async def evaluate_design_coherence(
        self,
        design_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate how well design elements work together.

        Args:
            design_elements: Dictionary of design elements like:
                           {
                             "furniture_style": "...",
                             "color_scheme": "...",
                             "materials": "...",
                             "lighting": "..."
                           }

        Returns:
            Coherence assessment with consistency score and recommendations
        """
        elements_text = "\n".join(f"{k}: {v}" for k, v in design_elements.items())

        prompt = f"""Evaluate the coherence of these design elements.

Design Elements:
{elements_text}

Assess:
1. Overall Coherence Score (0.0-1.0): How well do all elements work together?
2. Style Consistency: Are all elements consistent with the intended style?
3. Color Harmony: Do colors work together effectively?
4. Material Coordination: Do materials complement each other?
5. Conflicts: Any elements that clash or feel out of place?
6. Recommendations: How to improve overall coherence?

Respond in JSON format with keys: coherence_score, style_consistency, color_harmony, material_coordination, conflicts, recommendations"""

        response = await self.call_llm(prompt)

        try:
            result = json.loads(response)
            if "coherence_score" in result:
                result["coherence_score"] = max(0.0, min(1.0, float(result["coherence_score"])))
            return result
        except (json.JSONDecodeError, ValueError):
            return {
                "coherence_score": 0.5,
                "style_consistency": "Unable to evaluate",
                "color_harmony": "Unable to evaluate",
                "material_coordination": "Unable to evaluate",
                "conflicts": [],
                "recommendations": response
            }

    async def suggest_budget_allocation(
        self,
        total_budget: float,
        room_type: str,
        priorities: List[str]
    ) -> Dict[str, Any]:
        """
        Suggest how to allocate budget across design elements.

        Args:
            total_budget: Total available budget
            room_type: Type of room (living room, bedroom, kitchen, etc.)
            priorities: User's priority areas

        Returns:
            Budget breakdown with allocation recommendations
        """
        priorities_text = ", ".join(priorities) if priorities else "No specific priorities"

        prompt = f"""Suggest budget allocation for this interior design project.

Room Type: {room_type}
Total Budget: ${total_budget:,.2f}
Priorities: {priorities_text}

Provide budget allocation:
1. Category Breakdown: How much to spend on furniture, materials, labor, accessories, etc.
2. Priority Allocation: Ensure priority areas receive appropriate budget
3. Splurge vs Save: Where to invest vs where to economize
4. Percentage Breakdown: What percentage for each category
5. Phasing Suggestions: If budget is tight, what to do first vs later

Respond in JSON format with keys: category_breakdown, priority_allocation, splurge_vs_save, percentages, phasing"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "allocation_advice": response,
                "parsing_note": "Could not parse as JSON"
            }
