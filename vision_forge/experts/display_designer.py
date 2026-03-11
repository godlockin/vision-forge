"""Display Designer expert for product display and visual marketing."""

import json
from typing import Dict, Any, List

from .expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class DisplayDesignerExpert(Expert):
    """
    Commercial display designer expert for visual merchandising and product presentation.

    Specializes in:
    - Product display optimization for maximum visual impact
    - Visual marketing and consumer psychology
    - Visual hierarchy and focal point creation
    - Brand consistency in visual presentations
    - Lighting setups for different product types

    This expert analyzes product displays from a commercial perspective,
    optimizing for customer engagement and brand alignment.
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize display designer expert.

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
        Process display design task.

        Args:
            task_data: Task data containing:
                - product_info: Information about the product(s)
                - context: Display context (retail, e-commerce, exhibition, etc.)
                - brand_guidelines: Optional brand requirements
                - target_audience: Optional target demographic

        Returns:
            Display optimization recommendations with visual marketing insights
        """
        product_info = task_data.get("product_info", "")
        context = task_data.get("context", "retail")
        brand_guidelines = task_data.get("brand_guidelines", {})
        target_audience = task_data.get("target_audience", "")

        result: Dict[str, Any] = {
            "expert_id": self.id,
            "expert_role": self.role,
            "confidence": self._confidence,
        }

        # Optimize product display
        result["display_optimization"] = await self.optimize_product_display(
            product_info, context
        )

        # Visual marketing analysis
        if target_audience:
            product_type = task_data.get("product_type", "general")
            result["visual_marketing"] = await self.visual_marketing_analysis(
                target_audience, product_type
            )

        # Lighting suggestions for product
        if product_info:
            product_material = self._extract_material_info(product_info)
            desired_mood = self._get_mood_for_context(context)
            result["lighting_setup"] = await self.suggest_lighting_setup(
                product_material, desired_mood
            )

        # Brand consistency check if guidelines provided
        if brand_guidelines:
            result["brand_consistency"] = await self._check_brand_consistency(
                product_info, context, brand_guidelines
            )

        self.record_action("display_design_analysis", result)
        return result

    async def optimize_product_display(
        self,
        product_info: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Optimize product display for maximum visual impact.

        Args:
            product_info: Description of the product(s) to display
            context: Display context (retail store, e-commerce, exhibition, etc.)

        Returns:
            Dictionary containing:
                - display_strategy: Overall display approach
                - focal_point_placement: Where to position the main focal point
                - visual_hierarchy: How to layer visual elements
                - product_positioning: Optimal product angles and positions
                - supporting_elements: Props, backgrounds, or complementary items
                - color_strategy: Color choices for maximum impact
                - context_specific_tips: Tips specific to the display context
        """
        prompt = f"""Optimize this product display for maximum visual impact.

Product Information:
{product_info}

Display Context: {context}

Provide comprehensive display optimization:
1. Display Strategy: What's the overall approach for this product in this context?
2. Focal Point Placement: Where should the main visual focus be? How to draw attention there?
3. Visual Hierarchy: How to layer elements for optimal visual flow? (primary, secondary, tertiary)
4. Product Positioning: Best angles, heights, and positions for the product
5. Supporting Elements: What props, backgrounds, or complementary items would enhance the display?
6. Color Strategy: What colors work best for background, accents, and overall palette?
7. Context-Specific Tips: Specific recommendations for {context} displays

Respond in JSON format with keys: display_strategy, focal_point_placement, visual_hierarchy, product_positioning, supporting_elements, color_strategy, context_specific_tips"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "display_strategy": response,
                "focal_point_placement": "Analysis could not be parsed",
                "visual_hierarchy": [],
                "product_positioning": {},
                "supporting_elements": [],
                "color_strategy": {},
                "context_specific_tips": []
            }

    async def visual_marketing_analysis(
        self,
        target_audience: str,
        product_type: str
    ) -> Dict[str, Any]:
        """
        Analyze visual marketing strategy for target audience and product.

        Args:
            target_audience: Description of target demographic
            product_type: Category/type of product

        Returns:
            Dictionary containing:
                - audience_psychology: Key psychological triggers for the audience
                - visual_preferences: Likely visual preferences of the audience
                - messaging_alignment: How visuals support marketing messages
                - cultural_considerations: Cultural factors to consider
                - engagement_strategies: Tactics to maximize audience engagement
                - competitive_differentiation: How to stand out visually
        """
        prompt = f"""Analyze visual marketing strategy for this audience and product.

Target Audience:
{target_audience}

Product Type:
{product_type}

Provide visual marketing analysis:
1. Audience Psychology: What psychological triggers resonate with this audience? (aspirations, fears, desires)
2. Visual Preferences: What visual styles, colors, and aesthetics appeal to them?
3. Messaging Alignment: How should visuals support and reinforce marketing messages?
4. Cultural Considerations: Any cultural factors, symbols, or sensitivities to consider?
5. Engagement Strategies: Specific tactics to capture and hold their attention
6. Competitive Differentiation: How to visually stand out from competitors in this product category?

Respond in JSON format with keys: audience_psychology, visual_preferences, messaging_alignment, cultural_considerations, engagement_strategies, competitive_differentiation"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "audience_psychology": response,
                "visual_preferences": [],
                "messaging_alignment": "Analysis could not be parsed",
                "cultural_considerations": [],
                "engagement_strategies": [],
                "competitive_differentiation": []
            }

    async def suggest_lighting_setup(
        self,
        product_material: str,
        desired_mood: str
    ) -> Dict[str, Any]:
        """
        Suggest optimal lighting setup for product photography/display.

        Args:
            product_material: Primary material of the product (metal, glass, fabric, etc.)
            desired_mood: The emotional tone to convey (luxury, friendly, dramatic, etc.)

        Returns:
            Dictionary containing:
                - lighting_type: Recommended lighting type (soft, hard, mixed)
                - light_positions: Where to position key lights
                - color_temperature: Recommended color temperature
                - intensity_levels: Suggested light intensities
                - special_considerations: Material-specific lighting tips
                - mood_achievement: How the lighting creates the desired mood
                - equipment_suggestions: Specific lighting equipment recommendations
        """
        prompt = f"""Suggest optimal lighting setup for this product.

Product Material:
{product_material}

Desired Mood:
{desired_mood}

Provide comprehensive lighting recommendations:
1. Lighting Type: Soft, hard, or mixed lighting? Why?
2. Light Positions: Where to position key light, fill light, and accent lights?
3. Color Temperature: What Kelvin range works best? (warm 2700-3000K, neutral 3500-4500K, cool 5000K+)
4. Intensity Levels: Relative intensities for different lights
5. Special Considerations: Material-specific tips (e.g., avoiding reflections on metal/glass)
6. Mood Achievement: How does this lighting setup create the desired {desired_mood} mood?
7. Equipment Suggestions: Specific types of lights, modifiers, or equipment to use

Respond in JSON format with keys: lighting_type, light_positions, color_temperature, intensity_levels, special_considerations, mood_achievement, equipment_suggestions"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "lighting_type": "Analysis could not be parsed",
                "light_positions": {},
                "color_temperature": "N/A",
                "intensity_levels": {},
                "special_considerations": [],
                "mood_achievement": response,
                "equipment_suggestions": []
            }

    def _extract_material_info(self, product_info: str) -> str:
        """Extract material information from product description."""
        material_keywords = [
            "metal", "steel", "aluminum", "brass", "copper",
            "glass", "crystal", "plastic", "acrylic",
            "wood", "leather", "fabric", "textile", "ceramic",
            "stone", "marble", "concrete"
        ]

        found_materials = []
        product_info_lower = product_info.lower()

        for material in material_keywords:
            if material in product_info_lower:
                found_materials.append(material)

        return ", ".join(found_materials) if found_materials else "mixed materials"

    def _get_mood_for_context(self, context: str) -> str:
        """Get desired mood based on display context."""
        context_moods = {
            "retail": "inviting and commercial",
            "luxury": "sophisticated and exclusive",
            "e-commerce": "clean and product-focused",
            "exhibition": "dramatic and attention-grabbing",
            "corporate": "professional and trustworthy",
            "lifestyle": "relatable and aspirational",
        }
        return context_moods.get(context.lower(), "appealing and professional")

    async def _check_brand_consistency(
        self,
        product_info: str,
        context: str,
        brand_guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if display aligns with brand guidelines."""
        guidelines_summary = "\n".join(f"{k}: {v}" for k, v in brand_guidelines.items())

        prompt = f"""Check brand consistency for this product display.

Product Information:
{product_info}

Display Context: {context}

Brand Guidelines:
{guidelines_summary}

Evaluate:
1. Overall Brand Alignment Score (0.0-1.0): How well does the display align with brand guidelines?
2. Color Consistency: Do colors match brand palette?
3. Style Consistency: Does the visual style match brand personality?
4. Messaging Consistency: Does the display communicate brand values?
5. Potential Violations: Any elements that conflict with brand guidelines?
6. Recommendations: How to improve brand alignment?

Respond in JSON format with keys: alignment_score, color_consistency, style_consistency, messaging_consistency, potential_violations, recommendations"""

        response = await self.call_llm(prompt)

        try:
            result = json.loads(response)
            if "alignment_score" in result:
                result["alignment_score"] = max(0.0, min(1.0, float(result["alignment_score"])))
            return result
        except (json.JSONDecodeError, ValueError):
            return {
                "alignment_score": 0.5,
                "color_consistency": "Unable to evaluate",
                "style_consistency": "Unable to evaluate",
                "messaging_consistency": "Unable to evaluate",
                "potential_violations": [],
                "recommendations": response
            }

    async def analyze_competitor_displays(
        self,
        competitor_descriptions: List[str],
        product_category: str
    ) -> Dict[str, Any]:
        """
        Analyze competitor display strategies for differentiation opportunities.

        Args:
            competitor_descriptions: Descriptions of competitor displays
            product_category: The product category

        Returns:
            Competitive analysis with differentiation recommendations
        """
        competitors_text = "\n\n".join(
            f"Competitor {i+1}:\n{desc}"
            for i, desc in enumerate(competitor_descriptions)
        )

        prompt = f"""Analyze competitor displays for differentiation opportunities.

Product Category: {product_category}

Competitor Displays:
{competitors_text}

Provide competitive analysis:
1. Common Patterns: What approaches are competitors using?
2. Overused Elements: What visual cliches should be avoided?
3. Gaps in Market: What visual approaches are underserved?
4. Differentiation Opportunities: How can our display stand out?
5. Best Practices to Adopt: What are competitors doing well that we should learn from?
6. Risks to Avoid: What mistakes are competitors making?

Respond in JSON format with keys: common_patterns, overused_elements, market_gaps, differentiation_opportunities, best_practices, risks_to_avoid"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "parsing_note": "Could not parse as JSON"
            }

    async def suggest_seasonal_adaptations(
        self,
        base_display: str,
        seasons: List[str],
        product_type: str
    ) -> Dict[str, Any]:
        """
        Suggest how to adapt displays for different seasons or occasions.

        Args:
            base_display: Description of the base display setup
            seasons: List of seasons/occasions to adapt for
            product_type: Type of product

        Returns:
            Seasonal adaptation recommendations for each specified season
        """
        seasons_text = ", ".join(seasons)

        prompt = f"""Suggest seasonal adaptations for this product display.

Base Display:
{base_display}

Product Type: {product_type}

Seasons/Occasions to Adapt For: {seasons_text}

For each season/occasion, provide:
1. Color Adjustments: How to update the color palette
2. Thematic Elements: Seasonal props or decorations to add
3. Lighting Changes: How lighting should shift
4. Merchandising Focus: What products to emphasize

Respond in JSON format with a key for each season containing: color_adjustments, thematic_elements, lighting_changes, merchandising_focus"""

        response = await self.call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "adaptations": response,
                "parsing_note": "Could not parse as JSON"
            }
