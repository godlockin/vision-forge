"""Prompt Engineer expert for requirement clarification and model adaptation."""

from typing import Dict, Any, List, Optional
from ..experts.expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class PromptEngineer(Expert):
    """
    Prompt Engineering Expert for requirement clarification and model adaptation.

    Responsibilities:
    - Transform vague requests into precise, actionable prompts
    - Adapt prompts for specific target models (GPT-4o, Gemini, Imagen-3)
    - Inject artistic style references and constraints
    - Optimize prompts for image generation quality

    This expert has on_demand load strategy - loaded when:
    - User prompt is vague or ambiguous
    - Cross-model generation comparison is needed
    - Image generation results consistently fail to meet expectations
    """

    # Model-specific prompt optimization rules
    MODEL_PROMPT_RULES: Dict[str, Dict[str, Any]] = {
        "gpt-4o": {
            "style": "direct",
            "format": "natural_language",
            "strengths": ["creative_interpretation", "nuanced_understanding"],
            "weaknesses": ["literal_visualization"],
            "optimal_length": "medium",
            "tips": [
                "Use clear, descriptive language",
                "Specify mood and atmosphere explicitly",
                "Include compositional guidance"
            ]
        },
        "gemini": {
            "style": "detailed",
            "format": "structured",
            "strengths": ["visual_accuracy", "detail_rendering"],
            "weaknesses": ["abstract_concepts"],
            "optimal_length": "long",
            "tips": [
                "Provide detailed visual descriptions",
                "Specify lighting and shadows explicitly",
                "Include color palette references"
            ]
        },
        "imagen-3": {
            "style": "photographic",
            "format": "scene_description",
            "strengths": ["photorealism", "composition"],
            "weaknesses": ["text_rendering", "complex_abstractions"],
            "optimal_length": "medium",
            "tips": [
                "Describe scene as a photographer would",
                "Specify camera angle and focal length",
                "Include lighting setup details"
            ]
        }
    }

    # Style reference categories
    STYLE_CATEGORIES: Dict[str, List[str]] = {
        "art_movements": [
            "Impressionism", "Cubism", "Surrealism", "Art Nouveau", "Art Deco",
            "Minimalism", "Abstract Expressionism", "Pop Art", "Post-Impressionism"
        ],
        "famous_artists": [
            "Monet", "Picasso", "Dali", "Van Gogh", "Kandinsky", "Warhol",
            "Rembrandt", "Vermeer", "Turner", "Hopper"
        ],
        "photography_styles": [
            "portrait", "landscape", "street photography", "documentary",
            "fashion photography", "macro", "long exposure", "golden hour"
        ],
        "cinematic_styles": [
            "noir", "neon-lit", "pastel aesthetic", "high contrast",
            "soft focus", "anamorphic", "dramatic lighting"
        ],
        "digital_art_styles": [
            "concept art", "pixel art", "vector art", "3D render",
            "matte painting", "character design"
        ]
    }

    # Constraint types
    CONSTRAINT_TYPES: Dict[str, Dict[str, Any]] = {
        "composition": {
            "rule_of_thirds": "Apply rule of thirds composition",
            "centered": "Center the main subject",
            "leading_lines": "Use leading lines to guide viewer's eye",
            "symmetry": "Create symmetrical composition",
            "depth_layers": "Include foreground, midground, and background"
        },
        "lighting": {
            "soft_diffused": "Soft, diffused lighting",
            "dramatic_side": "Dramatic side lighting with deep shadows",
            "backlit": "Backlit subject with rim lighting",
            "golden_hour": "Warm golden hour lighting",
            "studio_lighting": "Professional studio three-point lighting"
        },
        "color": {
            "monochromatic": "Single color palette with value variations",
            "complementary": "Complementary color scheme",
            "analogous": "Analogous colors for harmony",
            "warm_tones": "Dominant warm color temperature",
            "cool_tones": "Dominant cool color temperature"
        },
        "technical": {
            "shallow_depth": "Shallow depth of field, bokeh background",
            "sharp_focus": "Sharp focus throughout",
            "motion_blur": "Intentional motion blur for dynamism",
            "high_dynamic_range": "High dynamic range with detail in shadows and highlights"
        }
    }

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize Prompt Engineer.

        Args:
            config: Expert configuration from YAML
            blackboard: Shared blackboard instance for expert communication
            model_router: Model router for LLM calls
        """
        super().__init__(config, blackboard, model_router)

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process prompt engineering task.

        Main entry point for prompt engineer operations:
        - Analyze input prompt and requirements
        - Clarify vague requirements
        - Adapt for target model if specified
        - Inject style and constraints
        - Return optimized prompt with parameters

        Args:
            task_data: Task data including:
                - prompt: Original user prompt
                - context: Optional context information
                - target_model: Optional target model name
                - style_reference: Optional style reference
                - constraints: Optional list of constraints

        Returns:
            Dictionary containing:
                - original_prompt: The input prompt
                - clarified_prompt: Clarified version if needed
                - adapted_prompt: Model-adapted version if target_model specified
                - final_prompt: Final optimized prompt
                - parameters: Generation parameters
                - rationale: Explanation of optimizations made
        """
        original_prompt = task_data.get("prompt", "")
        context = task_data.get("context")
        target_model = task_data.get("target_model")
        style_reference = task_data.get("style_reference")
        constraints = task_data.get("constraints", [])

        result = {
            "original_prompt": original_prompt,
            "clarified_prompt": None,
            "adapted_prompt": None,
            "final_prompt": original_prompt,
            "parameters": {},
            "rationale": []
        }

        # Step 1: Clarify vague requirements
        if self._is_vague(original_prompt):
            clarified = await self.clarify_requirement(original_prompt, context)
            result["clarified_prompt"] = clarified
            result["final_prompt"] = clarified
            result["rationale"].append("Clarified vague requirements")

        # Step 2: Adapt for target model
        if target_model:
            current_prompt = result["final_prompt"]
            adapted = await self.adapt_for_model(current_prompt, target_model)
            result["adapted_prompt"] = adapted
            result["final_prompt"] = adapted
            result["rationale"].append(f"Adapted for {target_model}")
            result["parameters"] = self._get_model_params(target_model)

        # Step 3: Inject style reference
        if style_reference:
            current_prompt = result["final_prompt"]
            styled = await self.inject_style(current_prompt, style_reference)
            result["final_prompt"] = styled
            result["rationale"].append(f"Injected style: {style_reference}")

        # Step 4: Inject constraints
        if constraints:
            current_prompt = result["final_prompt"]
            constrained = await self.inject_constraints(current_prompt, constraints)
            result["final_prompt"] = constrained
            result["rationale"].append(f"Added {len(constraints)} constraints")

        # Step 5: Optimize for generation
        optimization_result = await self.optimize_for_generation(
            result["final_prompt"],
            result["parameters"]
        )
        result["final_prompt"] = optimization_result["optimized_prompt"]
        result["parameters"].update(optimization_result.get("parameters", {}))

        return result

    def _is_vague(self, prompt: str) -> bool:
        """
        Check if prompt is vague and needs clarification.

        Args:
            prompt: Prompt to evaluate

        Returns:
            True if prompt is vague
        """
        vague_indicators = [
            len(prompt.split()) < 10,
            any(word in prompt.lower() for word in ["something", "make it", "fix", "improve"]),
            not any(word in prompt.lower() for word in ["show", "display", "create", "generate", "image", "photo", "picture"]),
        ]
        return sum(vague_indicators) >= 2

    def _get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get optimal generation parameters for a model.

        Args:
            model_name: Model name

        Returns:
            Dictionary of parameters
        """
        model_lower = model_name.lower()

        if "gemini" in model_lower:
            return {
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        elif "imagen" in model_lower:
            return {
                "width": 1024,
                "height": 1024,
                "safety_filter_level": "block_some",
                "person_generation": "allow_adult"
            }
        elif "gpt-4o" in model_lower or "dall-e" in model_lower:
            return {
                "size": "1024x1024",
                "quality": "hd",
                "style": "vivid"
            }
        else:
            return {
                "width": 1024,
                "height": 1024
            }

    async def clarify_requirement(
        self,
        vague_prompt: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Transform vague requests into detailed, precise prompts.

        Uses LLM to expand and clarify ambiguous requirements while
        preserving the original intent.

        Args:
            vague_prompt: The vague/ambiguous input prompt
            context: Optional context information (e.g., user preferences,
                     previous iterations, domain constraints)

        Returns:
            Clarified and detailed prompt
        """
        # Build context string
        context_str = ""
        if context:
            context_parts = []
            for key, value in context.items():
                if isinstance(value, list):
                    context_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                else:
                    context_parts.append(f"{key}: {value}")
            context_str = f"\n\nAdditional Context:\n" + "\n".join(context_parts)

        prompt = f"""Transform the following vague image request into a detailed, precise prompt:

Original Request:
"{vague_prompt}"
{context_str}

Create a comprehensive prompt that includes:
1. **Subject**: Clear description of main subject(s)
2. **Setting/Environment**: Where the scene takes place
3. **Style**: Artistic or photographic style
4. **Lighting**: Type and direction of lighting
5. **Composition**: Camera angle, framing, perspective
6. **Mood/Atmosphere**: Emotional tone and feeling
7. **Color Palette**: Dominant colors or color scheme
8. **Technical Details**: Any specific technical requirements

Provide the clarified prompt directly, without explanations. The prompt should be 3-5 sentences, rich in visual detail but focused."""

        response = await self.call_llm(prompt, max_tokens=500)

        # Extract the clarified prompt (remove any meta-commentary)
        clarified = self._extract_clean_prompt(response)

        return clarified

    async def adapt_for_model(self, prompt: str, target_model: str) -> str:
        """
        Adapt prompt for specific target model.

        Each model has unique strengths and prompt preferences.
        This method optimizes the prompt for the target model's characteristics.

        Args:
            prompt: The prompt to adapt
            target_model: Target model name (e.g., "gpt-4o", "gemini-2.0-pro", "imagen-3")

        Returns:
            Model-optimized prompt
        """
        # Get model-specific rules
        model_rules = self._get_model_rules(target_model)

        # Build adaptation instructions based on model characteristics
        style = model_rules.get("style", "direct")
        format_type = model_rules.get("format", "natural_language")
        optimal_length = model_rules.get("optimal_length", "medium")
        tips = model_rules.get("tips", [])

        tips_str = "\n".join(f"- {tip}" for tip in tips)

        prompt_text = f"""Adapt the following image generation prompt for {target_model}:

Original Prompt:
"{prompt}"

Model Characteristics:
- Style preference: {style}
- Format: {format_type}
- Optimal length: {optimal_length}

Best Practices for {target_model}:
{tips_str}

Rewrite the prompt to be optimal for this specific model. Consider:
- Adjusting level of detail
- Structuring information appropriately
- Emphasizing the model's strengths
- Avoiding the model's weaknesses

Provide only the adapted prompt, no explanations."""

        response = await self.call_llm(prompt_text, max_tokens=500)

        return self._extract_clean_prompt(response)

    async def inject_style(self, prompt: str, style_reference: str) -> str:
        """
        Inject artistic style reference into prompt.

        Supports:
        - Art movements (Impressionism, Cubism, etc.)
        - Famous artists (Monet, Picasso, etc.)
        - Photography styles (portrait, landscape, etc.)
        - Cinematic styles (noir, neon-lit, etc.)
        - Digital art styles (concept art, pixel art, etc.)

        Args:
            prompt: Base prompt to style
            style_reference: Style reference (artist name, art movement, or style descriptor)

        Returns:
            Stylized prompt
        """
        # Identify the type of style reference
        style_type = self._identify_style_type(style_reference)

        prompt_text = f"""Add the following style reference to the image prompt:

Base Prompt:
"{prompt}"

Style Reference: "{style_reference}"
Style Type: {style_type}

Integrate the style naturally into the prompt:
- If artist: Mention their characteristic techniques, color usage, brushwork
- If art movement: Include defining characteristics of that movement
- If photography style: Use appropriate photographic terminology
- If cinematic: Include lighting, color grading, and compositional elements

The style should enhance the image without overwhelming the original intent.
Provide the styled prompt directly, no explanations."""

        response = await self.call_llm(prompt_text, max_tokens=400)

        return self._extract_clean_prompt(response)

    async def inject_constraints(self, prompt: str, constraints: List[str]) -> str:
        """
        Add technical/artistic constraints to prompt.

        Constraints can include:
        - Composition rules (rule of thirds, symmetry, etc.)
        - Lighting requirements (soft, dramatic, etc.)
        - Color palette (monochromatic, warm, etc.)
        - Technical specs (depth of field, sharpness, etc.)

        Args:
            prompt: Base prompt
            constraints: List of constraint strings or names

        Returns:
            Prompt with constraints integrated
        """
        constraints_str = "\n".join(f"- {constraint}" for constraint in constraints)

        # Expand constraint names to descriptions if needed
        expanded_constraints = []
        for constraint in constraints:
            expanded = self._expand_constraint(constraint)
            if expanded:
                expanded_constraints.append(expanded)
            else:
                expanded_constraints.append(constraint)

        expanded_str = "\n".join(f"- {c}" for c in expanded_constraints)

        prompt_text = f"""Add the following constraints to the image generation prompt:

Base Prompt:
"{prompt}"

Constraints to Apply:
{expanded_str}

Integrate each constraint naturally into the prompt:
- Composition constraints affect framing and subject placement
- Lighting constraints specify light quality and direction
- Color constraints define palette and mood
- Technical constraints specify camera/focus effects

Ensure constraints enhance rather than conflict with the base prompt.
Provide the constrained prompt directly, no explanations."""

        response = await self.call_llm(prompt_text, max_tokens=500)

        return self._extract_clean_prompt(response)

    async def optimize_for_generation(
        self,
        prompt: str,
        image_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimize prompt for image generation with parameters.

        Returns both the optimized prompt and recommended generation parameters.

        Args:
            prompt: Prompt to optimize
            image_params: Optional existing parameters to consider

        Returns:
            Dictionary containing:
                - optimized_prompt: Enhanced prompt for generation
                - parameters: Recommended generation parameters
                - negative_prompt: Optional negative prompt for exclusion
                - rationale: Explanation of optimizations
        """
        params_str = str(image_params) if image_params else "None"

        prompt_text = f"""Optimize this image generation prompt and provide recommended parameters:

Current Prompt:
"{prompt}"

Current Parameters: {params_str}

Provide:
1. An optimized version of the prompt (clearer, more evocative, better structured)
2. Recommended parameters:
   - Aspect ratio (e.g., "1:1", "16:9", "3:4")
   - Resolution (width x height)
   - Guidance/scale value
   - Any model-specific settings
3. A negative prompt to exclude common artifacts

Return JSON in this format:
{{
    "optimized_prompt": "string",
    "parameters": {{
        "aspect_ratio": "string",
        "width": number,
        "height": number,
        "guidance_scale": number,
        "additional_params": {{}}
    }},
    "negative_prompt": "string",
    "optimization_notes": ["list of changes made"]
}}"""

        response = await self.call_llm(prompt_text, max_tokens=800)

        return self._parse_optimization_response(response, image_params)

    def _identify_style_type(self, style_reference: str) -> str:
        """
        Identify the type of style reference.

        Args:
            style_reference: Style reference string

        Returns:
            Style type: "artist", "art_movement", "photography", "cinematic", or "digital_art"
        """
        style_lower = style_reference.lower()

        # Check each category
        for category, items in self.STYLE_CATEGORIES.items():
            if any(item.lower() in style_lower or style_lower in item.lower() for item in items):
                return category.replace("_", " ")

        # Heuristics for unknown styles
        if any(word in style_lower for word in ["ism", "style", "movement"]):
            return "art_movement"
        if any(word in style_lower for word in ["photo", "portrait", "landscape", "shot"]):
            return "photography"
        if any(word in style_lower for word in ["cinema", "film", "movie", "noir"]):
            return "cinematic"
        if any(word in style_lower for word in ["digital", "3d", "render", "pixel", "vector"]):
            return "digital_art"

        return "artist"  # Default assumption

    def _get_model_rules(self, model_name: str) -> Dict[str, Any]:
        """
        Get prompt rules for a specific model.

        Args:
            model_name: Model name

        Returns:
            Dictionary of model-specific rules
        """
        model_lower = model_name.lower()

        # Match against known models
        for model_key, rules in self.MODEL_PROMPT_RULES.items():
            if model_key in model_lower:
                return rules

        # Default rules for unknown models
        return {
            "style": "balanced",
            "format": "natural_language",
            "strengths": ["general_purpose"],
            "weaknesses": [],
            "optimal_length": "medium",
            "tips": ["Be clear and descriptive", "Include visual details"]
        }

    def _expand_constraint(self, constraint: str) -> Optional[str]:
        """
        Expand constraint name to full description.

        Args:
            constraint: Constraint name

        Returns:
            Expanded description or None if not found
        """
        constraint_lower = constraint.lower()

        for category, constraints in self.CONSTRAINT_TYPES.items():
            for name, description in constraints.items():
                if name.lower() in constraint_lower or constraint_lower in name.lower():
                    return description

        return None

    def _extract_clean_prompt(self, response: str) -> str:
        """
        Extract clean prompt from LLM response.

        Removes meta-commentary and explanations.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned prompt
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Clarified Prompt:",
            "Adapted Prompt:",
            "Styled Prompt:",
            "Constrained Prompt:",
            "Optimized Prompt:",
            "Here is the",
            "The prompt is:",
        ]

        result = response.strip()
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        # Remove markdown quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        # Remove any trailing explanations after a double newline
        if "\n\n" in result:
            result = result.split("\n\n")[0].strip()

        return result

    def _parse_optimization_response(
        self,
        response: str,
        default_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Parse optimization response from LLM.

        Args:
            response: Raw LLM response
            default_params: Default parameters to fall back on

        Returns:
            Parsed optimization result
        """
        import json

        try:
            # Try to extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)

                return {
                    "optimized_prompt": data.get("optimized_prompt", ""),
                    "parameters": data.get("parameters", default_params or {}),
                    "negative_prompt": data.get("negative_prompt", ""),
                    "rationale": data.get("optimization_notes", [])
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return original with defaults
        return {
            "optimized_prompt": response.strip(),
            "parameters": default_params or {},
            "negative_prompt": "",
            "rationale": ["Could not parse optimization response"]
        }

    def _get_task_type(self) -> TaskType:
        """
        Get task type for model routing.

        Prompt engineering is a text reasoning task.

        Returns:
            TaskType.TEXT_REASONING
        """
        return TaskType.TEXT_REASONING
