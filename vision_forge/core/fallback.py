"""Auto-fallback handler for compliance violations."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from ..services.router import ModelRouter, TaskType
from ..services.base import ServiceResponse


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    attempt_number: int
    action: str
    success: bool
    error: Optional[str] = None
    modified_prompt: Optional[str] = None


class AutoFallbackHandler:
    """
    Handle automatic remediation for compliance violations.

    Strategies:
    - Replace copyrighted elements with generics
    - Remove branding from prompts
    - Stylize or abstract celebrity likenesses
    - Enhance details to fix quality issues
    """

    MAX_ATTEMPTS = 3

    # Remediation strategies by violation type
    STRATEGIES = {
        "copyright_ip": "replace_with_generic",
        "brand_trademark": "remove_branding",
        "celebrity_likeness": "stylize_or_abstract",
        "quality_issue": "enhance_details",
        "sensitive_content": "reframe_context",
    }

    def __init__(self, model_router: ModelRouter):
        """
        Initialize fallback handler.

        Args:
            model_router: Model router for LLM calls
        """
        self.model_router = model_router
        self.attempts: List[FallbackAttempt] = []
        self._original_prompt: Optional[str] = None

    async def attempt_remediation(
        self,
        violation_type: str,
        original_prompt: str,
        violation_details: str
    ) -> Tuple[bool, str]:
        """
        Attempt to fix violation automatically.

        Args:
            violation_type: Type of violation
            original_prompt: Original user prompt
            violation_details: Description of the violation

        Returns:
            Tuple of (success, modified_prompt)
        """
        self._original_prompt = original_prompt
        self.attempts.clear()

        strategy = self._get_strategy(violation_type)

        for attempt_num in range(1, self.MAX_ATTEMPTS + 1):
            modified_prompt = await self._apply_remediation(
                original_prompt,
                violation_details,
                strategy,
                attempt_num
            )

            # Validate the fix
            is_valid = await self._validate_fix(
                modified_prompt,
                violation_type,
                violation_details
            )

            self.attempts.append(FallbackAttempt(
                attempt_number=attempt_num,
                action=strategy,
                success=is_valid,
                modified_prompt=modified_prompt
            ))

            if is_valid:
                return True, modified_prompt

            # Try different approach on next attempt
            strategy = self._get_alternate_strategy(strategy)

        return False, original_prompt

    def _get_strategy(self, violation_type: str) -> str:
        """
        Get remediation strategy for violation type.

        Args:
            violation_type: Type of violation

        Returns:
            Strategy name
        """
        return self.STRATEGIES.get(
            violation_type,
            "modify_description"
        )

    def _get_alternate_strategy(self, current_strategy: str) -> str:
        """
        Get alternative strategy.

        Args:
            current_strategy: Current strategy

        Returns:
            Alternative strategy
        """
        alternates = {
            "replace_with_generic": "abstract_and_stylize",
            "remove_branding": "change_context",
            "stylize_or_abstract": "replace_with_generic",
            "enhance_details": "simplify_and_focus",
            "modify_description": "complete_reframe",
        }
        return alternates.get(current_strategy, "modify_description")

    async def _apply_remediation(
        self,
        prompt: str,
        violation: str,
        strategy: str,
        attempt: int
    ) -> str:
        """
        Apply remediation to prompt.

        Args:
            prompt: Original prompt
            violation: Violation description
            strategy: Remediation strategy
            attempt: Attempt number

        Returns:
            Modified prompt
        """
        system_prompt = f"""You are an expert at modifying prompts to avoid compliance violations.

Current strategy: {strategy}
Attempt: {attempt}

Your task is to modify the prompt to avoid the violation while preserving the user's original intent as much as possible.

Return ONLY the modified prompt, no explanation."""

        response = await self.model_router.route_request(
            TaskType.TEXT_REASONING,
            f"""Original prompt: {prompt}

Violation to address: {violation}

Modify this prompt using the {strategy} strategy. Return only the modified prompt."""
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def _validate_fix(
        self,
        prompt: str,
        violation_type: str,
        violation_details: str
    ) -> bool:
        """
        Validate that fix addresses the violation.

        Args:
            prompt: Modified prompt
            violation_type: Type of violation
            violation_details: Original violation details

        Returns:
            True if fix is valid
        """
        response = await self.model_router.route_request(
            TaskType.COMPLIANCE_CHECK,
            f"""Check if the following prompt addresses the {violation_type} concern.

Original violation: {violation_details}
Modified prompt: {prompt}

Respond with ONLY "PASS" if the violation is addressed, or "FAIL" with a brief reason if not."""
        )

        result = response.content if hasattr(response, 'content') else str(response)
        return "PASS" in result.upper()

    def get_attempts(self) -> List[Dict[str, Any]]:
        """
        Get all fallback attempts.

        Returns:
            List of attempt records
        """
        return [
            {
                "attempt": a.attempt_number,
                "action": a.action,
                "success": a.success,
                "error": a.error
            }
            for a in self.attempts
        ]

    def get_success_rate(self) -> float:
        """
        Get historical success rate.

        Returns:
            Success rate (0.0-1.0)
        """
        if not self.attempts:
            return 0.0

        successful = sum(1 for a in self.attempts if a.success)
        return successful / len(self.attempts)
