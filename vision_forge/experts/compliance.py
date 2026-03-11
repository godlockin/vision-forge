"""Compliance & Legal expert implementation with veto power."""

from typing import Dict, Any, List, Optional
from enum import Enum

from ..experts.expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceExpert(Expert):
    """
    Compliance & Legal expert with veto power.

    Responsibilities:
    - Pre-task compliance review (一票否决)
    - Final output compliance review
    - Auto-fallback for fixable violations
    """

    # Violation categories and their default severity
    VIOLATION_CATEGORIES = {
        "copyright_ip": ViolationSeverity.MEDIUM,
        "brand_trademark": ViolationSeverity.MEDIUM,
        "celebrity_likeness": ViolationSeverity.MEDIUM,
        "sensitive_content": ViolationSeverity.HIGH,
        "political_sensitive": ViolationSeverity.CRITICAL,
        "violence_gore": ViolationSeverity.CRITICAL,
        "adult_content": ViolationSeverity.CRITICAL,
        "hate_speech": ViolationSeverity.CRITICAL,
        "harassment": ViolationSeverity.HIGH,
        "self_harm": ViolationSeverity.CRITICAL,
    }

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        super().__init__(config, blackboard, model_router)

    async def review_request(self, user_prompt: str) -> Dict[str, Any]:
        """
        Review user request for compliance.

        Args:
            user_prompt: User's original prompt

        Returns:
            Review result with pass/fail and violations
        """
        system_prompt = """You are a compliance and legal expert. Review the following
user request for potential legal, copyright, or content safety issues.

Categories to check:
- Copyright/IP infringement
- Brand/trademark violations
- Celebrity likeness rights
- Sensitive political content
- Violence/gore
- Adult content
- Hate speech
- Harassment
- Self-harm

Return JSON with:
{
    "passed": bool,
    "violations": [{"type": str, "severity": str, "description": str}],
    "severity": "low|medium|high|critical",
    "can_autofix": bool
}"""

        response = await self.call_llm(
            f"Review this request for compliance issues:\n{user_prompt}",
            system_prompt=system_prompt,
            max_tokens=1000
        )

        return self._parse_compliance_response(response)

    async def review_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review final output before delivery.

        Args:
            output_data: Generated output data

        Returns:
            Review result
        """
        # Convert output to text for review
        output_text = str(output_data)

        return await self.review_request(output_text)

    def can_autofallback(self, violation_type: str) -> bool:
        """
        Check if violation can be auto-remediated.

        Args:
            violation_type: Type of violation

        Returns:
            True if auto-fallback is allowed
        """
        severity = self.VIOLATION_CATEGORIES.get(
            violation_type,
            ViolationSeverity.HIGH
        )
        return severity in [ViolationSeverity.LOW, ViolationSeverity.MEDIUM]

    def get_severity(self, violation_type: str) -> ViolationSeverity:
        """
        Get severity for violation type.

        Args:
            violation_type: Type of violation

        Returns:
            Severity level
        """
        return self.VIOLATION_CATEGORIES.get(
            violation_type,
            ViolationSeverity.HIGH
        )

    def has_veto_power(self) -> bool:
        """
        Check if this expert has veto power.

        Returns:
            True (always has veto power)
        """
        veto_config = self.config.veto_power
        return veto_config.get("enabled", True) if veto_config else True

    def _parse_compliance_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM compliance response.

        Args:
            response: LLM response text

        Returns:
            Parsed compliance result
        """
        import json
        try:
            # Try to parse JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: simple keyword analysis
        critical_keywords = ['illegal', 'copyright', 'trademark', 'political', 'violence', 'adult']
        response_lower = response.lower()

        violations = []
        for keyword in critical_keywords:
            if keyword in response_lower:
                violations.append({
                    "type": keyword,
                    "severity": "medium",
                    "description": f"Potential {keyword} issue detected"
                })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "severity": "high" if violations else "low",
            "can_autofix": len(violations) <= 2
        }

    async def build_rejection_report(
        self,
        violations: List[Dict[str, Any]]
    ) -> str:
        """
        Build detailed rejection report.

        Args:
            violations: List of violations

        Returns:
            Rejection report text
        """
        report_lines = [
            "## Compliance Review Failed",
            "",
            f"Total violations found: {len(violations)}",
            ""
        ]

        for i, v in enumerate(violations, 1):
            report_lines.append(
                f"{i}. **{v.get('type', 'Unknown')}** "
                f"(Severity: {v.get('severity', 'unknown')})"
            )
            report_lines.append(f"   Description: {v.get('description', 'N/A')}")
            report_lines.append("")

        report_lines.append("## Recommendation")
        report_lines.append(
            "Please revise your request to address the above compliance issues."
        )

        return "\n".join(report_lines)

    def _get_task_type(self) -> TaskType:
        """Get task type for model routing."""
        return TaskType.COMPLIANCE_CHECK
