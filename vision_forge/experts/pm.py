"""Project Manager expert implementation."""

from typing import Dict, Any, List
from ..experts.expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter, TaskType


class ProjectManagerExpert(Expert):
    """
    Project Manager expert implementation.

    Responsibilities:
    - Task decomposition
    - Expert coordination
    - Deadlock detection and resolution
    - Weighted voting
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        super().__init__(config, blackboard, model_router)

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage task decomposition and expert coordination.

        Args:
            task_data: Task data including task_id and requirements

        Returns:
            Decision and rationale
        """
        task_id = task_data.get("task_id", "unknown")

        # Gather opinions from blackboard
        opinions = self._gather_opinions(task_id)

        # Check for deadlock
        if self._is_deadlock(opinions):
            return await self._force_decision(task_data, opinions)

        # Conduct weighted voting
        decision = await self._weighted_vote(task_data, opinions)

        return {
            "decision": decision,
            "rationale": self._build_rationale(decision, opinions)
        }

    def _gather_opinions(self, task_id: str) -> List[Dict]:
        """
        Gather all opinions from blackboard.

        Args:
            task_id: Task ID

        Returns:
            List of opinion data
        """
        events = self.blackboard.get_events(task_id=task_id)
        return [
            e.data for e in events
            if e.data.get("opinion")
        ]

    def _is_deadlock(self, opinions: List[Dict], threshold: int = 5) -> bool:
        """
        Detect if discussion is deadlocked.

        Args:
            opinions: List of opinions
            threshold: Number of rounds to consider

        Returns:
            True if deadlocked
        """
        if len(opinions) < threshold:
            return False

        recent = opinions[-threshold:]
        # Check if opinions are not converging
        unique_opinions = len(set(str(o.get("opinion", "")) for o in recent))
        return unique_opinions == threshold

    async def _weighted_vote(
        self,
        task_data: Dict,
        opinions: List[Dict]
    ) -> Dict:
        """
        Conduct weighted voting.

        Args:
            task_data: Task data
            opinions: List of opinions

        Returns:
            Decision result
        """
        if not opinions:
            return {"action": "proceed", "confidence": 0.5}

        # Calculate weighted scores
        weighted_scores = []
        for opinion in opinions:
            base_score = opinion.get("score", 0.5)
            confidence = opinion.get("confidence", 0.5) or 0.5

            # Weight by confidence
            weighted = base_score * (0.5 + confidence * 0.5)
            weighted_scores.append(weighted)

        avg_score = sum(weighted_scores) / len(weighted_scores)

        return {
            "action": "proceed" if avg_score > 0.5 else "revise",
            "confidence": avg_score,
            "vote_count": len(opinions)
        }

    async def _force_decision(
        self,
        task_data: Dict,
        opinions: List[Dict]
    ) -> Dict:
        """
        Force decision when deadlocked.

        Args:
            task_data: Task data
            opinions: List of opinions

        Returns:
            Forced decision
        """
        # Use LLM to make final decision
        prompt = f"""Given the following expert opinions, make a final decision:

{chr(10).join(f'- Expert: {o.get("opinion", "N/A")}' for o in opinions)}

Provide a clear decision with rationale."""

        response = await self.call_llm(prompt, max_tokens=500)

        return {
            "action": "force_proceed",
            "reason": "deadlock_detected",
            "llm_decision": response
        }

    def _build_rationale(self, decision: Dict, opinions: List[Dict]) -> str:
        """
        Build decision rationale.

        Args:
            decision: Decision result
            opinions: List of opinions

        Returns:
            Rationale string
        """
        return (
            f"Decision made based on {len(opinions)} expert opinions. "
            f"Confidence: {decision.get('confidence', 0):.2f}. "
            f"Action: {decision.get('action', 'unknown')}"
        )

    async def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose task into subtasks.

        Args:
            task_description: Task description

        Returns:
            List of subtasks
        """
        prompt = f"""Decompose the following task into clear, actionable subtasks:

{task_description}

Return a list of subtasks with:
1. Subtask name
2. Required expert capabilities
3. Expected output

Format as JSON."""

        response = await self.call_llm(prompt, max_tokens=1000)

        # Parse response (simplified)
        return [
            {"name": "subtask", "description": response, "experts": ["visual_expert"]}
        ]

    def _get_task_type(self) -> TaskType:
        """Get task type for model routing."""
        return TaskType.TEXT_REASONING
