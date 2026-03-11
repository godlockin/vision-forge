"""Base class for all experts in the system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard
from ..services.router import ModelRouter
from ..services.base import ServiceResponse


class Expert(ABC):
    """
    Abstract base class for all experts.

    Each expert has:
    - Configuration from YAML
    - Access to shared blackboard
    - Access to model router for LLM calls
    - Confidence score for self-assessment
    - Executor/Critic mode
    """

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        """
        Initialize expert.

        Args:
            config: Expert configuration
            blackboard: Shared blackboard instance
            model_router: Model router for LLM access
        """
        self.config = config
        self.blackboard = blackboard
        self.model_router = model_router
        self._confidence: Optional[float] = None
        self._is_executor = True
        self._task_history: List[Dict[str, Any]] = []

    @property
    def id(self) -> str:
        """Get expert ID."""
        return self.config.id

    @property
    def role(self) -> str:
        """Get expert role name."""
        return self.config.role

    @property
    def is_executor(self) -> bool:
        """Check if expert is in executor mode."""
        return self._is_executor

    def set_confidence(self, confidence: float):
        """
        Set confidence score for this expert instance.

        Args:
            confidence: Confidence score (0.0-1.0)
        """
        self._confidence = max(0.0, min(1.0, confidence))

    def get_confidence(self) -> Optional[float]:
        """Get current confidence score."""
        return self._confidence

    def as_executor(self) -> "Expert":
        """Set expert to executor mode."""
        self._is_executor = True
        return self

    def as_critic(self) -> "Expert":
        """Set expert to critic mode."""
        self._is_executor = False
        return self

    @abstractmethod
    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task and return result.

        Args:
            task_data: Task-specific data

        Returns:
            Processing result
        """
        pass

    async def publish_opinion(
        self,
        task_id: str,
        opinion: str,
        score: float = 1.0,
        round_number: Optional[int] = None
    ):
        """
        Publish opinion to blackboard.

        Args:
            task_id: Task ID
            opinion: Opinion text
            score: Support score (0.0-1.0)
            round_number: Discussion round number
        """
        from ..memory.events import BlackboardEvent, EventType

        if round_number is None:
            round_number = self.blackboard.get_current_round(task_id)

        event = BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id=self.id,
            task_id=task_id,
            data={
                "opinion": opinion,
                "score": score,
                "is_executor": self._is_executor,
                "confidence": self._confidence
            },
            round_number=round_number
        )

        await self.blackboard.append(event)

    async def publish_score(
        self,
        task_id: str,
        target_expert_id: str,
        score: float,
        rationale: str = ""
    ):
        """
        Publish score for another expert's opinion.

        Args:
            task_id: Task ID
            target_expert_id: Expert being scored
            score: Score (0.0-1.0)
            rationale: Scoring rationale
        """
        from ..memory.events import BlackboardEvent, EventType

        event = BlackboardEvent(
            type=EventType.SCORE_SUBMITTED,
            expert_id=self.id,
            task_id=task_id,
            data={
                "target_expert": target_expert_id,
                "score": score,
                "rationale": rationale
            },
            round_number=self.blackboard.get_current_round(task_id)
        )

        await self.blackboard.append(event)

    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        Call LLM through model router.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        # Build enhanced prompt with expert persona
        enhanced_prompt = self._build_enhanced_prompt(prompt)

        response = await self.model_router.route_request(
            self._get_task_type(),
            enhanced_prompt,
        )

        return response.content if hasattr(response, 'content') else str(response)

    def _build_enhanced_prompt(self, prompt: str) -> str:
        """
        Build enhanced prompt with expert persona.

        Args:
            prompt: Original prompt

        Returns:
            Enhanced prompt
        """
        persona = self.config.persona

        prefix = f"""You are {persona.description} with background: {persona.background}.
Your personality: {persona.personality}.

Think using this framework:
{chr(10).join('- ' + t for t in self.config.thinking_framework)}

Your strengths:
{chr(10).join('- ' + s for s in self.config.strengths)}

Now, please address the following:"""

        return f"{prefix}\n\n{prompt}"

    def _get_task_type(self):
        """Get appropriate task type for model routing."""
        from ..services.router import TaskType

        # Default to text reasoning
        return TaskType.TEXT_REASONING

    def record_action(self, action: str, result: Any):
        """
        Record action in task history.

        Args:
            action: Action description
            result: Action result
        """
        self._task_history.append({
            "action": action,
            "result": result,
            "expert_id": self.id
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get task history.

        Returns:
            List of recorded actions
        """
        return self._task_history.copy()

    def clear_history(self):
        """Clear task history."""
        self._task_history.clear()

    def __repr__(self) -> str:
        """String representation."""
        mode = "executor" if self._is_executor else "critic"
        confidence = f"{self._confidence:.2f}" if self._confidence else "N/A"
        return f"<Expert {self.id}: {self.role} ({mode}, confidence: {confidence})>"
