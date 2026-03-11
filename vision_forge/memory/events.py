"""Blackboard event types for expert system communication."""

from dataclasses import dataclass, field
from typing import Any, Dict
from enum import Enum
import time
import uuid


class EventType(str, Enum):
    """Type of blackboard events."""
    OPINION_ADDED = "opinion_added"
    SCORE_SUBMITTED = "score_submitted"
    DECISION_MADE = "decision_made"
    ROUND_CLOSED = "round_closed"
    VETO_TRIGGERED = "veto_triggered"
    SNAPSHOT_COMPRESSED = "snapshot_compressed"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_STARTED = "task_started"
    EXPERT_JOINED = "expert_joined"


@dataclass
class BlackboardEvent:
    """
    Event representing an action on the blackboard.

    Uses append-only event sourcing pattern.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.OPINION_ADDED
    expert_id: str = ""
    task_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    round_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "expert_id": self.expert_id,
            "task_id": self.task_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "round_number": self.round_number
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackboardEvent":
        """Create event from dictionary."""
        return cls(
            id=data["id"],
            type=EventType(data["type"]),
            expert_id=data["expert_id"],
            task_id=data["task_id"],
            data=data["data"],
            timestamp=data["timestamp"],
            round_number=data.get("round_number", 0)
        )

    def __hash__(self) -> int:
        """Hash for deduplication."""
        content = f"{self.type}:{self.expert_id}:{self.task_id}:{str(self.data)}"
        return hash(content)
