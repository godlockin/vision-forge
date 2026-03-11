"""Shared blackboard for expert system communication.

Implements append-only event sourcing pattern for traceability.
"""

import asyncio
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .events import BlackboardEvent, EventType

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


class SharedBlackboard:
    """
    Append-only event sourcing blackboard for expert communication.

    All experts can write to and read from the blackboard.
    Events are persisted asynchronously to disk.
    """

    def __init__(self, persist_dir: str = "output/blackboard"):
        """
        Initialize blackboard.

        Args:
            persist_dir: Directory for persisting events
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._events: List[BlackboardEvent] = []
        self._lock = asyncio.Lock()
        self._subscribers: List[Callable[[BlackboardEvent], None]] = []

        # Deduplication window (5 seconds in milliseconds)
        self._dedup_window_ms = 5000
        self._event_hashes: Dict[str, int] = {}

        # Round tracking per task
        self._task_rounds: Dict[str, int] = {}

    async def append(self, event: BlackboardEvent) -> bool:
        """
        Append event to blackboard (idempotent within dedup window).

        Args:
            event: Event to append

        Returns:
            True if appended, False if duplicate
        """
        async with self._lock:
            # Check for duplicates within dedup window
            event_hash = self._hash_event(event)
            if event_hash in self._event_hashes:
                original_timestamp = self._event_hashes[event_hash]
                if event.timestamp - original_timestamp < self._dedup_window_ms:
                    return False  # Duplicate within window

            self._events.append(event)
            self._event_hashes[event_hash] = event.timestamp

            # Update round tracking
            if event.task_id:
                current_round = self._task_rounds.get(event.task_id, 0)
                if event.round_number > current_round:
                    self._task_rounds[event.task_id] = event.round_number

            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(event)
                except Exception:
                    pass

        # Async flush to disk (fire and forget)
        if HAS_AIOFILES:
            asyncio.create_task(self._flush_event(event))

        return True

    def subscribe(self, callback: Callable[[BlackboardEvent], None]):
        """
        Subscribe to new events.

        Args:
            callback: Function to call when new event arrives
        """
        self._subscribers.append(callback)

    def get_events(
        self,
        task_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        expert_id: Optional[str] = None,
        round_number: Optional[int] = None,
        since_timestamp: Optional[int] = None
    ) -> List[BlackboardEvent]:
        """
        Query events with filters.

        Args:
            task_id: Filter by task ID
            event_type: Filter by event type
            expert_id: Filter by expert ID
            round_number: Filter by round number
            since_timestamp: Filter by timestamp (milliseconds)

        Returns:
            List of matching events
        """
        result = self._events

        if task_id:
            result = [e for e in result if e.task_id == task_id]
        if event_type:
            result = [e for e in result if e.type == event_type]
        if expert_id:
            result = [e for e in result if e.expert_id == expert_id]
        if round_number is not None:
            result = [e for e in result if e.round_number == round_number]
        if since_timestamp:
            result = [e for e in result if e.timestamp >= since_timestamp]

        return result

    def get_current_round(self, task_id: str) -> int:
        """
        Get current round number for a task.

        Args:
            task_id: Task ID

        Returns:
            Current round number (max round_number from events for this task)
        """
        events = [e for e in self._events if e.task_id == task_id]
        if not events:
            return 0
        return max(e.round_number for e in events)

    def get_opinions(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all opinions for a task.

        Args:
            task_id: Task ID

        Returns:
            List of opinion data
        """
        events = self.get_events(task_id=task_id, event_type=EventType.OPINION_ADDED)
        return [e.data for e in events if "opinion" in e.data]

    def get_scores(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all scores for a task.

        Args:
            task_id: Task ID

        Returns:
            List of score data
        """
        events = self.get_events(task_id=task_id, event_type=EventType.SCORE_SUBMITTED)
        return [e.data for e in events if "score" in e.data]

    async def _flush_event(self, event: BlackboardEvent):
        """
        Persist event to disk.

        Args:
            event: Event to persist
        """
        filepath = self.persist_dir / f"{event.task_id or 'default'}.jsonl"

        try:
            async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to persist event: {e}")

    def _hash_event(self, event: BlackboardEvent) -> str:
        """
        Create hash for deduplication.

        Args:
            event: Event to hash

        Returns:
            MD5 hash of event content
        """
        # Include round_number in hash to avoid dedup events in different rounds
        content = f"{event.type}:{event.expert_id}:{event.task_id}:{event.round_number}:{json.dumps(event.data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def clear(self, task_id: str):
        """
        Clear events for a completed task.

        Args:
            task_id: Task ID to clear
        """
        self._events = [e for e in self._events if e.task_id != task_id]
        if task_id in self._task_rounds:
            del self._task_rounds[task_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get blackboard statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "total_events": len(self._events),
            "unique_tasks": len(set(e.task_id for e in self._events if e.task_id)),
            "rounds": dict(self._task_rounds),
            "event_types": {
                et.value: len([e for e in self._events if e.type == et])
                for et in EventType
            }
        }

    def load_from_file(self, filepath: str) -> int:
        """
        Load events from a file.

        Args:
            filepath: Path to JSONL file

        Returns:
            Number of events loaded
        """
        path = Path(filepath)
        if not path.exists():
            return 0

        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event = BlackboardEvent.from_dict(data)
                    self._events.append(event)
                    count += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return count
