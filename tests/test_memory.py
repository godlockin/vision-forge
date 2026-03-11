"""Tests for blackboard and memory systems."""

import pytest
import asyncio
from pathlib import Path

from vision_forge.memory.blackboard import SharedBlackboard
from vision_forge.memory.events import BlackboardEvent, EventType
from vision_forge.memory.manager import MemoryManager


class TestBlackboardEvent:
    """Test BlackboardEvent dataclass."""

    def test_event_creation(self):
        event = BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id="test_expert",
            task_id="task_001",
            data={"opinion": "Test opinion"}
        )

        assert event.expert_id == "test_expert"
        assert event.task_id == "task_001"
        assert event.type == EventType.OPINION_ADDED

    def test_event_to_dict(self):
        event = BlackboardEvent(
            type=EventType.SCORE_SUBMITTED,
            expert_id="expert_1",
            task_id="task_001",
            data={"score": 0.8}
        )

        d = event.to_dict()
        assert d["type"] == "score_submitted"
        assert d["expert_id"] == "expert_1"
        assert d["data"] == {"score": 0.8}

    def test_event_from_dict(self):
        data = {
            "id": "test-id",
            "type": "opinion_added",
            "expert_id": "expert_1",
            "task_id": "task_001",
            "data": {"opinion": "Test"},
            "timestamp": 1234567890,
            "round_number": 1
        }

        event = BlackboardEvent.from_dict(data)
        assert event.id == "test-id"
        assert event.type == EventType.OPINION_ADDED
        assert event.round_number == 1


class TestSharedBlackboard:
    """Test SharedBlackboard class."""

    @pytest.mark.asyncio
    async def test_append_event(self, tmp_path):
        board = SharedBlackboard(persist_dir=str(tmp_path))

        event = BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id="test_expert",
            task_id="task_1",
            data={"opinion": "Test opinion"}
        )

        result = await board.append(event)
        assert result is True

        events = board.get_events(task_id="task_1")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_deduplication(self, tmp_path):
        board = SharedBlackboard(persist_dir=str(tmp_path))

        event = BlackboardEvent(
            type=EventType.SCORE_SUBMITTED,
            expert_id="test_expert",
            task_id="task_1",
            data={"score": 0.8}
        )

        # First append should succeed
        assert await board.append(event)

        # Second append (duplicate) should fail within dedup window
        assert not await board.append(event)

    @pytest.mark.asyncio
    async def test_query_filters(self, tmp_path):
        board = SharedBlackboard(persist_dir=str(tmp_path))

        # Add multiple events
        for i in range(5):
            await board.append(BlackboardEvent(
                type=EventType.OPINION_ADDED,
                expert_id=f"expert_{i}",
                task_id="task_1",
                round_number=i % 2,
                data={"round": i}
            ))

        # Filter by round 0 (events 0, 2, 4)
        events = board.get_events(round_number=0)
        assert len(events) == 3  # 0, 2, 4

        # Filter by round 1 (events 1, 3)
        events = board.get_events(round_number=1)
        assert len(events) == 2  # 1, 3

        # Filter by specific expert
        events = board.get_events(expert_id="expert_0")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_get_current_round(self, tmp_path):
        board = SharedBlackboard(persist_dir=str(tmp_path))

        await board.append(BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id="expert_1",
            task_id="task_1",
            round_number=0
        ))

        await board.append(BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id="expert_1",
            task_id="task_1",
            round_number=1
        ))

        assert board.get_current_round("task_1") == 1

    @pytest.mark.asyncio
    async def test_clear_task(self, tmp_path):
        board = SharedBlackboard(persist_dir=str(tmp_path))

        await board.append(BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id="expert_1",
            task_id="task_1"
        ))

        await board.append(BlackboardEvent(
            type=EventType.OPINION_ADDED,
            expert_id="expert_1",
            task_id="task_2"
        ))

        board.clear("task_1")

        events = board.get_events()
        assert len(events) == 1
        assert events[0].task_id == "task_2"

    def test_get_stats(self, tmp_path):
        board = SharedBlackboard(persist_dir=str(tmp_path))
        stats = board.get_stats()

        assert "total_events" in stats
        assert "unique_tasks" in stats
        assert "event_types" in stats


class TestMemoryManager:
    """Test MemoryManager class."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_session(self):
        manager = MemoryManager()

        await manager.store("key1", {"data": "value"}, scope="session")
        result = await manager.retrieve("key1", scope="session")

        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_store_and_retrieve_task(self):
        manager = MemoryManager()

        await manager.store("task_key", "value1", scope="task", task_id="task_1")
        await manager.store("task_key", "value2", scope="task", task_id="task_2")

        result1 = await manager.retrieve("task_key", scope="task", task_id="task_1")
        result2 = await manager.retrieve("task_key", scope="task", task_id="task_2")

        assert result1 == "value1"
        assert result2 == "value2"

    @pytest.mark.asyncio
    async def test_clear_task_memory(self):
        manager = MemoryManager()

        await manager.store("task_key", "value", scope="task", task_id="task_1")
        await manager.clear_task_memory("task_1")

        result = await manager.retrieve("task_key", scope="task", task_id="task_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        manager = MemoryManager()

        await manager.store("key1", "value", scope="session")
        result = await manager.delete("key1", scope="session")

        assert result is True

        retrieved = await manager.retrieve("key1", scope="session")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_search(self):
        manager = MemoryManager()

        await manager.store("user_preference", "dark_theme", scope="session")
        await manager.store("user_name", "Alice", scope="session")

        results = manager.search("user", scope="session")
        assert len(results) == 2

    def test_get_stats(self):
        manager = MemoryManager()
        stats = manager.get_stats()

        assert "task_memory_entries" in stats
        assert "session_memory_entries" in stats
        assert "persistent_memory_entries" in stats
