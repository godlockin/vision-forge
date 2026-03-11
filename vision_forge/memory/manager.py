"""Memory manager for three-layer memory system."""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """Single memory entry."""
    key: str
    value: Any
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    access_count: int = 0
    importance: float = 1.0
    scope: str = "session"
    task_id: Optional[str] = None

    def touch(self):
        """Update access count and timestamp."""
        self.access_count += 1
        self.updated_at = int(time.time())


class MemoryManager:
    """
    Three-layer memory manager.

    Layers:
    - Task memory: Cleared after task completion
    - Session memory: Cleared after session ends
    - Persistent memory: Compressed and stored permanently
    """

    def __init__(self, compression_threshold: int = 1000):
        """
        Initialize memory manager.

        Args:
            compression_threshold: Number of entries before compression triggers
        """
        # Task memory (cleared after task)
        self.task_memory: Dict[str, Dict[str, MemoryEntry]] = {}

        # Session memory (cleared after session)
        self.session_memory: Dict[str, MemoryEntry] = {}

        # Persistent memory (compressed, permanent)
        self.persistent_memory: Dict[str, MemoryEntry] = {}

        self.compression_threshold = compression_threshold
        self._lock = asyncio.Lock()
        self._compression_in_progress = False

    async def store(
        self,
        key: str,
        value: Any,
        scope: str = "session",
        task_id: Optional[str] = None,
        importance: float = 1.0
    ):
        """
        Store memory entry.

        Args:
            key: Memory key
            value: Memory value
            scope: Memory scope (task/session/persistent)
            task_id: Task ID for task-scoped memory
            importance: Importance score (0.0-1.0)
        """
        async with self._lock:
            entry = MemoryEntry(
                key=key,
                value=value,
                importance=importance,
                scope=scope,
                task_id=task_id
            )

            if scope == "task":
                if task_id not in self.task_memory:
                    self.task_memory[task_id] = {}
                self.task_memory[task_id][key] = entry
            elif scope == "session":
                self.session_memory[key] = entry
            elif scope == "persistent":
                self.persistent_memory[key] = entry

        # Check if compression needed
        if len(self.session_memory) > self.compression_threshold:
            if not self._compression_in_progress:
                asyncio.create_task(self.compress_session_memory())

    async def retrieve(
        self,
        key: str,
        scope: str = "session",
        task_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve memory entry.

        Args:
            key: Memory key
            scope: Memory scope
            task_id: Task ID for task-scoped memory

        Returns:
            Memory value or None if not found
        """
        async with self._lock:
            if scope == "task":
                if task_id and task_id in self.task_memory:
                    entry = self.task_memory[task_id].get(key)
                    if entry:
                        entry.touch()
                        return entry.value
            elif scope == "session":
                entry = self.session_memory.get(key)
                if entry:
                    entry.touch()
                    return entry.value
            elif scope == "persistent":
                entry = self.persistent_memory.get(key)
                if entry:
                    entry.touch()
                    return entry.value

        return None

    async def delete(
        self,
        key: str,
        scope: str = "session",
        task_id: Optional[str] = None
    ) -> bool:
        """
        Delete memory entry.

        Args:
            key: Memory key
            scope: Memory scope
            task_id: Task ID for task-scoped memory

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if scope == "task":
                if task_id and task_id in self.task_memory:
                    if key in self.task_memory[task_id]:
                        del self.task_memory[task_id][key]
                        return True
            elif scope == "session":
                if key in self.session_memory:
                    del self.session_memory[key]
                    return True
            elif scope == "persistent":
                if key in self.persistent_memory:
                    del self.persistent_memory[key]
                    return True

        return False

    async def clear_task_memory(self, task_id: str):
        """
        Clear memory for completed task.

        Args:
            task_id: Task ID to clear
        """
        async with self._lock:
            if task_id in self.task_memory:
                del self.task_memory[task_id]

    async def clear_session_memory(self):
        """Clear all session memory."""
        async with self._lock:
            self.session_memory.clear()

    async def compress_session_memory(self, target_entries: int = 100):
        """
        Compress session memory to persistent storage.

        Args:
            target_entries: Target number of entries to keep in session
        """
        self._compression_in_progress = True

        try:
            async with self._lock:
                # Sort by importance and access count
                sorted_entries = sorted(
                    self.session_memory.items(),
                    key=lambda x: (x[1].importance, x[1].access_count),
                    reverse=True
                )

                # Move top entries to persistent
                for key, entry in sorted_entries[:target_entries]:
                    if entry.importance > 0.7:
                        self.persistent_memory[key] = entry
                        del self.session_memory[key]

        finally:
            self._compression_in_progress = False

    def search(self, pattern: str, scope: str = "session") -> List[Dict[str, Any]]:
        """
        Search memory by key pattern.

        Args:
            pattern: Search pattern (substring match)
            scope: Memory scope to search

        Returns:
            List of matching entries with their values
        """
        results = []

        if scope == "task":
            for task_entries in self.task_memory.values():
                for key, entry in task_entries.items():
                    if pattern.lower() in key.lower():
                        results.append({
                            "key": key,
                            "value": entry.value,
                            "scope": "task",
                            "importance": entry.importance
                        })
        elif scope == "session":
            for key, entry in self.session_memory.items():
                if pattern.lower() in key.lower():
                    results.append({
                        "key": key,
                        "value": entry.value,
                        "scope": "session",
                        "importance": entry.importance
                    })
        elif scope == "persistent":
            for key, entry in self.persistent_memory.items():
                if pattern.lower() in key.lower():
                    results.append({
                        "key": key,
                        "value": entry.value,
                        "scope": "persistent",
                        "importance": entry.importance
                    })

        return results

    def get_stats(self) -> Dict[str, int]:
        """
        Get memory statistics.

        Returns:
            Dictionary with entry counts per layer
        """
        return {
            "task_memory_entries": sum(len(m) for m in self.task_memory.values()),
            "task_memory_tasks": len(self.task_memory),
            "session_memory_entries": len(self.session_memory),
            "persistent_memory_entries": len(self.persistent_memory),
            "total_entries": (
                sum(len(m) for m in self.task_memory.values()) +
                len(self.session_memory) +
                len(self.persistent_memory)
            )
        }

    async def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all memory to dictionary.

        Returns:
            Dictionary representation of all memories
        """
        async with self._lock:
            return {
                "task_memory": {
                    task_id: {k: v.value for k, v in entries.items()}
                    for task_id, entries in self.task_memory.items()
                },
                "session_memory": {k: v.value for k, v in self.session_memory.items()},
                "persistent_memory": {k: v.value for k, v in self.persistent_memory.items()}
            }
