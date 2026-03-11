"""Knowledge Administrator expert implementation.

Manages shared blackboard, context compression, and knowledge indexing.
"""

import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from ..experts.expert import Expert
from ..core.models import ExpertConfig
from ..memory.blackboard import SharedBlackboard, BlackboardEvent
from ..memory.events import EventType
from ..services.router import ModelRouter, TaskType
from ..memory.manager import MemoryManager


class KnowledgeAdmin(Expert):
    """
    Knowledge Administrator for blackboard management and memory compression.

    Responsibilities:
    - Context compression and summarization
    - Knowledge indexing and retrieval
    - Memory consolidation
    - Blackboard maintenance (cleanup, archiving)

    This is a critical expert that ensures the system doesn't get overwhelmed
    by context. It uses LLM-based summarization for intelligent compression.
    """

    # Default configuration for compression
    DEFAULT_COMPRESSION_THRESHOLD = 1000  # tokens
    DEFAULT_RETENTION_RATE = 0.3  # Keep 30% after compression
    MAX_KEY_POINTS = 10

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Initialize Knowledge Administrator.

        Args:
            config: Expert configuration from YAML
            blackboard: Shared blackboard instance
            model_router: Model router for LLM access
            memory_manager: Optional memory manager for persistence
        """
        super().__init__(config, blackboard, model_router)
        self.memory_manager = memory_manager or MemoryManager()

        # Index structures for knowledge retrieval
        self._knowledge_index: Dict[str, Dict[str, Any]] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of knowledge_ids

        # Compression statistics
        self._compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "last_compression_at": 0
        }

    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process knowledge management task.

        Args:
            task_data: Task data including:
                - task_id: Task identifier
                - action: Action to perform (compress/index/cleanup)
                - content: Content to process (optional)

        Returns:
            Processing result with status and details
        """
        task_id = task_data.get("task_id", "unknown")
        action = task_data.get("action", "summarize")

        if action == "compress":
            events = task_data.get("events", [])
            max_tokens = task_data.get("max_tokens", self.DEFAULT_COMPRESSION_THRESHOLD)
            summary = await self.compress_context(events, max_tokens)
            return {
                "status": "compressed",
                "summary": summary,
                "task_id": task_id
            }

        elif action == "summarize":
            summary = await self.generate_summary(task_id)
            return {
                "status": "summarized",
                "summary": summary,
                "task_id": task_id
            }

        elif action == "index":
            content = task_data.get("content", "")
            tags = task_data.get("tags", [])
            result = await self.index_knowledge(content, tags)
            return {
                "status": "indexed",
                "knowledge_id": result.get("id"),
                "task_id": task_id
            }

        elif action == "retrieve":
            query = task_data.get("query", "")
            tags = task_data.get("tags")
            results = await self.retrieve_knowledge(query, tags)
            return {
                "status": "retrieved",
                "results": results,
                "count": len(results),
                "task_id": task_id
            }

        elif action == "cleanup":
            retain_important = task_data.get("retain_important", True)
            result = await self.cleanup_blackboard(task_id, retain_important)
            return {
                "status": "cleaned",
                "events_removed": result.get("events_removed", 0),
                "task_id": task_id
            }

        elif action == "consolidate":
            result = await self.consolidate_memories()
            return {
                "status": "consolidated",
                "memories_consolidated": result.get("count", 0),
                "task_id": task_id
            }

        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "task_id": task_id
            }

    async def compress_context(
        self,
        events: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> str:
        """
        Summarize blackboard events into condensed context.

        Uses LLM to intelligently compress events while preserving
        key information, decisions, and rationale.

        Args:
            events: List of blackboard events to compress
            max_tokens: Maximum tokens for compressed output

        Returns:
            Compressed summary string
        """
        if not events:
            return ""

        # Convert events to text format
        events_text = self._events_to_text(events)

        # Build compression prompt
        prompt = f"""Summarize the following blackboard events into a concise summary.
Focus on preserving:
1. Key decisions made
2. Important opinions and their rationale
3. Final outcomes and conclusions
4. Critical task information

Remove:
- Redundant information
- Minor discussion details
- Repetitive scores

Maximum tokens: {max_tokens}

Events:
{events_text}

Provide a clear, structured summary:"""

        # Use LLM for compression
        try:
            response = await self.call_llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for more focused summary
            )
            summary = response.strip()

            # Update statistics
            original_tokens = self._estimate_tokens(events_text)
            compressed_tokens = self._estimate_tokens(summary)
            self._compression_stats["total_compressions"] += 1
            self._compression_stats["total_tokens_saved"] += (
                original_tokens - compressed_tokens
            )
            self._compression_stats["last_compression_at"] = int(
                datetime.now().timestamp() * 1000
            )

            return summary

        except Exception as e:
            # Fallback: simple truncation
            return self._simple_truncate(events_text, max_tokens)

    async def generate_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Create task summary from blackboard events.

        Args:
            task_id: Task ID to summarize

        Returns:
            Summary dictionary with:
                - task_id: Task identifier
                - summary: Text summary
                - key_points: List of key points
                - decisions: List of decisions made
                - experts_involved: List of expert IDs
        """
        # Get all events for this task
        events = self.blackboard.get_events(task_id=task_id)

        if not events:
            return {
                "task_id": task_id,
                "summary": "No events found for this task.",
                "key_points": [],
                "decisions": [],
                "experts_involved": [],
                "status": "no_events"
            }

        # Extract key information
        experts_involved = list(set(e.expert_id for e in events if e.expert_id))
        opinions = [e.data.get("opinion", "") for e in events if e.data.get("opinion")]
        decisions = [
            e.data for e in events
            if e.type == EventType.DECISION_MADE
        ]

        # Build summary text
        events_text = self._events_to_text([e.to_dict() for e in events])

        # Generate structured summary using LLM
        prompt = f"""Analyze the following task events and create a structured summary.
Return your response in JSON format with these fields:
- summary: Brief overall summary (2-3 sentences)
- key_points: List of up to {self.MAX_KEY_POINTS} most important points
- decisions: List of key decisions made
- timeline: Chronological flow of major events

Events:
{events_text}

Respond with valid JSON only."""

        try:
            response = await self.call_llm(prompt, max_tokens=1500, temperature=0.3)

            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                summary_data = json.loads(json_match.group())
            else:
                summary_data = {
                    "summary": response,
                    "key_points": [],
                    "decisions": [],
                    "timeline": []
                }

            return {
                "task_id": task_id,
                "summary": summary_data.get("summary", ""),
                "key_points": summary_data.get("key_points", []),
                "decisions": summary_data.get("decisions", decisions),
                "experts_involved": experts_involved,
                "timeline": summary_data.get("timeline", []),
                "status": "success"
            }

        except (json.JSONDecodeError, Exception) as e:
            # Fallback to basic summary
            return {
                "task_id": task_id,
                "summary": f"Task involved {len(events)} events from {len(experts_involved)} experts.",
                "key_points": opinions[:self.MAX_KEY_POINTS],
                "decisions": decisions,
                "experts_involved": experts_involved,
                "status": "fallback",
                "error": str(e)
            }

    async def index_knowledge(
        self,
        content: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Index knowledge for future retrieval.

        Creates searchable index entries with tags and embeddings.

        Args:
            content: Knowledge content to index
            tags: Optional list of tags for categorization

        Returns:
            Index result with:
                - id: Generated knowledge ID
                - status: Indexing status
                - tags: Applied tags
        """
        # Generate unique ID based on content
        knowledge_id = hashlib.sha256(
            f"{content}{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        # Generate embedding for semantic search
        embedding = await self._generate_embedding(content)

        # Extract key concepts using LLM
        key_concepts = await self._extract_key_concepts(content)

        # Create index entry
        index_entry = {
            "id": knowledge_id,
            "content": content,
            "tags": tags or [],
            "key_concepts": key_concepts,
            "embedding": embedding,
            "created_at": int(datetime.now().timestamp() * 1000),
            "access_count": 0,
            "importance": 1.0  # Default importance
        }

        # Add to index
        self._knowledge_index[knowledge_id] = index_entry

        # Update tag index
        for tag in (tags or []):
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(knowledge_id)

        # Store in memory manager if available
        if self.memory_manager:
            await self.memory_manager.store(
                key=f"knowledge:{knowledge_id}",
                value=index_entry,
                scope="persistent",
                importance=0.8
            )

        return {
            "id": knowledge_id,
            "status": "indexed",
            "tags": tags or [],
            "key_concepts": key_concepts,
            "indexed_at": index_entry["created_at"]
        }

    async def retrieve_knowledge(
        self,
        query: str,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge based on query.

        Uses semantic similarity and tag matching for retrieval.

        Args:
            query: Search query
            tags: Optional tags to filter by

        Returns:
            List of relevant knowledge entries sorted by relevance
        """
        if not self._knowledge_index:
            return []

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Filter by tags if provided
        candidate_ids: Set[str] = set(self._knowledge_index.keys())
        if tags:
            tagged_ids = set()
            for tag in tags:
                tagged_ids.update(self._tag_index.get(tag, set()))
            candidate_ids = candidate_ids.intersection(tagged_ids)

        if not candidate_ids:
            return []

        # Calculate similarity scores
        results = []
        for knowledge_id in candidate_ids:
            entry = self._knowledge_index[knowledge_id]

            # Calculate cosine similarity
            similarity = self._cosine_similarity(
                query_embedding,
                entry.get("embedding", [0] * 1536)
            )

            # Boost score for tag matches
            if tags:
                tag_matches = len(set(entry.get("tags", [])) & set(tags))
                similarity += tag_matches * 0.1

            # Update access count
            entry["access_count"] += 1

            results.append({
                "id": knowledge_id,
                "content": entry["content"],
                "tags": entry.get("tags", []),
                "key_concepts": entry.get("key_concepts", []),
                "similarity": similarity,
                "access_count": entry["access_count"]
            })

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:10]  # Return top 10 results

    async def cleanup_blackboard(
        self,
        task_id: str,
        retain_important: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up completed task events from blackboard.

        Args:
            task_id: Task ID to clean up
            retain_important: Whether to retain important events

        Returns:
            Cleanup result with:
                - events_removed: Number of events removed
                - events_retained: Number of events retained
                - summary: Summary of cleaned up task
        """
        # Get all events for this task
        events = self.blackboard.get_events(task_id=task_id)

        if not events:
            return {
                "events_removed": 0,
                "events_retained": 0,
                "summary": "No events to clean up"
            }

        # Generate summary before cleanup
        summary = await self.generate_summary(task_id)

        # Determine which events to retain
        events_to_retain: List[BlackboardEvent] = []

        if retain_important:
            # Always retain these event types
            important_types = {
                EventType.DECISION_MADE,
                EventType.TASK_COMPLETED,
                EventType.VETO_TRIGGERED
            }

            for event in events:
                if event.type in important_types:
                    events_to_retain.append(event)

            # Also retain high-score opinions
            for event in events:
                if (event.type == EventType.OPINION_ADDED and
                        event.data.get("score", 0) > 0.8):
                    if event not in events_to_retain:
                        events_to_retain.append(event)

        # Calculate removal stats
        events_removed = len(events) - len(events_to_retain)

        # Clear blackboard for this task
        self.blackboard.clear(task_id)

        # Re-add retained events
        for event in events_to_retain:
            event.task_id = f"archived_{task_id}"
            await self.blackboard.append(event)

        # Store summary in memory
        if self.memory_manager:
            await self.memory_manager.store(
                key=f"task_summary:{task_id}",
                value=summary,
                scope="persistent",
                importance=0.9
            )

        return {
            "events_removed": events_removed,
            "events_retained": len(events_to_retain),
            "summary": summary,
            "task_id": task_id
        }

    async def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolidate session memories into persistent memory.

        Reviews session memories and moves important ones to persistent storage.

        Returns:
            Consolidation result with:
                - count: Number of memories consolidated
                - details: List of consolidated memories
        """
        if not self.memory_manager:
            return {
                "count": 0,
                "details": [],
                "status": "no_memory_manager"
            }

        # Get session memory stats
        stats = self.memory_manager.get_stats()
        session_count = stats.get("session_memory_entries", 0)

        if session_count == 0:
            return {
                "count": 0,
                "details": [],
                "status": "no_session_memories"
            }

        # Use LLM to identify important memories
        session_data = await self.memory_manager.export_to_dict()
        session_memories = session_data.get("session_memory", {})

        if not session_memories:
            return {
                "count": 0,
                "details": [],
                "status": "no_session_memories"
            }

        # Build memory review prompt
        memories_text = "\n".join(
            f"- {k}: {str(v)[:200]}"
            for k, v in list(session_memories.items())[:20]
        )

        prompt = f"""Review these session memories and identify which ones should be
moved to persistent storage. Consider:
- Task outcomes and decisions
- User preferences discovered
- Important patterns or learnings
- Knowledge that would be useful for future tasks

Memories:
{memories_text}

Return a JSON list of memory keys that should be persisted, with a brief reason for each.
Format: [{{"key": "memory_key", "reason": "why it should be persisted"}}]"""

        try:
            response = await self.call_llm(prompt, max_tokens=1000, temperature=0.3)

            # Parse response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                consolidation_plan = json.loads(json_match.group())
            else:
                # Fallback: consolidate all
                consolidation_plan = [
                    {"key": k, "reason": "Session memory"}
                    for k in list(session_memories.keys())[:10]
                ]

            # Execute consolidation
            consolidated = []
            for item in consolidation_plan:
                key = item.get("key", "")
                value = session_memories.get(key)

                if value is not None:
                    # Store in persistent memory
                    await self.memory_manager.store(
                        key=key,
                        value=value,
                        scope="persistent",
                        importance=0.8
                    )

                    # Remove from session memory
                    await self.memory_manager.delete(key, scope="session")

                    consolidated.append({
                        "key": key,
                        "reason": item.get("reason", "")
                    })

            return {
                "count": len(consolidated),
                "details": consolidated,
                "status": "consolidated"
            }

        except Exception as e:
            # Fallback: use built-in compression
            await self.memory_manager.compress_session_memory(target_entries=50)
            return {
                "count": 0,
                "details": [],
                "status": "fallback_compression",
                "error": str(e)
            }

    def _get_task_type(self) -> TaskType:
        """
        Get task type for model routing.

        Knowledge admin tasks are text reasoning/summary tasks.

        Returns:
            TaskType.SUMMARY for compression and summarization tasks
        """
        return TaskType.SUMMARY

    # ==================== Helper Methods ====================

    def _events_to_text(self, events: List[Dict[str, Any]]) -> str:
        """
        Convert events to text format for LLM processing.

        Args:
            events: List of event dictionaries

        Returns:
            Formatted text string
        """
        lines = []
        for event in events:
            event_type = event.get("type", "unknown")
            expert_id = event.get("expert_id", "unknown")
            timestamp = event.get("timestamp", 0)
            data = event.get("data", {})

            lines.append(f"[{event_type}] Expert: {expert_id}")

            if isinstance(data, dict):
                for key, value in data.items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  data: {data}")

            lines.append("")

        return "\n".join(lines)

    def _simple_truncate(self, text: str, max_tokens: int) -> str:
        """
        Simple truncation fallback.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        # Rough token estimation: 1 token ~= 4 characters
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate and add indicator
        return text[:max_chars - 100] + "\n... [truncated]"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ~= 4 characters for English
        return len(text) // 4

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Note: This is a placeholder. In production, use a real
        embedding model like text-embedding-3-small.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (placeholder)
        """
        # Placeholder: hash-based pseudo-embedding
        # In production, replace with actual embedding API call
        hash_value = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to 1536-dimensional vector (OpenAI embedding size)
        embedding = []
        for i in range(0, len(hash_value), 2):
            byte_value = int(hash_value[i:i+2], 16)
            # Normalize to [-1, 1] range
            normalized = (byte_value - 128) / 128.0
            embedding.extend([normalized] * 96)  # Repeat to fill dimensions

        # Ensure exactly 1536 dimensions
        while len(embedding) < 1536:
            embedding.append(0.0)

        return embedding[:1536]

    async def _extract_key_concepts(self, content: str) -> List[str]:
        """
        Extract key concepts from content using LLM.

        Args:
            content: Content to analyze

        Returns:
            List of key concepts
        """
        prompt = f"""Extract the key concepts from the following content.
Return up to 10 concepts as a JSON array of strings.
Focus on nouns and noun phrases that capture the main topics.

Content:
{content[:2000]}

Respond with JSON array only:"""

        try:
            response = await self.call_llm(prompt, max_tokens=200, temperature=0.3)

            # Parse JSON response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                concepts = json.loads(json_match.group())
                if isinstance(concepts, list):
                    return concepts[:self.MAX_KEY_POINTS]
        except Exception:
            pass

        # Fallback: return empty list
        return []

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Dictionary with compression stats
        """
        return {
            **self._compression_stats,
            "knowledge_index_size": len(self._knowledge_index),
            "tag_index_size": len(self._tag_index)
        }

    def get_knowledge_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current knowledge index.

        Returns:
            Copy of knowledge index
        """
        return self._knowledge_index.copy()

    def clear_knowledge_index(self):
        """Clear the knowledge index."""
        self._knowledge_index.clear()
        self._tag_index.clear()
