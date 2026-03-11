"""Memory compression using LLM summarization."""

from typing import List, Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Result of memory compression."""
    summary: str
    key_points: List[str]
    original_size: int
    compressed_size: int
    compression_ratio: float


class MemoryCompressor:
    """
    Compress memory entries using LLM summarization.

    Can use either LLM-based intelligent compression
    or simple fallback summarization.
    """

    def __init__(self, llm_service: Optional[Any] = None):
        """
        Initialize compressor.

        Args:
            llm_service: Optional LLM service for intelligent compression
        """
        self.llm_service = llm_service
        self._compression_history: List[CompressionResult] = []

    async def compress(
        self,
        content: str,
        target_length: Optional[int] = None,
        compression_ratio: float = 0.3
    ) -> CompressionResult:
        """
        Compress content to shorter form.

        Args:
            content: Content to compress
            target_length: Target length in characters (optional)
            compression_ratio: Target ratio if target_length not specified

        Returns:
            CompressionResult with summary and stats
        """
        original_size = len(content)

        if self.llm_service and hasattr(self.llm_service, 'generate_text'):
            summary = await self._llm_compress(content, target_length)
        else:
            summary = self._simple_compress(content, compression_ratio)

        # Extract key points
        key_points = self._extract_key_points(summary)

        result = CompressionResult(
            summary=summary,
            key_points=key_points,
            original_size=original_size,
            compressed_size=len(summary),
            compression_ratio=len(summary) / original_size if original_size > 0 else 1.0
        )

        self._compression_history.append(result)
        return result

    async def _llm_compress(self, content: str, target_length: Optional[int]) -> str:
        """
        Compress using LLM.

        Args:
            content: Content to compress
            target_length: Target length

        Returns:
            Compressed summary
        """
        if target_length:
            instruction = f"Summarize this content to approximately {target_length} characters"
        else:
            instruction = "Summarize this content concisely, preserving all key information"

        prompt = f"""{instruction}. Focus on:
1. Key decisions and conclusions
2. Important facts and data
3. Action items and outcomes

Content to summarize:
{content}

Summary:"""

        try:
            response = await self.llm_service.generate_text(
                prompt,
                system_prompt="You are an expert at summarizing technical discussions. "
                             "Extract key information while being concise.",
                max_tokens=1000
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            # Fallback to simple compression
            return self._simple_compress(content)

    def _simple_compress(self, content: str, ratio: float = 0.3) -> str:
        """
        Simple compression by extracting key sentences.

        Args:
            content: Content to compress
            ratio: Target compression ratio

        Returns:
            Compressed content
        """
        sentences = content.split('.')
        if len(sentences) <= 3:
            return content

        # Keep first and last sentences, plus some middle ones
        keep_count = max(3, int(len(sentences) * ratio))

        selected = [sentences[0]]  # First sentence
        if len(sentences) > 2:
            # Add middle sentences
            step = len(sentences) // keep_count
            for i in range(1, min(keep_count - 1, len(sentences) - 1)):
                idx = i * step
                if idx < len(sentences) - 1:
                    selected.append(sentences[idx])
        selected.append(sentences[-1])  # Last sentence

        return '. '.join(selected) + '.'

    def _extract_key_points(self, summary: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from summary.

        Args:
            summary: Summary text
            max_points: Maximum number of points

        Returns:
            List of key points
        """
        # Simple extraction: split by newlines and periods
        lines = [line.strip() for line in summary.split('\n') if line.strip()]

        # Also split long sentences
        points = []
        for line in lines:
            if '.' in line:
                parts = [p.strip() for p in line.split('.') if p.strip()]
                points.extend(parts)
            else:
                points.append(line)

        # Filter and limit
        points = [p.rstrip('.') for p in points if len(p) > 10]
        return points[:max_points]

    async def compress_events(
        self,
        events: List[Dict[str, Any]],
        compression_ratio: float = 0.3
    ) -> str:
        """
        Compress a list of events to summary.

        Args:
            events: List of event dictionaries
            compression_ratio: Target compression ratio

        Returns:
            Summary string
        """
        if not events:
            return ""

        # Build text representation
        lines = []
        for event in events[-50:]:  # Last 50 events max
            event_type = event.get('type', 'unknown')
            expert_id = event.get('expert_id', 'unknown')
            data = event.get('data', {})
            lines.append(f"[{event_type}] {expert_id}: {data}")

        content = "\n".join(lines)

        if self.llm_service:
            result = await self.compress(content, compression_ratio=compression_ratio)
            return result.summary
        else:
            return self._simple_compress(content, compression_ratio)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Dictionary with stats
        """
        if not self._compression_history:
            return {"total_compressions": 0}

        total_original = sum(r.original_size for r in self._compression_history)
        total_compressed = sum(r.compressed_size for r in self._compression_history)

        return {
            "total_compressions": len(self._compression_history),
            "average_compression_ratio": sum(r.compression_ratio for r in self._compression_history) / len(self._compression_history),
            "total_saved_bytes": total_original - total_compressed,
            "average_original_size": total_original / len(self._compression_history),
            "average_compressed_size": total_compressed / len(self._compression_history)
        }
