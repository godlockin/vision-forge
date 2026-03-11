"""Main workflow orchestrator for the expert system."""

import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time

from ..core.models import Task, TaskStatus
from ..experts.expert_registry import ExpertRegistry
from ..memory.blackboard import SharedBlackboard
from ..memory.manager import MemoryManager
from ..services.router import ModelRouter, TaskType
from .fallback import AutoFallbackHandler


class WorkflowOrchestrator:
    """
    Main workflow orchestrator.

    Coordinates:
    - Expert selection and execution
    - Blackboard management
    - Compliance checks
    - Fallback handling
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        blackboard: SharedBlackboard,
        memory: MemoryManager,
        model_router: ModelRouter
    ):
        """
        Initialize orchestrator.

        Args:
            registry: Expert registry
            blackboard: Shared blackboard
            memory: Memory manager
            model_router: Model router
        """
        self.registry = registry
        self.blackboard = blackboard
        self.memory = memory
        self.model_router = model_router
        self.output_dir = Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process_request(
        self,
        user_prompt: str,
        images: Optional[List[bytes]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process complete user request.

        Args:
            user_prompt: User's prompt
            images: Optional input images
            context: Optional context data

        Returns:
            Processing result
        """
        task_id = self._generate_task_id()
        start_time = time.time()

        try:
            # Step 1: Compliance review (一票否决)
            compliance_result = await self._compliance_review(user_prompt)
            if not compliance_result["passed"]:
                return self._build_rejection_response(
                    task_id, compliance_result, time.time() - start_time
                )

            # Step 2: Initialize task
            task = self._create_task(task_id, user_prompt, images, context)

            # Step 3: Select experts
            selected_experts = await self._select_experts(task)

            # Step 4: Execute experts
            results = await self._execute_experts(task, selected_experts)

            # Step 5: Final compliance review
            final_review = await self._final_compliance_review(results)
            if not final_review["passed"]:
                # Try auto-fallback
                fallback_result = await self._attempt_fallback(
                    user_prompt, final_review
                )
                if not fallback_result["success"]:
                    return self._build_rejection_response(
                        task_id, final_review, time.time() - start_time
                    )
                # Use fallback result
                results["fallback_applied"] = True

            # Step 6: Save trace
            trace_path = await self._save_trace(task_id, results)

            # Step 7: Build success response
            return self._build_success_response(
                task_id, results, trace_path, time.time() - start_time
            )

        except Exception as e:
            return self._build_error_response(task_id, str(e), time.time() - start_time)

    async def _compliance_review(self, prompt: str) -> Dict[str, Any]:
        """
        Initial compliance review.

        Args:
            prompt: User prompt

        Returns:
            Compliance review result
        """
        from .experts.compliance import ComplianceExpert

        config = self.registry._expert_configs.get("compliance_legal_01")
        if not config:
            # No compliance expert configured, allow by default
            return {"passed": True, "violations": []}

        expert = ComplianceExpert(config, self.blackboard, self.model_router)
        return await expert.review_request(prompt)

    async def _select_experts(self, task: Task) -> List[str]:
        """
        Select experts for task.

        Args:
            task: Task object

        Returns:
            List of expert IDs
        """
        # Default: PM + Visual Expert
        selected = ["project_manager_01", "visual_expert_01"]

        # Analyze task to determine needed experts
        task_type = task.type

        if task_type == "image_generation":
            selected.append("prompt_engineer_01")
        elif task_type == "image_editing":
            selected.append("image_editor_01")
        elif task_type == "interior_design":
            selected.append("interior_designer_01")
        elif task_type == "product_display":
            selected.append("display_designer_01")
        elif task_type == "photography":
            selected.append("photographer_01")

        # Filter to only registered experts
        return [e for e in selected if self.registry.has_expert(e)]

    async def _execute_experts(
        self,
        task: Task,
        expert_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Execute selected experts.

        Args:
            task: Task object
            expert_ids: List of expert IDs

        Returns:
            Results from all experts
        """
        results = {}

        for expert_id in expert_ids:
            expert = self.registry.create_instance(
                expert_id,
                self.blackboard,
                self.model_router,
                as_executor=True
            )

            if expert:
                # Set confidence based on feasibility (default 0.8)
                expert.set_confidence(0.8)

                # Process task
                result = await expert.process({
                    "task_id": task.id,
                    "task_type": task.type,
                    "data": task.data
                })

                results[expert_id] = {
                    "result": result,
                    "confidence": expert.get_confidence(),
                    "history": expert.get_history()
                }

        return results

    async def _final_compliance_review(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Final compliance review before delivery.

        Args:
            results: Expert results

        Returns:
            Compliance review result
        """
        from .experts.compliance import ComplianceExpert

        config = self.registry._expert_configs.get("compliance_legal_01")
        if not config:
            return {"passed": True, "violations": []}

        expert = ComplianceExpert(config, self.blackboard, self.model_router)

        # Review the combined results
        output_data = {
            "expert_results": results
        }

        return await expert.review_output(output_data)

    async def _attempt_fallback(
        self,
        original_prompt: str,
        violations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt auto-fallback for fixable violations.

        Args:
            original_prompt: Original user prompt
            violations: Violation details

        Returns:
            Fallback result
        """
        handler = AutoFallbackHandler(self.model_router)

        violations_list = violations.get("violations", [])
        if not violations_list:
            return {"success": False}

        # Try to fix the first violation
        first_violation = violations_list[0]
        violation_type = first_violation.get("type", "unknown")

        if not handler.can_autofallback(violation_type):
            return {"success": False, "reason": "violation_not_fixable"}

        success, modified_prompt = await handler.attempt_remediation(
            violation_type,
            original_prompt,
            first_violation.get("description", "")
        )

        return {
            "success": success,
            "modified_prompt": modified_prompt,
            "attempts": handler.get_attempts()
        }

    async def _save_trace(
        self,
        task_id: str,
        results: Dict[str, Any]
    ) -> str:
        """
        Save execution trace to disk.

        Args:
            task_id: Task ID
            results: Expert results

        Returns:
            Path to trace file
        """
        trace_dir = self.output_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)

        trace_path = trace_dir / f"{task_id}.json"

        trace_data = {
            "task_id": task_id,
            "timestamp": int(time.time() * 1000),
            "results": results,
            "blackboard_events": [
                e.to_dict() for e in self.blackboard.get_events(task_id=task_id)
            ]
        }

        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        return str(trace_path)

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        import uuid
        return str(uuid.uuid4())

    def _create_task(
        self,
        task_id: str,
        prompt: str,
        images: Optional[List[bytes]],
        context: Optional[Dict]
    ) -> Task:
        """Create task object."""
        return Task(
            id=task_id,
            type=self._infer_task_type(prompt),
            status=TaskStatus.PROCESSING,
            data={
                "prompt": prompt,
                "images": [len(img) for img in images] if images else None,
                "context": context or {}
            },
            created_at=int(time.time()),
            updated_at=int(time.time())
        )

    def _infer_task_type(self, prompt: str) -> str:
        """Infer task type from prompt."""
        prompt_lower = prompt.lower()

        if any(kw in prompt_lower for kw in ["generate", "create", "make"]):
            return "image_generation"
        elif any(kw in prompt_lower for kw in ["edit", "modify", "remove", "add"]):
            return "image_editing"
        elif any(kw in prompt_lower for kw in ["room", "interior", "furniture"]):
            return "interior_design"
        elif any(kw in prompt_lower for kw in ["product", "display", "showcase"]):
            return "product_display"
        elif any(kw in prompt_lower for kw in ["photo", "lighting", "composition"]):
            return "photography"

        return "general"

    def _build_success_response(
        self,
        task_id: str,
        results: Dict[str, Any],
        trace_path: str,
        duration_sec: float
    ) -> Dict[str, Any]:
        """Build success response."""
        return {
            "status": "success",
            "task_id": task_id,
            "duration_sec": round(duration_sec, 2),
            "results": results,
            "trace_path": trace_path,
            "memory_stats": self.memory.get_stats()
        }

    def _build_rejection_response(
        self,
        task_id: str,
        review: Dict[str, Any],
        duration_sec: float
    ) -> Dict[str, Any]:
        """Build rejection response."""
        return {
            "status": "rejected",
            "task_id": task_id,
            "duration_sec": round(duration_sec, 2),
            "reason": "compliance_violation",
            "violations": review.get("violations", []),
            "severity": review.get("severity", "unknown")
        }

    def _build_error_response(
        self,
        task_id: str,
        error: str,
        duration_sec: float
    ) -> Dict[str, Any]:
        """Build error response."""
        return {
            "status": "error",
            "task_id": task_id,
            "duration_sec": round(duration_sec, 2),
            "error": error
        }

    def _finalize_error_response(
        self,
        task_id: str,
        error: str,
        duration_sec: float
    ) -> Dict[str, Any]:
        """Build final error response with fallback info."""
        return {
            "status": "error",
            "task_id": task_id,
            "duration_sec": round(duration_sec, 2),
            "error": error,
            "blackboard_cleared": True
        }
