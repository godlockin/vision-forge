# Vision Forge Architecture

This document provides a comprehensive overview of the Vision Forge expert system architecture.

## System Overview

Vision Forge is an intelligent vision review expert system that receives images + natural language requirements + reference information and outputs high-quality image results + review reports + traceable operation links.

## Table of Contents

1. [Expert System Design](#expert-system-design)
2. [Blackboard Pattern](#blackboard-pattern)
3. [Memory Architecture](#memory-architecture)
4. [Service Layer](#service-layer)
5. [Workflow](#workflow)
6. [Component Diagrams](#component-diagrams)

---

## Expert System Design

### 10 Static Experts

Vision Forge includes 10 static experts, each configured via YAML files in `experts/static/`:

| Expert ID | Role | Archetype | Load Strategy |
|-----------|------|-----------|---------------|
| `project_manager_01` | Project Manager | static | always |
| `compliance_legal_01` | Compliance & Legal | static | always |
| `visual_expert_01` | Visual Expert | static | always |
| `knowledge_admin_01` | Knowledge Admin | static | always |
| `hr_expert_01` | Senior HR Expert | static | on_demand |
| `prompt_engineer_01` | Prompt Engineer | static | on_demand |
| `photographer_01` | Photographer | static | on_demand |
| `image_editor_01` | Image Editor | static | on_demand |
| `interior_designer_01` | Interior Designer | static | on_demand |
| `display_designer_01` | Display Designer | static | on_demand |

### Expert YAML Schema

Each expert is defined using a comprehensive YAML schema:

```yaml
expert:
  id: "unique_expert_id"
  role: "Role Name"
  archetype: "static | dynamic"
  load_strategy: "always | on_demand | dynamic"

  persona:
    description: "Role description"
    background: "Background story"
    personality: "Personality traits"

  thinking_framework:
    - "Thinking principle 1"
    - "Thinking principle 2"

  strengths:
    - "Core strength 1"
    - "Core strength 2"

  weaknesses:
    - "Potential weakness 1"

  blind_spots:
    - "Blind spot 1"

  superhuman_insights:
    - "Superhuman insight 1"

  capabilities:
    - name: "Capability Name"
      params: { param1: "value1" }

  i_o_spec:
    input: { type: "input_type", format: "format" }
    output: { type: "output_type", format: "format" }

  decision_style:
    risk_tolerance: "very_low | low | medium | high"
    consensus_need: "none | low | medium | high"

  memory:
    scope: "task | session | persistent"
    retention: "task | session | persistent | permanent"

  # Optional fields
  trigger_conditions:
    - "Trigger condition 1"

  veto_power:
    enabled: true
    stages: ["pre_task", "pre_delivery"]

  split_behavior:
    enabled: true
    instances:
      - role: "executor"
        modifier: "Description"
      - role: "critic"
        modifier: "Description"
```

### Dynamic Expert Generation

Dynamic experts are generated at runtime when specialized domain knowledge is needed:

1. **Gap Analysis**: HR expert identifies capability gaps
2. **Profile Definition**: HR defines expert profile requirements
3. **LLM Generation**: LLM fills dynamic expert template
4. **Feasibility Scoring**: Generated expert is scored for feasibility
5. **Registration**: Expert registered to expert index

Dynamic expert templates are stored in `experts/dynamic/` with naming convention:
`dynamic_expert_<domain>_<timestamp>.yml`

### Expert Base Class

All experts inherit from the `Expert` abstract base class:

```python
class Expert(ABC):
    """Abstract base class for all experts."""

    def __init__(
        self,
        config: ExpertConfig,
        blackboard: SharedBlackboard,
        model_router: ModelRouter
    ):
        self.config = config
        self.blackboard = blackboard
        self.model_router = model_router
        self._confidence: Optional[float] = None
        self._is_executor = True

    @abstractmethod
    async def process(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process task and return result."""
        pass

    async def publish_opinion(
        self,
        task_id: str,
        opinion: str,
        score: float = 1.0,
        round_number: Optional[int] = None
    ):
        """Publish opinion to blackboard."""
        pass

    async def publish_score(
        self,
        task_id: str,
        target_expert_id: str,
        score: float,
        rationale: str = ""
    ):
        """Publish score for another expert's opinion."""
        pass
```

---

## Blackboard Pattern

Vision Forge uses the **Blackboard Pattern** with **Append-Only Event Sourcing** for expert collaboration.

### Event Types

```python
class EventType(Enum):
    OPINION_ADDED = "opinion_added"
    SCORE_SUBMITTED = "score_submitted"
    DECISION_MADE = "decision_made"
    ROUND_CLOSED = "round_closed"
    VETO_TRIGGERED = "veto_triggered"
    SNAPSHOT_COMPRESSED = "snapshot_compressed"
```

### BlackboardEvent Structure

```python
@dataclass
class BlackboardEvent:
    type: EventType
    expert_id: str
    task_id: str
    data: Dict[str, Any]
    timestamp: int  # milliseconds
    round_number: int
```

### Key Features

- **Append-Only**: Experts can only add events, not modify history
- **Idempotent**: Deduplication within 5-second window
- **Round-Based**: Events organized into discussion rounds
- **Async Persistence**: Events flushed to disk asynchronously
- **Subscriber Pattern**: Components can subscribe to event notifications

### Usage Example

```python
from vision_forge import SharedBlackboard, BlackboardEvent, EventType

blackboard = SharedBlackboard(persist_dir="output/blackboard")

# Append event
event = BlackboardEvent(
    type=EventType.OPINION_ADDED,
    expert_id="visual_expert_01",
    task_id="task-123",
    data={"opinion": "Image quality is excellent", "score": 0.9},
    round_number=1
)
await blackboard.append(event)

# Query events
opinions = blackboard.get_opinions(task_id="task-123")
scores = blackboard.get_scores(task_id="task-123")
current_round = blackboard.get_current_round(task_id="task-123")
```

---

## Memory Architecture

Vision Forge implements a **Three-Layer Memory System**:

| Layer | Scope | Content | Retention |
|-------|-------|---------|-----------|
| Task Memory | Single task | Intermediate states, temporary decisions | Cleared after task |
| Session Memory | Single session | Expert interaction records | Cleared after session |
| Persistent Memory | Global | Task history, user preferences, domain knowledge | Compressed, permanent |

### MemoryManager API

```python
class MemoryManager:
    """Three-layer memory manager."""

    async def store(
        self,
        key: str,
        value: Any,
        scope: str = "session",
        task_id: Optional[str] = None,
        importance: float = 1.0
    ):
        """Store memory entry."""
        pass

    async def retrieve(
        self,
        key: str,
        scope: str = "session",
        task_id: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve memory entry."""
        pass

    async def compress_session_memory(self, target_entries: int = 100):
        """Compress session memory to persistent storage."""
        pass

    def search(self, pattern: str, scope: str = "session") -> List[Dict[str, Any]]:
        """Search memory by key pattern."""
        pass
```

### Memory Compression

Automatic compression triggers when session memory exceeds threshold (default: 1000 entries):

1. Entries sorted by importance and access count
2. Top entries (importance > 0.7) moved to persistent memory
3. Low-importance entries discarded
4. Compression rate typically ~30%

---

## Service Layer

The service layer provides abstractions for LLM provider interactions.

### Architecture

```
┌─────────────────────────────────────────┐
│         ModelRouter                      │
│  - Smart model selection                 │
│  - Cost optimization                     │
│  - Provider failover                     │
└─────────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐       ┌─────────┐
│  Rate   │       │  Retry  │
│ Limiter │       │ Handler │
└─────────┘       └─────────┘
    │                   │
    ▼                   ▼
┌─────────────────────────────────┐
│       BaseService                │
│  - Azure OpenAI Service          │
│  - Vertex AI Service             │
└─────────────────────────────────┘
```

### Task Types

```python
class TaskType(Enum):
    TEXT_REASONING = "text_reasoning"
    VISUAL_ANALYSIS = "visual_analysis"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    SUMMARY = "summary"
    COMPLIANCE_CHECK = "compliance_check"
    EMBEDDING = "embedding"
```

### Model Router

```python
class ModelRouter:
    """Smart model router based on task type and cost optimization."""

    def get_best_model(self, task_type: TaskType) -> Tuple[str, BaseService]:
        """Get the best model for a task type."""
        pass

    async def route_request(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> ServiceResponse | ImageGenerationResponse:
        """Route request to appropriate model."""
        pass

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for a request."""
        pass
```

### Model Routing Rules

| Task Type | Primary Model | Fallback | Rationale |
|-----------|---------------|----------|-----------|
| Text Reasoning | gpt-4o | gemini-2.0-pro | Best overall reasoning |
| Summary | gpt-4o-mini | gemini-2.0-flash | Cost optimization |
| Compliance Check | gpt-4o | gemini-2.0-pro | Highest reliability |
| Image Generation | imagen-3.0-generate-001 | N/A | Only image model |
| Embedding | text-embedding-3-small | text-embedding-005 | Industry standard |

### Rate Limiting

Token bucket algorithm with configurable limits:

```yaml
# rate_limits.yml
azure_openai:
  rpm_limit: 60
  tpm_limit: 500000
  burst_size: 10

vertex_ai:
  rpm_limit: 30
  tpm_limit: 300000
  image_gen_rpm_limit: 10
```

### Retry Policy

Exponential backoff with jitter:

```yaml
# fallback_policy.yml
retry:
  max_attempts: 3
  backoff_multiplier: 2.0
  initial_delay_ms: 1000
  max_delay_ms: 30000
  jitter: 0.1
```

---

## Workflow

Vision Forge follows a 7-step workflow with built-in quality gates:

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Request Entry                            │
│         (Image + Natural Language + Constraints)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│             [Veto Check] Compliance & Legal Review              │
│             Pass? Yes → Continue | No → Reject                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Task Init & Feasibility Analysis                   │
│     PM + Visual Expert + HR select experts → Feasibility Score  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Expert Confidence Self-Assessment               │
│        Experts set Confidence Score (0.0-1.0) after accepting   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Faction Split & Concurrency                   │
│    ┌─────────────────┐         ┌─────────────────┐              │
│    │ Executor Group  │         │   Critic Group  │              │
│    │ Complete tasks  │         │ Critical review │              │
│    └─────────────────┘         └─────────────────┘              │
│        Dependencies → Serial | No Dependencies → Parallel        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Blackboard Discussion & Weighted Scoring          │
│  - All outputs synced to blackboard                             │
│  - Experts freely comment and score                             │
│  - Final = Score × (Feasibility × Confidence)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          [Forced Convergence] Voting Algorithm                  │
│    Discussion exceeds threshold → PM organizes voting           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               [Veto Check] Final Compliance Review              │
│              Compliance expert reviews final output             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Result Delivery & Trace Recording              │
│  - Final media result                                           │
│  - Expert review report (multi-dimensional scoring)             │
│  - Operation link log (delete A, add B, adjust C...)            │
│  - Supports user fork and continuation                          │
└─────────────────────────────────────────────────────────────────┘
```

### Workflow Orchestration

The `WorkflowOrchestrator` coordinates all workflow steps:

```python
class WorkflowOrchestrator:
    """Main workflow orchestrator."""

    async def process_request(
        self,
        user_prompt: str,
        images: Optional[List[bytes]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process complete user request."""
        # Step 1: Compliance review (veto check)
        compliance_result = await self._compliance_review(user_prompt)
        if not compliance_result["passed"]:
            return self._build_rejection_response(...)

        # Step 2: Initialize task
        task = self._create_task(task_id, user_prompt, images, context)

        # Step 3: Select experts
        selected_experts = await self._select_experts(task)

        # Step 4: Execute experts
        results = await self._execute_experts(task, selected_experts)

        # Step 5: Final compliance review
        final_review = await self._final_compliance_review(results)
        if not final_review["passed"]:
            fallback_result = await self._attempt_fallback(user_prompt, final_review)

        # Step 6: Save trace
        trace_path = await self._save_trace(task_id, results)

        # Step 7: Build response
        return self._build_success_response(...)
```

---

## Component Diagrams

### Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Core Models                              │
├─────────────────────────────────────────────────────────────────┤
│  ExpertConfig  │  Task  │  Job  │  Persona  │  Capability       │
│  Archetype     │  TaskStatus  │  MemoryConfig  │  DecisionStyle │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Expert Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Expert (ABC)                                                   │
│    ├── PMExpert                                                 │
│    ├── ComplianceExpert                                         │
│    ├── VisualExpert                                             │
│    ├── KnowledgeAdminExpert                                     │
│    ├── HRExpert                                                 │
│    ├── PromptEngineerExpert                                     │
│    ├── PhotographerExpert                                       │
│    ├── ImageEditorExpert                                        │
│    ├── InteriorDesignerExpert                                   │
│    └── DisplayDesignerExpert                                    │
│                                                                 │
│  ExpertRegistry                                                 │
│  DynamicExpertGenerator                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Memory Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  MemoryManager                                                  │
│    ├── TaskMemory                                               │
│    ├── SessionMemory                                            │
│    └── PersistentMemory                                         │
│                                                                 │
│  SharedBlackboard                                               │
│    ├── BlackboardEvent                                          │
│    └── EventType                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Service Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  BaseService (ABC)                                              │
│    ├── AzureOpenAIService                                       │
│    └── VertexAIService                                          │
│                                                                 │
│  ModelRouter                                                    │
│  RateLimiter                                                    │
│  RetryHandler                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  WorkflowOrchestrator                                           │
│  AutoFallbackHandler                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Sequence Diagram: Expert Collaboration

```
User → Orchestrator → Compliance → PM → HR → Experts → Blackboard → Memory

1. User submits request
2. Orchestrator invokes compliance review
3. If passed, PM analyzes task
4. HR identifies capability gaps
5. Experts recruited and instantiated
6. Experts execute in parallel/serial
7. Opinions published to blackboard
8. Scores submitted for peer review
9. PM aggregates results
10. Final compliance review
11. Results delivered with trace
```

### Deployment Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Local Machine                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Vision Forge Expert System                      ││
│  │  - Expert instances                                          ││
│  │  - Blackboard                                                ││
│  │  - Memory manager                                            ││
│  │  - Model router                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│         HTTPS/API Calls                                            │
└─────────────────────────────────────────────────────────────────┘
               │                        │
               ▼                        ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│     Azure OpenAI        │  │      Google Vertex AI   │
│  - GPT-4o               │  │  - Gemini 2.5/3.0       │
│  - GPT-4o-mini          │  │  - Imagen-3             │
│  - o1-preview/mini      │  │                         │
└─────────────────────────┘  └─────────────────────────┘
```

---

## Configuration Files

Vision Forge uses YAML configuration files for all system settings:

| File | Purpose |
|------|---------|
| `static_experts.yml` | 10 static expert definitions |
| `dynamic_expert_template.yml` | Dynamic expert generation template |
| `rate_limits.yml` | API rate limiting configuration |
| `model_routing.yml` | Model whitelist and routing rules |
| `blackboard_events.yml` | Event type definitions |
| `intervention_rules.yml` | Confidence thresholds, intervention triggers |
| `fallback_policy.yml` | Compliance violation classification, auto-fallback |

### Configuration Location

All configuration files are located in `sys_init/settings/` and loaded at startup via `ConfigLoader`.

---

## Design Decisions

### Key Architectural Choices

1. **Append-Only Blackboard**: Enables full traceability and debugging
2. **Executor/Critic Split**: Prevents groupthink, ensures quality
3. **Weighted Consensus**: Combines multiple signals for robust decisions
4. **Three-Layer Memory**: Balances performance with persistence
5. **Model Router**: Abstracts provider differences, optimizes cost
6. **Auto-Fallback**: Automatic remediation for fixable violations
7. **Compliance Veto**: Ensures legal/safety requirements met

### Trade-offs

- **In-Memory Expert Instances**: Fast access but requires restart for config changes
- **Event Sourcing**: Full history but requires compression for long sessions
- **YAML Configuration**: Human-readable but requires validation at startup

---

## Performance Considerations

### Concurrency Limits

```yaml
concurrency:
  max_concurrent_requests: 10
  max_concurrent_per_provider: 5
  max_concurrent_per_expert: 2
```

### Memory Thresholds

```yaml
memory:
  compression_threshold: 1000  # entries before compression
  target_persistent_entries: 100
```

### Rate Limits

| Provider | RPM | TPM | Burst |
|----------|-----|-----|-------|
| Azure OpenAI | 60 | 500,000 | 10 |
| Vertex AI | 30 | 300,000 | 5 |
| Vertex AI (Image) | 10 | N/A | 2 |

---

## Security Considerations

1. **Environment Variables**: All secrets loaded from environment
2. **No Hardcoded Credentials**: API keys never in code or config
3. **Input Validation**: All user inputs validated via Pydantic
4. **Compliance Review**: All prompts and outputs reviewed for safety

---

## Extensibility

### Adding a New Expert

1. Create YAML configuration in `experts/static/`
2. Implement expert class inheriting from `Expert`
3. Register expert in `ExpertRegistry`
4. Update documentation

### Adding a New Provider

1. Create service class inheriting from `BaseService`
2. Implement `generate_text()` and `generate_image()`
3. Add to `ModelRouter` service dictionary
4. Configure rate limits and model whitelist

### Adding a New Task Type

1. Add `TaskType` enum value
2. Implement routing rule in `ModelRouter.get_best_model()`
3. Add expert capability if needed
4. Update workflow orchestrator
