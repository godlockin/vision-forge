# Vision Forge API Reference

This document provides comprehensive API reference for all public classes and functions in Vision Forge.

## Table of Contents

1. [Core Models](#core-models)
2. [Expert Base Class](#expert-base-class)
3. [ExpertRegistry](#expertregistry)
4. [Blackboard](#blackboard)
5. [MemoryManager](#memorymanager)
6. [ModelRouter](#modelrouter)
7. [Service Interfaces](#service-interfaces)
8. [WorkflowOrchestrator](#workfloworchestrator)

---

## Core Models

### Archetype

Expert archetype enumeration.

```python
class Archetype(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
```

### LoadStrategy

Expert loading strategy enumeration.

```python
class LoadStrategy(str, Enum):
    ALWAYS = "always"
    ON_DEMAND = "on_demand"
    DYNAMIC = "dynamic"
```

### TaskStatus

Task status enumeration.

```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Persona

Expert persona configuration.

```python
class Persona(BaseModel):
    description: str       # Brief role description
    background: str        # Background story
    personality: str       # Personality traits
```

### Capability

Expert capability definition.

```python
class Capability(BaseModel):
    name: str                          # Capability name
    params: Dict[str, Any]             # Capability parameters
```

### IOSpec

Input/Output specification for experts.

```python
class IOSpec(BaseModel):
    input: Dict[str, str]              # Input type and format
    output: Dict[str, str]             # Output type and format
```

### DecisionStyle

Decision making style configuration.

```python
class DecisionStyle(BaseModel):
    risk_tolerance: Literal["very_low", "low", "medium", "high"]
    consensus_need: Literal["none", "low", "medium", "high"]
```

### MemoryConfig

Memory configuration for experts.

```python
class MemoryConfig(BaseModel):
    scope: Literal["task", "session", "persistent"]
    retention: Literal["task", "session", "persistent", "permanent"]
```

### ExpertConfig

Complete expert configuration.

```python
class ExpertConfig(BaseModel):
    id: str                           # Unique expert identifier
    role: str                         # Role name
    archetype: Archetype              # Static or dynamic
    load_strategy: LoadStrategy       # Loading strategy
    persona: Persona                  # Expert persona
    thinking_framework: List[str]     # Thinking principles
    strengths: List[str]              # Core strengths
    weaknesses: List[str]             # Potential weaknesses
    blind_spots: List[str]            # Blind spots
    superhuman_insights: List[str]    # Superhuman insights
    capabilities: List[Capability]    # Capabilities
    i_o_spec: Optional[IOSpec]        # I/O specification
    decision_style: Optional[DecisionStyle]  # Decision style
    memory: Optional[MemoryConfig]    # Memory configuration

    # Optional fields
    trigger_conditions: Optional[List[str]]     # Trigger conditions
    veto_power: Optional[Dict[str, Any]]        # Veto power config
    split_behavior: Optional[Dict[str, Any]]    # Split behavior config
```

### Task

Task representation.

```python
class Task(BaseModel):
    id: str                           # Unique task ID
    type: str                         # Task type
    status: TaskStatus                # Current status
    data: Dict[str, Any]              # Task data
    result: Optional[Dict[str, Any]]  # Task result
    error: Optional[str]              # Error message
    created_at: int                   # Creation timestamp
    updated_at: int                   # Update timestamp
```

### Job

Job representation for async operations.

```python
class Job(BaseModel):
    id: str                           # Unique job ID
    status: TaskStatus                # Current status
    action: str                       # Action name
    data: Dict[str, Any]              # Job data
    result: Optional[Dict[str, Any]]  # Job result
    error: Optional[str]              # Error message
    created_at: int                   # Creation timestamp
    updated_at: int                   # Update timestamp
```

---

## Expert Base Class

### Expert

Abstract base class for all experts.

```python
class Expert(ABC):
    """Abstract base class for all experts."""
```

#### Constructor

```python
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
```

#### Properties

```python
@property
def id(self) -> str:
    """Get expert ID."""
    pass

@property
def role(self) -> str:
    """Get expert role name."""
    pass

@property
def is_executor(self) -> bool:
    """Check if expert is in executor mode."""
    pass
```

#### Confidence Management

```python
def set_confidence(self, confidence: float):
    """
    Set confidence score for this expert instance.

    Args:
        confidence: Confidence score (0.0-1.0)
    """
    pass

def get_confidence(self) -> Optional[float]:
    """Get current confidence score."""
    pass
```

#### Mode Setting

```python
def as_executor(self) -> "Expert":
    """Set expert to executor mode."""
    pass

def as_critic(self) -> "Expert":
    """Set expert to critic mode."""
    pass
```

#### Core Methods

```python
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
    pass

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
    pass
```

#### LLM Integration

```python
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
    pass
```

#### History Management

```python
def record_action(self, action: str, result: Any):
    """
    Record action in task history.

    Args:
        action: Action description
        result: Action result
    """
    pass

def get_history(self) -> List[Dict[str, Any]]:
    """
    Get task history.

    Returns:
        List of recorded actions
    """
    pass

def clear_history(self):
    """Clear task history."""
    pass
```

---

## ExpertRegistry

Registry for expert classes and instances.

```python
class ExpertRegistry:
    """Registry for managing expert instances."""
```

### Constructor

```python
def __init__(self):
    """Initialize registry."""
```

### Class Registration

```python
def register_class(self, expert_id: str, expert_class: Type[Expert]):
    """
    Register an expert class.

    Args:
        expert_id: Expert ID to register for
        expert_class: Expert class type
    """
    pass

def register_config(self, config: ExpertConfig):
    """
    Register an expert configuration.

    Args:
        config: Expert configuration
    """
    pass
```

### Instance Management

```python
def create_instance(
    self,
    expert_id: str,
    blackboard,
    model_router,
    as_executor: bool = True
) -> Optional[Expert]:
    """
    Create expert instance.

    Args:
        expert_id: Expert ID to create
        blackboard: Shared blackboard instance
        model_router: Model router instance
        as_executor: Whether to create as executor (vs critic)

    Returns:
        Expert instance or None if not found
    """
    pass

def get_instance(self, expert_id: str) -> Optional[Expert]:
    """
    Get active expert instance.

    Args:
        expert_id: Expert ID

    Returns:
        Expert instance or None
    """
    pass

def remove_instance(self, expert_id: str):
    """
    Remove expert instance from registry.

    Args:
        expert_id: Expert ID to remove
    """
    pass

def clear_instances(self):
    """Clear all expert instances."""
    pass
```

### Configuration Queries

```python
def get_all_configs(self) -> List[ExpertConfig]:
    """
    Get all registered configurations.

    Returns:
        List of ExpertConfig objects
    """
    pass

def get_configs_by_archetype(self, archetype: str) -> List[ExpertConfig]:
    """
    Get configs filtered by archetype.

    Args:
        archetype: "static" or "dynamic"

    Returns:
        List of matching ExpertConfig objects
    """
    pass

def get_configs_by_load_strategy(self, strategy: str) -> List[ExpertConfig]:
    """
    Get configs filtered by load strategy.

    Args:
        strategy: "always", "on_demand", or "dynamic"

    Returns:
        List of matching ExpertConfig objects
    """
    pass

def get_expert_ids(self) -> List[str]:
    """
    Get list of registered expert IDs.

    Returns:
        List of expert IDs
    """
    pass

def has_expert(self, expert_id: str) -> bool:
    """
    Check if expert is registered.

    Args:
        expert_id: Expert ID

    Returns:
        True if registered
    """
    pass
```

### Factory Methods

```python
@classmethod
def from_directory(cls, directory: str = "experts/static") -> "ExpertRegistry":
    """
    Create registry from YAML configs in directory.

    Args:
        directory: Directory containing expert YAML files

    Returns:
        Configured ExpertRegistry instance
    """
    pass

def load_additional_configs(self, directory: str):
    """
    Load additional configs from directory.

    Args:
        directory: Directory containing YAML files
    """
    pass
```

### Statistics

```python
def get_stats(self) -> Dict[str, int]:
    """
    Get registry statistics.

    Returns:
        Dictionary with stats
    """
    pass
```

---

## Blackboard

### EventType

Event type enumeration for blackboard events.

```python
class EventType(Enum):
    OPINION_ADDED = "opinion_added"
    SCORE_SUBMITTED = "score_submitted"
    DECISION_MADE = "decision_made"
    ROUND_CLOSED = "round_closed"
    VETO_TRIGGERED = "veto_triggered"
    SNAPSHOT_COMPRESSED = "snapshot_compressed"
```

### BlackboardEvent

Event data structure for blackboard.

```python
@dataclass
class BlackboardEvent:
    type: EventType                  # Event type
    expert_id: str                   # Expert ID
    task_id: str                     # Task ID
    data: Dict[str, Any]             # Event data
    timestamp: int                   # Timestamp (ms)
    round_number: int                # Discussion round
```

### SharedBlackboard

Append-only event sourcing blackboard.

```python
class SharedBlackboard:
    """Append-only event sourcing blackboard."""
```

#### Constructor

```python
def __init__(self, persist_dir: str = "output/blackboard"):
    """
    Initialize blackboard.

    Args:
        persist_dir: Directory for persisting events
    """
```

#### Event Operations

```python
async def append(self, event: BlackboardEvent) -> bool:
    """
    Append event to blackboard (idempotent within dedup window).

    Args:
        event: Event to append

    Returns:
        True if appended, False if duplicate
    """
    pass

def subscribe(self, callback: Callable[[BlackboardEvent], None]):
    """
    Subscribe to new events.

    Args:
        callback: Function to call when new event arrives
    """
    pass
```

#### Query Operations

```python
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
        since_timestamp: Filter by timestamp

    Returns:
        List of matching events
    """
    pass

def get_current_round(self, task_id: str) -> int:
    """
    Get current round number for a task.

    Args:
        task_id: Task ID

    Returns:
        Current round number
    """
    pass

def get_opinions(self, task_id: str) -> List[Dict[str, Any]]:
    """
    Get all opinions for a task.

    Args:
        task_id: Task ID

    Returns:
        List of opinion data
    """
    pass

def get_scores(self, task_id: str) -> List[Dict[str, Any]]:
    """
    Get all scores for a task.

    Args:
        task_id: Task ID

    Returns:
        List of score data
    """
    pass
```

#### Management Operations

```python
def clear(self, task_id: str):
    """
    Clear events for a completed task.

    Args:
        task_id: Task ID to clear
    """
    pass

def get_stats(self) -> Dict[str, Any]:
    """
    Get blackboard statistics.

    Returns:
        Dictionary with stats
    """
    pass

def load_from_file(self, filepath: str) -> int:
    """
    Load events from a file.

    Args:
        filepath: Path to JSONL file

    Returns:
        Number of events loaded
    """
    pass
```

---

## MemoryManager

Three-layer memory manager.

```python
class MemoryManager:
    """Three-layer memory manager."""
```

### Constructor

```python
def __init__(self, compression_threshold: int = 1000):
    """
    Initialize memory manager.

    Args:
        compression_threshold: Number of entries before compression triggers
    """
```

### Core Operations

```python
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
    pass

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
    pass

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
    pass
```

### Memory Management

```python
async def clear_task_memory(self, task_id: str):
    """
    Clear memory for completed task.

    Args:
        task_id: Task ID to clear
    """
    pass

async def clear_session_memory(self):
    """Clear all session memory."""
    pass

async def compress_session_memory(self, target_entries: int = 100):
    """
    Compress session memory to persistent storage.

    Args:
        target_entries: Target number of entries to keep in session
    """
    pass
```

### Search and Export

```python
def search(self, pattern: str, scope: str = "session") -> List[Dict[str, Any]]:
    """
    Search memory by key pattern.

    Args:
        pattern: Search pattern (substring match)
        scope: Memory scope to search

    Returns:
        List of matching entries with their values
    """
    pass

async def export_to_dict(self) -> Dict[str, Any]:
    """
    Export all memory to dictionary.

    Returns:
        Dictionary representation of all memories
    """
    pass

def get_stats(self) -> Dict[str, int]:
    """
    Get memory statistics.

    Returns:
        Dictionary with entry counts per layer
    """
    pass
```

---

## ModelRouter

Smart model router based on task type and cost optimization.

```python
class ModelRouter:
    """Smart model router."""
```

### Constructor

```python
def __init__(self, services: Dict[str, BaseService]):
    """
    Initialize model router.

    Args:
        services: Dictionary of provider name to service instance
    """
```

### Factory Methods

```python
@classmethod
def from_env(cls) -> "ModelRouter":
    """
    Create router from environment configuration.

    Returns:
        ModelRouter instance with configured services
    """
    pass
```

### Model Selection

```python
def get_best_model(self, task_type: TaskType) -> Tuple[str, BaseService]:
    """
    Get the best model for a task type.

    Args:
        task_type: Type of task

    Returns:
        Tuple of (model_name, service)

    Raises:
        ValueError: If no suitable service available
    """
    pass

def get_available_models(self) -> Dict[str, list]:
    """
    Get list of available models by provider.

    Returns:
        Dictionary of provider to model list
    """
    pass
```

### Request Routing

```python
async def route_request(
    self,
    task_type: TaskType,
    prompt: str,
    **kwargs
) -> ServiceResponse | ImageGenerationResponse:
    """
    Route request to appropriate model.

    Args:
        task_type: Type of task
        prompt: Input prompt
        **kwargs: Additional parameters

    Returns:
        Service response

    Raises:
        ValueError: If no suitable service
    """
    pass
```

### Cost Estimation

```python
def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Estimate cost for a request.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    pass
```

---

## Service Interfaces

### TaskType

Task type enumeration for model routing.

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

### ServiceResponse

Response from text generation service.

```python
@dataclass
class ServiceResponse:
    content: str                    # Generated text
    usage: Dict[str, int]           # Token usage
    model: str                      # Model used
    latency_ms: int                 # Request latency
    finish_reason: Optional[str]    # Finish reason
```

### ImageGenerationResponse

Response from image generation service.

```python
@dataclass
class ImageGenerationResponse:
    image_url: Optional[str]        # Image URL
    image_data: Optional[bytes]     # Image bytes
    prompt_used: str                # Prompt used
    model: str                      # Model used
    latency_ms: int                 # Request latency
    negative_prompt: Optional[str]  # Negative prompt
    parameters: Dict[str, Any]      # Generation parameters
```

### BaseService

Abstract base class for LLM services.

```python
class BaseService(ABC):
    """Abstract base class for LLM services."""
```

#### Constructor

```python
def __init__(self, provider_name: str):
    """
    Initialize base service.

    Args:
        provider_name: Provider name
    """
```

#### Core Methods

```python
@abstractmethod
async def generate_text(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> ServiceResponse:
    """
    Generate text response.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific parameters

    Returns:
        ServiceResponse with generated text
    """
    pass

@abstractmethod
async def generate_image(
    self,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> ImageGenerationResponse:
    """
    Generate image.

    Args:
        prompt: Text prompt for image
        negative_prompt: What to exclude from image
        width: Image width
        height: Image height
        **kwargs: Additional provider-specific parameters

    Returns:
        ImageGenerationResponse with image data
    """
    pass
```

#### Statistics

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get service statistics.

    Returns:
        Dictionary with stats
    """
    pass
```

---

## WorkflowOrchestrator

Main workflow orchestrator for coordinating expert system execution.

```python
class WorkflowOrchestrator:
    """Main workflow orchestrator."""
```

### Constructor

```python
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
```

### Main Method

```python
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
        Processing result with status, task_id, and results
    """
    pass
```

### Response Structure

**Success Response:**
```python
{
    "status": "success",
    "task_id": "uuid",
    "duration_sec": 12.34,
    "results": {...},
    "trace_path": "output/trace/uuid.json",
    "memory_stats": {...}
}
```

**Rejection Response:**
```python
{
    "status": "rejected",
    "task_id": "uuid",
    "duration_sec": 0.5,
    "reason": "compliance_violation",
    "violations": [...],
    "severity": "high"
}
```

**Error Response:**
```python
{
    "status": "error",
    "task_id": "uuid",
    "duration_sec": 0.1,
    "error": "Error message"
}
```

---

## Utility Functions

### Config Loader

```python
def load_expert_config(yaml_path: str) -> ExpertConfig:
    """
    Load expert configuration from YAML file.

    Args:
        yaml_path: Path to YAML file

    Returns:
        ExpertConfig object
    """
    pass

def load_all_static_experts(directory: str = "experts/static") -> List[ExpertConfig]:
    """
    Load all static expert configurations from directory.

    Args:
        directory: Directory containing YAML files

    Returns:
        List of ExpertConfig objects
    """
    pass
```

---

## Examples

### Creating an Expert

```python
from vision_forge import ExpertRegistry, ExpertConfig, Persona

# Create expert config
config = ExpertConfig(
    id="custom_expert_01",
    role="Custom Expert",
    archetype=Archetype.DYNAMIC,
    load_strategy=LoadStrategy.ON_DEMAND,
    persona=Persona(
        description="A custom expert",
        background="Extensive experience",
        personality="Professional"
    ),
    thinking_framework=["Think step by step"],
    strengths=["Specialized knowledge"],
    capabilities=[Capability(name="custom_capability", params={})]
)

# Register expert
registry = ExpertRegistry()
registry.register_config(config)
```

### Using the Blackboard

```python
from vision_forge import SharedBlackboard, BlackboardEvent, EventType

blackboard = SharedBlackboard()

# Publish opinion
event = BlackboardEvent(
    type=EventType.OPINION_ADDED,
    expert_id="expert_01",
    task_id="task_123",
    data={"opinion": "This is good", "score": 0.9},
    round_number=1,
    timestamp=int(time.time() * 1000)
)
await blackboard.append(event)

# Query opinions
opinions = blackboard.get_opinions("task_123")
```

### Memory Operations

```python
from vision_forge import MemoryManager

memory = MemoryManager()

# Store memory
await memory.store(
    key="user_preference",
    value={"style": "minimalist"},
    scope="session",
    importance=0.8
)

# Retrieve memory
value = await memory.retrieve("user_preference", scope="session")

# Search memory
results = memory.search("preference", scope="session")
```

### Model Routing

```python
from vision_forge import ModelRouter, TaskType

router = ModelRouter.from_env()

# Get best model
model_name, service = router.get_best_model(TaskType.TEXT_REASONING)

# Route request
response = await router.route_request(
    task_type=TaskType.TEXT_REASONING,
    prompt="Analyze this image",
    temperature=0.7
)
```

---

## Error Handling

All public methods follow consistent error handling patterns:

- **ValueError**: Invalid arguments or configuration
- **KeyError**: Resource not found
- **RuntimeError**: Service unavailable or internal error
- **TimeoutError**: Request timeout

### Retry Behavior

Methods that call external services implement automatic retry:

- Max attempts: 3
- Backoff: Exponential (1s, 2s, 4s)
- Jitter: 10% randomization

### Fallback Behavior

For compliance violations, auto-fallback is attempted:

- Max attempts: 3 for copyright/IP issues
- Max attempts: 2 for brand/trademark issues
- Direct rejection for sensitive content
