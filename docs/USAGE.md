# Vision Forge Usage Guide

This document provides comprehensive usage instructions for Vision Forge, including CLI commands, programmatic usage, configuration options, and best practices.

## Table of Contents

1. [CLI Commands](#cli-commands)
2. [Programmatic Usage](#programmatic-usage)
3. [Configuration Options](#configuration-options)
4. [Best Practices](#best-practices)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

---

## CLI Commands

Vision Forge provides a command-line interface for common operations.

### Installation

```bash
# Install package
pip install -e .

# Verify installation
python -m vision_forge --version
```

### Command Overview

| Command | Description |
|---------|-------------|
| `review` | Process image generation/editing request |
| `experts` | List available experts |
| `config` | Show current configuration |
| `test` | Test LLM service connectivity |
| `stats` | Show system statistics |
| `status` | Check task status |

### `review` Command

Process an image generation or editing request.

```bash
python -m vision_forge review [OPTIONS]
```

**Options:**
| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--prompt` | `-p` | Yes | User prompt for image generation/review |
| `--image` | `-i` | No | Input image file (can specify multiple) |
| `--output` | `-o` | No | Output directory (default: `output`) |
| `--verbose` | `-v` | No | Verbose output (show full JSON response) |

**Examples:**

```bash
# Basic image generation
python -m vision_forge review -p "Generate a product photo for a luxury watch"

# Image editing with input
python -m vision_forge review \
  -p "Enhance the lighting and remove background distractions" \
  -i input.jpg

# Multiple input images
python -m vision_forge review \
  -p "Merge these two images into a cohesive composition" \
  -i image1.jpg -i image2.jpg

# Verbose output
python -m vision_forge review -p "Generate a minimalist logo" -v
```

**Output:**

Success response:
```
Status: success
Task ID: 550e8400-e29b-41d4-a716-446655440000
Duration: 12.34s
```

Rejection response:
```
Status: rejected
Violations: [{'type': 'sensitive_content', 'description': '...'}]
```

Error response:
```
Status: error
Error: Service unavailable
```

Verbose output shows full JSON:
```json
{
  "status": "success",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_sec": 12.34,
  "results": {...},
  "trace_path": "output/trace/550e8400-e29b-41d4-a716-446655440000.json",
  "memory_stats": {...}
}
```

### `experts` Command

List all available static experts.

```bash
python -m vision_forge experts
```

**Output:**
```
Loaded 10 static experts:

  - project_manager_01: 项目经理
    Archetype: static
    Load Strategy: always

  - compliance_legal_01: 合规与法律专家
    Archetype: static
    Load Strategy: always

  - visual_expert_01: 视觉专家
    Archetype: static
    Load Strategy: always

  ...
```

### `config` Command

Show current environment configuration.

```bash
python -m vision_forge config
```

**Output:**
```
Current Configuration:

  OpenAI API Key: sk-...
  OpenAI API Base: https://...
  OpenAI Model: gpt-4o
  Google API Key: (not set)
  Vertex AI Project ID: my-project
  Vertex AI Location: us-central1
```

### `test` Command

Test LLM service connectivity.

```bash
python -m vision_forge test [OPTIONS]
```

**Options:**
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | all | Provider to test (azure, vertex, all) |
| `--prompt` | `-p` | Hello | Test prompt |

**Examples:**

```bash
# Test all providers
python -m vision_forge test

# Test only Azure OpenAI
python -m vision_forge test -p azure

# Test with custom prompt
python -m vision_forge test --prompt "Summarize this in one sentence"
```

**Output:**
```
Testing Azure OpenAI...
  Status: OK
  Model: gpt-4o
  Latency: 234ms

Testing Vertex AI...
  Status: OK
  Model: gemini-2.0-pro
  Latency: 456ms
```

### `stats` Command

Show system statistics.

```bash
python -m vision_forge stats
```

**Output:**
```
System Statistics:

Experts:
  registered_classes: 10
  registered_configs: 10
  active_instances: 0
  static_experts: 10
  dynamic_experts: 0
  always_load: 4
  on_demand: 6

Memory:
  task_memory_entries: 0
  task_memory_tasks: 0
  session_memory_entries: 0
  persistent_memory_entries: 0
  total_entries: 0

Blackboard:
  total_events: 0
  unique_tasks: 0
  rounds: {}
  event_types: {...}
```

### `status` Command

Check task status by task ID.

```bash
python -m vision_forge status --task-id <TASK_ID>
```

**Example:**
```bash
python -m vision_forge status --task-id 550e8400-e29b-41d4-a716-446655440000
```

**Output:**
```
Task found: 550e8400-e29b-41d4-a716-446655440000
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1710123456789,
  "results": {...},
  "blackboard_events": [...]
}
```

---

## Programmatic Usage

### Basic Setup

```python
import asyncio
from vision_forge import (
    WorkflowOrchestrator,
    ExpertRegistry,
    SharedBlackboard,
    MemoryManager,
    ModelRouter
)

async def main():
    # Initialize components
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard(persist_dir="output/blackboard")
    memory = MemoryManager(compression_threshold=1000)
    router = ModelRouter.from_env()

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        registry, blackboard, memory, router
    )

    # Process request
    result = await orchestrator.process_request(
        user_prompt="Generate a product image",
        images=[image_bytes],  # Optional
        context={"target_audience": "premium"}  # Optional
    )

    print(f"Status: {result['status']}")
    print(f"Task ID: {result['task_id']}")

asyncio.run(main())
```

### Component Initialization

#### ExpertRegistry

```python
from vision_forge import ExpertRegistry, load_expert_config

# Load from directory
registry = ExpertRegistry.from_directory("experts/static")

# Load additional dynamic experts
registry.load_additional_configs("experts/dynamic")

# Get expert IDs
expert_ids = registry.get_expert_ids()

# Check if expert exists
has_expert = registry.has_expert("visual_expert_01")

# Get expert config
config = registry._expert_configs.get("visual_expert_01")

# Create expert instance
expert = registry.create_instance(
    expert_id="visual_expert_01",
    blackboard=blackboard,
    model_router=router,
    as_executor=True  # or False for critic mode
)
```

#### SharedBlackboard

```python
from vision_forge import SharedBlackboard, BlackboardEvent, EventType

# Initialize
blackboard = SharedBlackboard(persist_dir="output/blackboard")

# Append event
event = BlackboardEvent(
    type=EventType.OPINION_ADDED,
    expert_id="expert_01",
    task_id="task-123",
    data={"opinion": "Good quality", "score": 0.9},
    round_number=1,
    timestamp=int(time.time() * 1000)
)
await blackboard.append(event)

# Query events
all_events = blackboard.get_events(task_id="task-123")
opinions = blackboard.get_opinions("task-123")
scores = blackboard.get_scores("task-123")
current_round = blackboard.get_current_round("task-123")

# Subscribe to events
def on_event(event):
    print(f"New event: {event.type}")

blackboard.subscribe(on_event)

# Get statistics
stats = blackboard.get_stats()
```

#### MemoryManager

```python
from vision_forge import MemoryManager

# Initialize
memory = MemoryManager(compression_threshold=1000)

# Store memory
await memory.store(
    key="user_preference",
    value={"style": "minimalist"},
    scope="session",  # task, session, or persistent
    task_id="task-123",  # Required for task scope
    importance=0.8
)

# Retrieve memory
value = await memory.retrieve(
    key="user_preference",
    scope="session",
    task_id="task-123"
)

# Search memory
results = memory.search(pattern="preference", scope="session")

# Clear task memory
await memory.clear_task_memory("task-123")

# Clear session memory
await memory.clear_session_memory()

# Get statistics
stats = memory.get_stats()
```

#### ModelRouter

```python
from vision_forge import ModelRouter, TaskType

# Initialize from environment
router = ModelRouter.from_env()

# Get best model for task type
model_name, service = router.get_best_model(TaskType.TEXT_REASONING)

# Route request
response = await router.route_request(
    task_type=TaskType.TEXT_REASONING,
    prompt="Analyze this image",
    temperature=0.7
)

# Estimate cost
cost = router.estimate_cost(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500
)

# Get available models
models = router.get_available_models()
```

### Expert Usage

#### Direct Expert Calls

```python
from vision_forge.experts.visual import VisualExpert
from vision_forge import load_expert_config

# Load config
config = load_expert_config("experts/static/visual.yml")

# Create expert
expert = VisualExpert(config, blackboard, router)

# Set confidence
expert.set_confidence(0.9)

# Process task
result = await expert.process({
    "task_id": "task-123",
    "task_type": "image_editing",
    "data": {
        "prompt": "Enhance the lighting",
        "images": [image_bytes]
    }
})

# Get history
history = expert.get_history()
```

#### Publish Opinions and Scores

```python
# Publish opinion
await expert.publish_opinion(
    task_id="task-123",
    opinion="The composition follows rule of thirds",
    score=0.9,
    round_number=1
)

# Publish score for another expert
await expert.publish_score(
    task_id="task-123",
    target_expert_id="photographer_01",
    score=0.85,
    rationale="Good analysis but missed shadow consistency"
)
```

### Workflow Orchestration

#### Basic Workflow

```python
async def process_image():
    # Initialize
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(
        registry, blackboard, memory, router
    )

    # Process request
    result = await orchestrator.process_request(
        user_prompt="Generate a product photo for a lamp",
        images=[image_bytes],
        context={"style": "minimalist"}
    )

    # Handle result
    if result["status"] == "success":
        print(f"Task completed: {result['task_id']}")
        print(f"Trace: {result['trace_path']}")
    elif result["status"] == "rejected":
        print(f"Rejected: {result['violations']}")
    else:
        print(f"Error: {result['error']}")
```

#### Custom Expert Selection

```python
async def custom_workflow():
    # Initialize components
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(
        registry, blackboard, memory, router
    )

    # Create task
    task_id = "custom-task-123"
    task = Task(
        id=task_id,
        type="interior_design",
        status=TaskStatus.PROCESSING,
        data={"prompt": "Redesign living room"},
        created_at=int(time.time()),
        updated_at=int(time.time())
    )

    # Select custom experts
    selected_experts = [
        "project_manager_01",
        "interior_designer_01",
        "photographer_01"
    ]

    # Execute experts
    results = await orchestrator._execute_experts(task, selected_experts)
```

---

## Configuration Options

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_MODEL_NAME=gpt-4o
OPENAI_API_VERSION=2024-12-01-preview

# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key
GEMINI_MAX_OUTPUT_TOKENS=8192

# Vertex AI Configuration
VERTEX_PROJECT_ID=your_project_id
VERTEX_LOCATION=us-central1
VERTEX_CREDENTIALS_PATH=path/to/credentials.json
```

### YAML Configuration Files

#### Rate Limits (`sys_init/settings/rate_limits.yml`)

```yaml
azure_openai:
  rpm_limit: 60           # Requests per minute
  tpm_limit: 500000       # Tokens per minute
  burst_size: 10          # Token bucket capacity

vertex_ai:
  rpm_limit: 30
  tpm_limit: 300000
  image_gen_rpm_limit: 10 # Stricter for image generation
```

#### Retry Policy (`sys_init/settings/fallback_policy.yml`)

```yaml
retry:
  max_attempts: 3
  backoff_multiplier: 2.0     # Exponential backoff multiplier
  initial_delay_ms: 1000      # Initial delay 1 second
  max_delay_ms: 30000         # Max delay 30 seconds
  jitter: 0.1                 # Random jitter 10%
```

#### Model Routing (`sys_init/settings/model_routing.yml`)

```yaml
allowed_models:
  azure:
    - gpt-4o
    - gpt-4o-mini
    - o1-preview
    - o1-mini
  vertex:
    - gemini-2.0-pro
    - gemini-2.0-flash
    - gemini-3.0-pro
    - gemini-3.0-flash

prohibited_models:
  - gpt-4
  - gpt-4-32k
  - gpt-4-turbo
  - gemini-1.0
  - gemini-1.5
```

#### Intervention Rules (`sys_init/settings/intervention_rules.yml`)

```yaml
confidence_thresholds:
  low_confidence_threshold: 0.5    # Notify PM if below
  critical_low_threshold: 0.3      # Auto-trigger HR recruitment
  high_confidence_threshold: 0.8   # Bonus weight

discussion_control:
  attention_round: 3       # PM starts monitoring
  warning_round: 5         # PM warning
  intervention_round: 8    # PM suggests intervention
  voting_round: 10         # Force voting
  emergency_round: 15      # PM direct decision
```

#### Fallback Policy (`sys_init/settings/fallback_policy.yml`)

```yaml
violation_categories:
  sensitive_content:
    severity: high
    auto_fallback: false
  copyright_ip:
    severity: medium
    auto_fallback: true
    max_attempts: 3
  brand_trademark:
    severity: medium
    auto_fallback: true
    max_attempts: 2
  celebrity_likeness:
    severity: medium
    auto_fallback: true
    max_attempts: 2
  quality_below_target:
    severity: low
    auto_fallback: true
    max_attempts: 2
```

---

## Best Practices

### Prompt Engineering

1. **Be Specific**: Clear, detailed prompts yield better results
2. **Include Context**: Specify target audience, use case, constraints
3. **Reference Examples**: Mention style references when applicable
4. **Iterate**: Use feedback from initial results to refine prompts

**Good Prompt:**
```
Generate a professional product photo for a luxury watch on a marble pedestal
with soft, diffused lighting. Style should be minimalist and premium,
targeting high-end consumers. The watch should be the focal point with
shallow depth of field.
```

**Bad Prompt:**
```
Make a watch image
```

### Expert Selection

1. **Match Domain**: Select experts based on task domain
2. **Balance Team**: Include both executor and critic perspectives
3. **Consider Load**: On-demand experts only when needed
4. **Monitor Confidence**: Low confidence may indicate need for additional experts

### Memory Management

1. **Use Appropriate Scope**:
   - Task scope: Temporary data for single task
   - Session scope: Data needed across tasks in session
   - Persistent scope: Long-term knowledge

2. **Set Importance**: Higher importance = more likely to persist
3. **Search Before Store**: Check if memory already exists
4. **Clean Up**: Clear task memory after completion

### Blackboard Usage

1. **Publish Early**: Share opinions as soon as formed
2. **Be Specific**: Include detailed rationale
3. **Score Constructively**: Provide reasoning for scores
4. **Review History**: Check blackboard before forming opinions

### Error Handling

```python
try:
    result = await orchestrator.process_request(...)
except Exception as e:
    # Log error
    logger.error(f"Processing failed: {e}")

    # Check task status
    if result["status"] == "rejected":
        # Handle compliance rejection
        review_violations(result["violations"])
    elif result["status"] == "error":
        # Handle system error
        retry_or_fallback(result["error"])
```

### Performance Optimization

1. **Batch Requests**: Group similar requests when possible
2. **Use Caching**: Check memory before making new requests
3. **Parallel Execution**: Independent experts can run in parallel
4. **Monitor Rate Limits**: Respect API.py limits to avoid throttling

---

## Examples

### Example 1: Product Image Generation

```python
import asyncio
from vision_forge import WorkflowOrchestrator, ExpertRegistry, SharedBlackboard, MemoryManager, ModelRouter

async def generate_product_image():
    # Initialize
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

    # Process request
    result = await orchestrator.process_request(
        user_prompt="""
        Generate a professional product photo for a luxury Swiss watch.
        The watch should be placed on a black marble pedestal with soft,
        diffused lighting from the left. Background should be gradient
        from dark gray to black. Style: premium, minimalist, commercial.
        Target audience: high-net-worth individuals aged 35-55.
        """,
        images=[],
        context={
            "product_category": "luxury_goods",
            "brand_positioning": "premium",
            "usage": "e-commerce"
        }
    )

    return result

# Run
result = asyncio.run(generate_product_image())
```

### Example 2: Interior Design

```python
async def redesign_interior():
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

    # Load input image
    with open("living_room.jpg", "rb") as f:
        image_bytes = f.read()

    result = await orchestrator.process_request(
        user_prompt="""
        Redesign this living room in Scandinavian minimalist style.
        Requirements:
        - Light color palette (whites, light grays, natural wood)
        - Minimalist furniture with clean lines
        - Maximize natural light
        - Add indoor plants for warmth
        - Maintain current room dimensions
        """,
        images=[image_bytes],
        context={
            "style": "scandinavian_minimalist",
            "budget_range": "mid_to_high",
            "room_type": "living_room"
        }
    )

    return result
```

### Example 3: Image Editing with Mask

```python
async def edit_with_mask():
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

    # Load image and mask
    with open("photo.jpg", "rb") as f:
        image_bytes = f.read()
    with open("mask.png", "rb") as f:
        mask_bytes = f.read()

    result = await orchestrator.process_request(
        user_prompt="""
        Remove the person from the background (indicated by mask) and
        replace with a sunset sky. Enhance the warm tones of the sunset
        and ensure seamless blending with the foreground.
        """,
        images=[image_bytes],
        context={
            "mask": mask_bytes,
            "preserve_foreground": True,
            "blend_mode": "seamless"
        }
    )

    return result
```

### Example 4: Batch Processing

```python
async def batch_process(products):
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

    results = []
    for product in products:
        result = await orchestrator.process_request(
            user_prompt=f"Generate product photo for {product['name']}",
            images=product.get("images", []),
            context={
                "category": product["category"],
                "style": product["style"]
            }
        )
        results.append({
            "product_id": product["id"],
            "task_id": result.get("task_id"),
            "status": result["status"]
        })
        # Respect rate limits
        await asyncio.sleep(1)

    return results
```

---

## Troubleshooting

### Common Issues

#### "No service available"

**Cause**: Missing environment configuration

**Solution**:
```bash
# Check .env file exists
cp .env.example .env

# Verify environment variables
python -m vision_forge config

# Test connectivity
python -m vision_forge test
```

#### "Compliance violation"

**Cause**: Prompt or output violates content policy

**Solution**:
1. Review violation details in response
2. Modify prompt to remove problematic elements
3. Re-submit with adjusted request

#### "Rate limit exceeded"

**Cause**: Too many requests in short time

**Solution**:
```python
# Add delays between requests
await asyncio.sleep(1)  # 1 second between requests

# Or configure rate limits in rate_limits.yml
```

#### "Expert not found"

**Cause**: Expert not registered or YAML config missing

**Solution**:
```bash
# Check expert configs exist
ls experts/static/

# Verify YAML syntax
python -c "from vision_forge import load_expert_config; load_expert_config('experts/static/visual.yml')"
```

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vision_forge")
```

### Check Task Trace

```bash
# View task trace
cat output/trace/<task_id>.json

# View blackboard events
cat output/blackboard/<task_id>.jsonl
```

### Reset State

```python
# Clear all memory
await memory.clear_session_memory()

# Clear blackboard
blackboard.clear(task_id)

# Reset registry
registry.clear_instances()
```

---

## Support

For additional help:
- Documentation: `docs/` directory
- Issues: GitHub Issues
- Discussions: GitHub Discussions
