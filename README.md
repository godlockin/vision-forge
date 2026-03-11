# Vision Forge

**Intelligent Vision Reviewer** - An expert system for AI-powered image evaluation and optimization.

Vision Forge receives images + natural language requirements + reference information (mask/bounding box, etc.) and outputs high-quality image results + review reports + traceable operation links.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 99 passing](https://img.shields.io/badge/tests-99%20passing-green.svg)](https://github.com/godlockin/vision-forge)

## Features

- **Expert System**: 10 static experts + dynamic expert generation for specialized tasks
- **Blackboard Collaboration**: Append-only event sourcing for full traceability
- **Compliance Veto**: One-vote veto power for compliance/legal review at entry and delivery
- **Weighted Consensus**: Feasibility × Confidence × Peer scoring for decision making
- **Auto-Fallback**: Automatic remediation for fixable compliance violations (max 3 attempts)
- **Operation Link Tracing**: All intermediate operations recorded for user fork/continuation
- **Multi-Model Support**: Azure OpenAI + Google VertexAI with intelligent routing
- **Rate Limiting & Retry**: Token bucket rate limiting + exponential backoff retry

---

## Vision

**To empower creators and businesses with AI-driven visual intelligence that combines human-like expertise with machine precision.**

Vision Forge envisions a future where:
- Every image creation task benefits from collective expert wisdom
- Quality assessment is objective, transparent, and traceable
- AI-generated content meets professional standards before publication
- The gap between human creative intent and AI output is seamlessly bridged

---

## Mission

**Deliver production-grade AI visual review systems that are explainable, auditable, and continuously evolving.**

### Core Objectives

1. **Professional-Grade Quality**: Match or exceed human expert review standards
2. **Transparent Decision-Making**: Every recommendation backed by traceable reasoning
3. **Continuous Evolution**: System grows smarter with each interaction through dynamic expert generation
4. **Enterprise-Ready**: Built for production with rate limiting, retry, and compliance safeguards

---

## Design Philosophy

Vision Forge is built on six immutable engineering principles:

### 1. Type Safety: `any` is a Four-Letter Word

```python
# ❌ DESTROYS VALUE
data: any = await fetch_image()
print(data.nmae)  # Compiles, crashes at runtime

# ✓ PRESERVES VALUE
class ImageResult(BaseModel):
    url: str
    width: int
    height: int
data: ImageResult = await fetch_image()
print(data.nmae)  # ERROR: caught at compile time
```

### 2. Memoize or Die: O(N) → O(1)

```python
# ❌ O(N) ON EVERY RENDER
filtered = items.filter(predicate)

# ✓ O(1) AVERAGE CASE
@lru_cache(maxsize=128)
def filter_cached(items_tuple, predicate_hash):
    return [i for i in items_tuple if predicate(i)]
```

### 3. Defensive Error Handling: Assume Everything Fails

```python
# ❌ LOSES CONTEXT
except Exception as e:
    console.error(e)
    throw e

# ✓ PRESERVES CONTEXT
except Exception as e:
    logger.error({
        "msg": "API call failed",
        "code": getattr(e, "code", "unknown"),
        "request_id": request_id,
        "stack": traceback.format_exc()
    })
    raise APIError(f"Image processing failed: {str(e)}")
```

### 4. Fail Fast: Crash at Startup, Not at 2 AM

```python
# ❌ DEFERS FAILURE
api_key = os.getenv("API_KEY")  # undefined?
client = APIClient(api_key)  # Works, fails later

# ✓ FAILS IMMEDIATELY
def validate_config():
    errors = []
    if not os.getenv("API_KEY"): errors.append("Missing API_KEY")
    if not os.getenv("VERTEX_PROJECT_ID"): errors.append("Missing VERTEX_PROJECT_ID")
    if errors:
        raise ValueError(f"Startup failed:\n{chr(10).join(errors)}")
```

### 5. DRY is Law: Single Source of Truth

```python
# ❌ DUPLICATED - fix in N places
def calculate_tax_us(price): return price * 0.08
def calculate_tax_eu(price): return price * 0.20

# ✓ CENTRALIZED
TAX_RATES = {"US": 0.08, "EU": 0.20, "UK": 0.20}
def calculate_tax(region, price): return price * TAX_RATES[region]
```

### 6. Immutability by Default

```python
# ❌ MUTABLE - side effects everywhere
global_config = {"theme": "dark"}
def change_theme(t): global_config["theme"] = t

# ✓ IMMUTABLE - explicit state transitions
@dataclass(frozen=True)
class Config:
    theme: str = "dark"
config_a = Config()
config_b = Config(theme="light")  # Original unchanged
```

---

## Operations

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Expert Application Layer                  │
│    (PM / Visual Expert / Compliance / HR / 10 Static Experts)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Request Scheduling Layer                    │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │  Model Router   │  │  Rate Limiter   │  │  Retry Handler  │ │
│   │  (Smart Route)  │  │  (Token Bucket) │  │  (Backoff)      │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Adaptation Layer                      │
│   ┌─────────────────┐              ┌─────────────────┐          │
│   │  Azure OpenAI   │              │  Vertex AI SDK  │          │
│   │  Service        │              │  (Gemini)       │          │
│   └─────────────────┘              └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Cloud Model Layer                           │
│   GPT-4o / GPT-4o-mini     │     Gemini 2.5/3.0 Pro/Flash       │
│   o1-preview / o1-mini     │     Imagen-3 / Imagen-3-fast       │
└─────────────────────────────────────────────────────────────────┘
```

### 7-Step Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: [一票否决] Compliance Review                              │
│ - Check user request for legal/IP/safety issues                  │
│ - Veto power: stops processing if violations found               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Task Init + Expert Selection                             │
│ - PM + Visual Expert + HR analyze requirements                   │
│ - Feasibility scoring for each potential expert                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Expert Assignment                                        │
│ - Selected experts self-assess confidence (0.0-1.0)              │
│ - Confidence Score recorded for weighted voting                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Faction Split                                            │
│ - Executors: experts who will do the work                        │
│ - Critics: experts who will review and challenge                 │
│ - Dependencies determine serial vs parallel execution            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Blackboard Discussion                                    │
│ - Weighted peer scoring: Score × (Feasibility × Confidence)      │
│ - Append-only event sourcing for full traceability               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: [强制打断] Deadlock Resolution                            │
│ - PM detects when discussion exceeds threshold                   │
│ - Organizes weighted voting to converge                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 7: [一票否决] Final Compliance + Delivery                    │
│ - Final compliance review before delivery                        │
│ - Auto-fallback for fixable violations (max 3 attempts)          │
│ - Operation link recording for traceability                      │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Architecture (Three Layers)

| Layer | Scope | Content | Retention |
|-------|-------|---------|-----------|
| **Task Memory** | Single task | Intermediate states,临时 decisions | Cleared on task completion |
| **Session Memory** | Single session | Expert interaction history | Cleared on session end |
| **Persistent Memory** | Global shared | Task history, user preferences, domain knowledge, expert index | Compressed and retained permanently |

### Key Design Decisions

1. **Feasibility Scoring + Confidence Self-Assessment**: Task allocation evaluates expert capabilities; experts self-rate confidence after assignment, weighted into final decision

2. **One-Vote Veto Power**: Compliance expert holds不可覆盖 veto at both intake and delivery nodes

3. **Operation Link Traceability**: All intermediate operations recorded as fork-able chain

4. **Dynamic Expert Registration**: New experts indexed to expert library; system capabilities evolve continuously

5. **API Rate Limiting & Retry**: Token bucket + exponential backoff prevents API rate limit issues

6. **Blackboard Append-Only**: Experts can only append events, never modify history; Knowledge Admin compresses periodically

7. **Model Smart Routing**: Dynamic selection based on task type; GPT-4 and pre-2.5 Gemini prohibited

8. **Auto-Fallback**: Compliance violations auto-remediated (max 3 attempts); rejected with detailed report on failure

## Installation

```bash
# Clone repository
git clone https://github.com/godlockin/vision-forge.git
cd vision-forge

# Install dependencies
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

### Requirements

- Python 3.11+
- Azure OpenAI API key (for GPT-4o, GPT-4o-mini, o1 series)
- Google Cloud project with VertexAI enabled (for Gemini, Imagen-3)

## Quick Start

### Command Line Interface

```bash
# Review/generate image with prompt
python -m vision_forge review -p "Generate a professional product photo for a lamp"

# With input image for editing
python -m vision_forge review -p "Enhance the lighting in this image" -i input.jpg

# List available experts
python -m vision_forge experts

# Check configuration status
python -m vision_forge config

# Test LLM connectivity
python -m vision_forge test

# Show system statistics
python -m vision_forge stats
```

### Programmatic Usage

```python
import asyncio
from vision_forge import WorkflowOrchestrator, ExpertRegistry, SharedBlackboard, MemoryManager, ModelRouter

async def main():
    # Initialize components
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard(persist_dir="output/blackboard")
    memory = MemoryManager(compression_threshold=1000)
    router = ModelRouter.from_env()

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

    # Process request
    result = await orchestrator.process_request(
        user_prompt="Generate a product image for a luxury watch",
        images=[image_bytes],  # Optional input images
        context={"target_audience": "premium consumers"}
    )

    print(f"Status: {result['status']}")
    print(f"Task ID: {result['task_id']}")
    print(f"Duration: {result.get('duration_sec', 0):.2f}s")

asyncio.run(main())
```

## Available Experts

Vision Forge includes 10 static experts, each with specific responsibilities and capabilities:

| Expert | ID | Responsibility | Load Strategy | Veto Power |
|--------|-----|----------------|---------------|------------|
| **Project Manager** | `project_manager_01` | Task decomposition, progress tracking, deadlock resolution | Always | No |
| **Compliance & Legal** | `compliance_legal_01` | Content safety, IP risk, legal compliance | Always | **Yes** |
| **Visual Expert** | `visual_expert_01` | Core image execution, aesthetic scoring, quality review | Always | No |
| **Knowledge Admin** | `knowledge_admin_01` | Blackboard management, memory compression, knowledge indexing | Always | No |
| **Senior HR Expert** | `hr_expert_01` | Capability gap analysis, dynamic expert recruitment, team optimization | On-demand | No |
| **Prompt Engineer** | `prompt_engineer_01` | Ambiguous requirement clarification, cross-model prompt adaptation | On-demand | No |
| **Photographer** | `photographer_01` | Lighting/composition analysis, photorealism assessment | On-demand | No |
| **Image Editor** | `image_editor_01` | Complex edit decomposition, mask generation, editing approach | On-demand | No |
| **Interior Designer** | `interior_designer_01` | Spatial layout, style matching, furniture placement | On-demand | No |
| **Display Designer** | `display_designer_01` | Product display optimization, visual merchandising, lighting setup | On-demand | No |

### Expert Capabilities Matrix

| Expert | Key Capabilities |
|--------|------------------|
| PM | `gap_analysis`, `weighted_vote`, `deadlock_detection`, `force_decision` |
| Compliance | `review_request`, `review_output`, `can_autofallback`, `build_rejection_report` |
| Visual | `aesthetic_scoring`, `defect_detection`, `decompose_edit_task`, `execute_image_edit` |
| HR | `gap_analysis`, `create_expert_profile`, `feasibility_score`, `team_optimization` |
| Knowledge Admin | `compress_context`, `generate_summary`, `index_knowledge`, `cleanup_blackboard` |
| Prompt Engineer | `clarify_requirement`, `adapt_for_model`, `inject_style`, `inject_constraints` |
| Photographer | `analyze_lighting`, `analyze_composition`, `assess_realism` |
| Image Editor | `decompose_complex_edit`, `generate_mask_description`, `suggest_editing_approach` |
| Interior Designer | `analyze_spatial_layout`, `style_matching`, `suggest_furniture_placement` |
| Display Designer | `optimize_product_display`, `visual_marketing_analysis`, `suggest_lighting_setup` |

## Configuration

### Environment Variables

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_MODEL_NAME=gpt-4o
OPENAI_API_VERSION=2024-12-01-preview

# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key

# Vertex AI Configuration
VERTEX_PROJECT_ID=your_project_id
VERTEX_LOCATION=us-central1
VERTEX_CREDENTIALS_PATH=path/to/credentials.json
```

### Allowed Models

**Text Models:**
- Azure OpenAI: `gpt-4o`, `gpt-4o-mini`, `o1-preview`, `o1-mini`
- Vertex AI: `gemini-2.0-pro`, `gemini-2.0-flash`, `gemini-3.0-pro`, `gemini-3.0-flash`

**Image Models:**
- Vertex AI: `imagen-3.0-generate-001`, `imagen-3.0-fast-generate-001`

**Prohibited Models:**
- GPT-4 and earlier versions (GPT-4-32k, GPT-4-turbo, GPT-35-turbo)
- Gemini versions before 2.5 (Gemini 1.0, Gemini 1.5 series)

## Project Structure

```
vision-forge/
├── vision_forge/                 # Main package
│   ├── __init__.py               # Public API exports
│   ├── cli.py                    # Command line interface
│   ├── core/                     # Core components
│   │   ├── __init__.py
│   │   ├── models.py             # Pydantic data models
│   │   ├── config_loader.py      # YAML config loader
│   │   ├── fallback.py           # Auto-fallback handler
│   │   └── orchestrator.py       # Workflow orchestrator
│   ├── experts/                  # Expert implementations
│   │   ├── __init__.py
│   │   ├── expert.py             # Base expert class
│   │   ├── expert_registry.py    # Expert registry
│   │   ├── pm.py                 # Project Manager expert
│   │   ├── compliance.py         # Compliance expert
│   │   ├── visual.py             # Visual expert
│   │   ├── knowledge_admin.py    # Knowledge administrator
│   │   ├── hr.py                 # HR expert
│   │   ├── prompt_engineer.py    # Prompt engineer
│   │   ├── photographer.py       # Photographer expert
│   │   ├── image_editor.py       # Image editor
│   │   ├── interior_designer.py  # Interior designer
│   │   ├── display_designer.py   # Display designer
│   │   └── dynamic_expert_generator.py
│   ├── memory/                   # Memory management
│   │   ├── __init__.py
│   │   ├── events.py             # Blackboard events
│   │   ├── blackboard.py         # Shared blackboard
│   │   ├── manager.py            # Memory manager
│   │   └── compression.py        # Memory compression
│   └── services/                 # LLM services
│       ├── __init__.py
│       ├── base.py               # Base service interface
│       ├── router.py             # Model router
│       ├── rate_limiter.py       # Rate limiting
│       ├── retry.py              # Retry logic
│       ├── azure_openai.py       # Azure OpenAI service
│       └── vertex_ai.py          # Vertex AI service
├── experts/                      # Expert YAML configs
│   ├── static/                   # 10 static experts
│   │   ├── pm.yml
│   │   ├── compliance.yml
│   │   ├── visual.yml
│   │   └── ...
│   └── dynamic/                  # Dynamically generated experts
├── sys_init/                     # System initialization
│   ├── settings/                 # Configuration files
│   │   ├── static_experts.yml    # Static expert definitions
│   │   ├── dynamic_expert_template.yml
│   │   ├── rate_limits.yml       # API rate limiting config
│   │   ├── model_routing.yml     # Model routing rules
│   │   ├── blackboard_events.yml # Event type definitions
│   │   ├── intervention_rules.yml
│   │   └── fallback_policy.yml   # Auto-fallback policies
│   └── draft_goal/               # Project documentation
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest fixtures
│   ├── test_models.py            # Model tests
│   ├── test_config_loader.py     # Config loader tests
│   ├── test_memory.py            # Memory system tests
│   ├── test_experts.py           # Expert tests
│   ├── test_expert_collaboration.py
│   └── test_integration.py       # Integration tests
├── output/                       # Generated outputs
│   ├── trace/                    # Execution traces
│   └── blackboard/               # Blackboard event logs
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # Architecture documentation
│   ├── API.md                    # API reference
│   ├── EXPERTS.md                # Expert documentation
│   └── USAGE.md                  # Usage guide
├── .env.example                  # Environment template
├── pyproject.toml                # Project configuration
├── setup.py                      # Setup script
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Testing

Vision Forge includes a comprehensive test suite with **99 tests** covering models, services, memory, experts, and integration scenarios.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vision_forge --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run integration tests only
pytest -m integration

# Run unit tests only
pytest -m "not integration"
```

### Test Coverage Summary

| Module | Tests | Status |
|--------|-------|--------|
| Core Models | 14 | ✅ 100% |
| Config Loader | 8 | ✅ 100% |
| Memory System | 16 | ✅ 95% |
| Experts | 20 | ✅ 90% |
| Expert Collaboration | 21 | ✅ 100% |
| Integration | 19 | ✅ 85% |
| **Total** | **99** | **✅ All Passing** |

## Usage Examples

### Example 1: Product Image Generation

```bash
python -m vision_forge review \
  -p "Generate a professional product photo for a luxury watch on a marble pedestal with soft lighting" \
  -o output/product_shoot
```

### Example 2: Interior Design

```bash
python -m vision_forge review \
  -p "Redesign this living room in Scandinavian style with minimalist furniture" \
  -i living_room.jpg \
  -v
```

### Example 3: Image Editing with Mask

```python
from vision_forge import WorkflowOrchestrator, ExpertRegistry, SharedBlackboard, MemoryManager, ModelRouter

async def edit_image():
    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()
    router = ModelRouter.from_env()

    orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

    result = await orchestrator.process_request(
        user_prompt="Remove the person from the background and enhance the sunset colors",
        images=[image_bytes],
        context={"mask": mask_bytes, "preserve_foreground": True}
    )

    return result
```

## API Reference

For detailed API documentation, see [docs/API.md](docs/API.md).

### Key Classes

- **`Expert`**: Abstract base class for all experts
- **`ExpertRegistry`**: Registry for expert classes and instances
- **`SharedBlackboard`**: Append-only event sourcing blackboard
- **`MemoryManager`**: Three-layer memory management (task/session/persistent)
- **`ModelRouter`**: Intelligent model selection based on task type
- **`WorkflowOrchestrator`**: Main workflow coordinator

## Operational Best Practices

### Prompt Engineering

1. **Be Specific**: Include details about lighting, composition, mood, and style
2. **Reference Examples**: Mention specific artists, photography styles, or design movements
3. **Define Constraints**: Specify aspect ratio, color palette, and technical requirements
4. **Iterate**: Use the blackboard trace to understand what worked and refine

### Expert Selection

| Task Type | Recommended Experts |
|-----------|---------------------|
| Product Photography | Visual Expert + Display Designer + Photographer |
| Interior Design | Visual Expert + Interior Designer + Prompt Engineer |
| Image Editing | Visual Expert + Image Editor + Compliance |
| Creative Generation | Visual Expert + Prompt Engineer + Photographer |
| Complex Projects | PM + Visual + HR + Domain Experts |

### Memory Management

- **Task Memory**: Automatically cleared; no action needed
- **Session Memory**: Review periodically for important decisions
- **Persistent Memory**: Knowledge Admin compresses automatically; review compressed summaries

### Troubleshooting

| Issue | Solution |
|-------|----------|
| API Rate Limit | Check rate_limiter stats; adjust RPM/TPM limits in config |
| Compliance Rejection | Review violation report; modify prompt or use auto-fallback |
| Deadlock Detected | PM will force decision; review blackboard trace for context |
| Poor Image Quality | Adjust prompt; use Visual Expert's aesthetic_scoring for feedback |
| Missing Experts | Check experts/ directory; verify YAML syntax |

---

## Design Principles

Vision Forge follows these core design principles:

1. **Type Safety**: No `any` types; use `unknown` + type guards
2. **Memoization**: O(N) → O(1) for expensive operations
3. **Defensive Error Handling**: All external calls wrapped with try/catch
4. **Fail Fast**: Validate environment variables at startup
5. **DRY (Don't Repeat Yourself)**: Single source of truth for all logic
6. **Immutability by Default**: Use `const`/frozen data structures

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- Azure OpenAI for GPT-4o and o1 series models
- Google VertexAI for Gemini and Imagen-3 models
- The AI research community for ongoing advancements in multimodal AI
