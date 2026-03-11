# Vision Forge

**Intelligent Vision Reviewer** - An expert system for AI-powered image evaluation and optimization.

Vision Forge receives images + natural language requirements + reference information (mask/bounding box, etc.) and outputs high-quality image results + review reports + traceable operation links.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 52 passing](https://img.shields.io/badge/tests-52%20passing-green.svg)](https://github.com/godlockin/vision-forge)

## Features

- **Expert System**: 10 static experts + dynamic expert generation for specialized tasks
- **Blackboard Collaboration**: Append-only event sourcing for full traceability
- **Compliance Veto**: One-vote veto power for compliance/legal review at entry and delivery
- **Weighted Consensus**: Feasibility × Confidence × Peer scoring for decision making
- **Auto-Fallback**: Automatic remediation for fixable compliance violations (max 3 attempts)
- **Operation Link Tracing**: All intermediate operations recorded for user fork/continuation
- **Multi-Model Support**: Azure OpenAI + Google VertexAI with intelligent routing
- **Rate Limiting & Retry**: Token bucket rate limiting + exponential backoff retry

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

## Architecture

Vision Forge uses a layered architecture with intelligent routing and comprehensive error handling:

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

## Workflow

Vision Forge follows a 7-step process with built-in quality gates:

```
1. [Veto Check] Compliance review of user request → Reject if violations
2. Task Init → PM + Visual Expert + HR select experts → Feasibility scoring
3. Expert Assignment → Experts self-assess confidence (Confidence Score)
4. Faction Split → Executor group vs Critic group → Serial/Parallel execution
5. Blackboard Discussion → Weighted peer scoring → Score × (Feasibility × Confidence)
6. [Forced Convergence] Discussion exceeds threshold → PM organizes voting
7. [Veto Check] Final compliance review → Deliver + Operation link recording
```

## Available Experts

Vision Forge includes 10 static experts, each with specific responsibilities:

| Expert | Responsibility | Load Strategy |
|--------|----------------|---------------|
| **Project Manager** | Task decomposition, progress tracking, deadlock resolution | Always loaded |
| **Compliance & Legal** | One-vote veto for content safety and IP risk | Always loaded |
| **Visual Expert** | Core image execution and quality review | Always loaded |
| **Knowledge Admin** | Blackboard management, memory compression | Always loaded |
| **Senior HR Expert** | Capability gap analysis, dynamic expert recruitment | On-demand |
| **Prompt Engineer** | Ambiguous requirement clarification, cross-model adaptation | On-demand |
| **Photographer** | Lighting/composition analysis, photorealism assessment | On-demand |
| **Image Editor** | Complex edit decomposition, mask generation | On-demand |
| **Interior Designer** | Spatial layout, style matching, furniture planning | On-demand |
| **Display Designer** | Product display optimization, visual merchandising | On-demand |

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

Vision Forge includes a comprehensive test suite with 52 tests covering models, services, memory, experts, and integration scenarios.

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

| Module | Tests | Coverage |
|--------|-------|----------|
| Core Models | 14 | 100% |
| Config Loader | 8 | 100% |
| Memory System | 10 | 95% |
| Experts | 12 | 90% |
| Integration | 8 | 85% |

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
