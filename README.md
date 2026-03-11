# Vision Forge

**Intelligent Vision Reviewer** - 一个全流程负责的图像操作 Agent 专家系统

接收用户「图片 + 自然语言需求 + 参考信息」，输出「满足要求的高质量图像结果 + 评审报告 + 可追溯的操作链路」。

## Features

- **10 位静态专家**：项目经理、合规专家、视觉专家、知识管理员、HR 专家、Prompt 工程师、摄影师、图像编辑师、室内设计师、陈列设计师
- **动态专家生成**：根据任务需求动态引入领域专家
- **黑板协作模式**：所有专家通过共享黑板进行协作和互评
- **执行/审核分裂**：专家组分裂为执行阵营和挑剔审核阵营
- **加权共识机制**：可行性打分 × 置信度 × 互评评分
- **一票否决权**：合规专家在准入和交付节点拥有否决权
- **Auto-Fallback**：合规违规自动补救（最多 3 次）
- **操作链路可追溯**：记录所有中间操作，支持用户 fork 继续处理

## Installation

```bash
# Clone repository
git clone https://github.com/godlockin/vision-forge.git
cd vision-forge

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

## Quick Start

```bash
# Review/generate image
python -m vision_forge review -p "Generate a professional product photo for a lamp"

# With input image
python -m vision_forge review -p "Enhance the lighting in this image" -i input.jpg

# List available experts
python -m vision_forge experts

# Check configuration
python -m vision_forge config

# Test LLM connectivity
python -m vision_forge test

# Show system statistics
python -m vision_forge stats
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        专家应用层                                │
│    (PM / 视觉专家 / 合规专家 / HR / ... 10 位静态专家)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      请求调度层                                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │  模型智能路由    │  │  Token 桶限流    │  │  指数退避重试   │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      模型适配层                                  │
│   ┌─────────────────┐              ┌─────────────────┐          │
│   │  Azure OpenAI   │              │  Vertex AI SDK  │          │
│   └─────────────────┘              └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      云端模型层                                  │
│   GPT-4o / GPT-4o-mini     │     Gemini 2.5/3.0 Pro/Flash       │
│   o1-preview / o1-mini     │     Imagen-3 / Imagen-3-fast       │
└─────────────────────────────────────────────────────────────────┘
```

## Workflow

```
1. [一票否决] 合规审查用户需求 → 不通过则驳回
2. 任务初始化 → PM+ 视觉专家+HR 决定引入哪些专家 → 可行性打分
3. 专家接单 → 自评置信度 (Confidence Score)
4. 阵营分裂 → 执行组 vs 审核组 → 依赖串行/无依赖并行
5. 黑板讨论 → 加权互评 → 评分 × (可行性 × 置信度)
6. [强制打断] 讨论超阈值 → PM 组织投票收敛
7. [一票否决] 最终合规审查 → 交付 + 操作链路记录
```

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
- Azure OpenAI: GPT-4o, GPT-4o-mini, o1-preview, o1-mini
- Vertex AI: Gemini 2.5/3.0 Pro/Flash

**Image Models:**
- Vertex AI: Imagen-3, Imagen-3-fast

**Prohibited Models:**
- GPT-4 and earlier versions
- Gemini versions before 2.5

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=vision_forge

# Run specific test file
pytest tests/test_models.py -v

# Run integration tests only
pytest -m integration
```

## Project Structure

```
vision-forge/
├── vision_forge/                 # Main package
│   ├── __init__.py
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
│   │   └── compliance.py         # Compliance expert
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
│   └── static/                   # 10 static experts
├── tests/                        # Test suite
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_config_loader.py
│   ├── test_memory.py
│   └── test_experts.py
├── output/                       # Generated outputs
│   ├── trace/                    # Execution traces
│   └── blackboard/               # Blackboard events
├── .env.example                  # Environment template
├── pyproject.toml
├── setup.py
└── README.md
```

## License

MIT License
