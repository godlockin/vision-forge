# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Intelligent Vision Reviewer** - 一个全流程负责的图像操作 Agent 专家系统，接收用户「图片 + 自然语言需求 + 参考信息 (mask/boundingbox 等)」，输出「满足要求的高质量图像结果 + 评审报告 + 可追溯的操作链路」。

## Core Design Principles

1. **混合评估模式** - 规则检查（客观）+ LLM 评估（主观）+ 用户确认
2. **黑板协作模式** - 所有专家通过共享黑板进行协作和互评
3. **执行/审核分裂** - 专家组分裂为执行阵营和挑剔审核阵营
4. **加权共识机制** - 可行性打分 × 置信度 × 互评评分
5. **操作链路可追溯** - 记录所有中间操作，支持用户 fork 继续处理

## Architecture

### Expert System (专家系统)

**10 位静态专家** (配置于 `sys_init/settings/static_experts.yml`):

| 专家 | 职责 | 加载策略 |
|-----|------|---------|
| 项目经理 | 任务拆解、进度把控、死锁打断 | 常驻 |
| 合规与法律专家 | 一票否决权，准入 + 交付审查 | 常驻 |
| 视觉专家 | 核心执行与审核 | 常驻 |
| 知识管理员 | 共享黑板管理、记忆压缩 | 常驻 |
| 资深 HR 专家 | 能力缺口识别、动态专家招聘 | 按需 |
| Prompt 工程师 | 模糊需求转化、跨模型适配 | 按需 |
| 摄影师 | 光线/构图分析、真实感评估 | 按需 |
| 图像编辑师 | 复杂编辑拆解、mask 生成 | 按需 |
| 室内设计师 | 空间布局、风格搭配 | 按需 |
| 陈列设计师 | 商品展示、视觉营销 | 按需 |

**动态专家** (模板见 `sys_init/settings/dynamic_expert_template.yml`):
- 由 HR 专家主导，领域相关静态专家配合
- LLM 基于模板根据领域描述填充完整配置
- 生成后进行可行性打分，通过后注册到专家索引库

### Workflow (7 步流转)

```
1. [一票否决] 合规审查用户需求 → 不通过则驳回
2. 任务初始化 → PM+ 视觉专家+HR 决定引入哪些专家 → 可行性打分
3. 专家接单 → 自评置信度 (Confidence Score)
4. 阵营分裂 → 执行组 vs 审核组 → 依赖串行/无依赖并行
5. 黑板讨论 → 加权互评 → 评分 × (可行性 × 置信度)
6. [强制打断] 讨论超阈值 → PM 组织投票收敛
7. [一票否决] 最终合规审查 → 交付 + 操作链路记录
```

### Memory Architecture (三层记忆)

| 层次 | 范围 | 内容 | 保留策略 |
|-----|------|------|---------|
| 任务记忆 | 单任务 | 中间状态、临时决策 | 任务结束清除 |
| 会话记忆 | 单 session | 专家交互记录 | Session 结束清除 |
| 持久记忆 | 全局共享 | 任务历史、用户偏好、领域知识、专家索引 | 压缩后永久保存 |

### Technical Stack

**架构分层**:
```
专家应用层 → 请求调度层 (限流/路由/重试) → 模型适配层 → 云端模型层
```

**关键技术点**:
- **大模型层**: Azure OpenAI SDK + Google VertexAI SDK (多模型消除偏见)
- **模型路由**: 根据任务类型动态选择最合适模型
  - 允许：GPT-4o 系列、Gemini 2.5/3.0 Pro/Flash、Imagen-3
  - 禁止：GPT-4 及更早版本、Gemini 2.5 之前版本
- **限流重试**: Token 桶限流 + 指数退避重试，防止 API Rate Limit
- **并发层**: Python 多线程/多进程混合框架
- **黑板控制**: Append-only Event Sourcing + 知识管理员定期压缩
- **存储层**: 本地文件系统 + SQLite/JSON + 向量数据库 (可选)
- **媒体内核**: VertexAI 调用 Gemini/Imagen 生图/生视频模型
- **部署模式**: 专家系统本地运行，大模型云端 API 调用
- **Auto-Fallback**: 合规违规自动补救 (最多 3 次) + 驳回报告

## Project Structure

```
intelligent_vision_reviewer/
├── CLAUDE.md                              # 本文件
├── sys_init/
│   ├── auth/                              # GCP 认证配置
│   │   ├── .env
│   │   └── gcp-auth.json
│   ├── draft_goal/                        # 项目原始目标文档
│   │   └── goal_settings.md
│   └── settings/                          # 专家系统配置与文档
│       ├── expert_system_architecture.md  # 完整架构设计文档
│       ├── static_experts.yml             # 10 位静态专家 YAML 配置
│       ├── dynamic_expert_template.yml    # 动态专家生成模板
│       ├── context_compression_summary.md # 上下文压缩摘要
│       ├── session_history.md             # 会话历史记录与项目理解
│       ├── DESIGN_STATUS.md               # 设计文档状态总览
│       ├── rate_limits.yml                # API 限流、重试、降级配置
│       ├── model_routing.yml              # 模型白名单、任务路由、成本优化
│       ├── blackboard_events.yml          # 黑板事件类型、并发控制
│       ├── intervention_rules.yml         # 置信度阈值、干预触发规则
│       └── fallback_policy.yml            # 合规违规分类、Auto-Fallback 策略
└── [待创建]
    ├── experts/                           # 专家配置目录
    │   ├── static/                        # 静态专家 (可从 settings 复制)
    │   └── dynamic/                       # 动态生成专家
    ├── core/                              # 核心专家系统逻辑
    ├── memory/                            # 记忆空间管理
    ├── services/                          # 大模型服务抽象
    └── utils/                             # 工具函数
```

## Key Files Reference

| 文件 | 内容摘要 |
|-----|---------|
| `sys_init/draft_goal/goal_settings.md` | 项目总目标、专家分类、工作流程、技术设定 |
| `sys_init/settings/expert_system_architecture.md` | 完整架构设计，含工作流转图、技术架构 |
| `sys_init/settings/static_experts.yml` | 10 位静态专家的完整 YAML 配置 |
| `sys_init/settings/dynamic_expert_template.yml` | 动态专家生成模板与流程说明 |
| `sys_init/settings/context_compression_summary.md` | 设计讨论的压缩摘要，快速检索 |
| `sys_init/settings/DESIGN_STATUS.md` | 设计文档状态总览，完成度检查清单 |
| `sys_init/settings/rate_limits.yml` | API 限流、重试、降级配置 |
| `sys_init/settings/model_routing.yml` | 模型白名单、任务路由、成本优化 |
| `sys_init/settings/blackboard_events.yml` | 黑板事件类型、并发控制、知识管理 |
| `sys_init/settings/intervention_rules.yml` | 置信度阈值、干预触发、死锁处理 |
| `sys_init/settings/fallback_policy.yml` | 合规违规分类、Auto-Fallback 策略 |

## Development Guidelines

### 专家 YAML Schema 核心字段

```yaml
expert:
  id: "unique_expert_id"
  role: "角色名称"
  archetype: "static | dynamic"
  load_strategy: "always | on_demand | dynamic"
  persona: { description, background, personality }
  thinking_framework: [...]
  strengths: [...]
  weaknesses: [...]
  blind_spots: [...]
  superhuman_insights: [...]
  capabilities: [{ name, params }]
  i_o_spec: { input, output }
  decision_style: { risk_tolerance, consensus_need }
  memory: { scope, retention }
```

### 关键设计决策

1. **可行性打分 + 置信度自签** - 任务分配时评估专家能力，专家接单后自评置信度，加权计入决策
2. **一票否决权** - 合规专家在准入和交付两个节点拥有不可覆盖的否决权
3. **操作链路可追溯** - 所有中间操作记录为可 fork 的链路
4. **动态专家注册** - 新专家被索引到专家库，系统能力持续进化
5. **API 限流与重试** - Token 桶限流 + 指数退避重试，防止并发请求压垮 API
6. **黑板 Append-only** - 专家只能追加事件，不能修改历史，由知识管理员定期压缩
7. **模型智能路由** - 根据任务类型动态选择最合适模型，禁止使用 GPT-4 和 Gemini 2.5 之前版本
8. **Auto-Fallback** - 合规违规可修复时自动尝试补救（最多 3 次），失败后驳回并附详细报告

## Current Status

**设计完成阶段** - 架构设计、静态专家配置、动态专家模板已完成并文档化。准备进入实现规划阶段。

## Next Steps

1. 确认架构设计文档是否符合预期
2. 审核静态专家配置是否需要调整
3. 选择 1-2 个典型使用场景进行详细设计
4. 使用 `superpowers:writing-plans` skill 创建实现计划
