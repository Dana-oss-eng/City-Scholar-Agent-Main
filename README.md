# CityScholar-Agent

> 面向城市治理、城市规划与学术研究辅助场景的 LLM 智能体项目。
> 支持 PDF 论文批量解析、结构化分析、多篇比较、综述提纲生成、知识图谱可视化导出。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 目录

- [项目目标](#项目目标)
- [文件结构](#文件结构)
- [主要模块说明](#主要模块说明)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [运行示例](#运行示例)
- [知识图谱可视化](#知识图谱可视化)
- [依赖说明](#依赖说明)
- [技术架构](#技术架构)
- [路线图](#路线图)

---

## 项目目标

CityScholar-Agent 是一个面向城市研究学者的 LLM 智能体，核心能力包括：

1. **CLI 交互式界面**：支持 20+ 种命令的终端交互（问答、分析、对比、工作流等）
2. **PDF 论文管理**：批量扫描、解析、元数据提取（标题/作者/DOI/摘要/关键词）
3. **结构化分析**：自动提取研究问题、方法、数据来源、结论等 7 大维度
4. **多篇比较**：跨论文的方法、数据、发现对比，共同主题挖掘（LLM 增强 9 维度）
5. **综述提纲生成**：基于比较结果自动生成结构化综述提纲（6 章框架）
6. **一键工作流**：论文对比 → 综述提纲 → 富文本 Markdown 报告自动导出
7. **知识图谱可视化**：论文关系交互式 HTML 图谱 + 静态 PNG 导出
8. **GraphRAG 增强检索**：实体共现图谱构建 + 图增强搜索
9. **RAG 混合检索**：关键词 + 向量语义混合检索 + LLM Reranker 精排
10. **多 LLM 支持**：兼容 DeepSeek / 阿里 DashScope (Qwen)，可灵活切换
11. **MCP 服务器**：标准 MCP (Model Context Protocol) 接口，7 个标准工具
12. **安全防护**：输入注入检测、频率限制、内容审查、审计日志
13. **学术辅助工具**：结构化摘要、中英翻译、相似论文推荐、BibTeX 导出
14. **对话记忆**：多轮对话上下文管理 + 论文指代消解

---

## 文件结构

```text
CityScholar-Agent/
│
├── App.py                          # ★ CLI 交互式主入口（命令行界面）
├── config.py                       # 集中配置管理（API Key、路径、参数）
│
├── llm/                            # LLM 客户端抽象层
│   ├── __init__.py                 # 包初始化
│   ├── base.py                     # LLM 抽象基类 (BaseLLMClient) 与配置
│   ├── deepseek_client.py          # DeepSeek API 客户端（OpenAI 兼容接口）
│   ├── dashscope_adapter.py        # DashScope → BaseLLMClient 适配器
│   └── factory.py                  # LLM 客户端工厂函数，按优先级创建客户端
│
├── llm_dashscope.py                # DashScope 原始客户端（chat + embedding）
│                                   #   使用 urllib 直接调用，无需 dashscope SDK
│
├── rag/                            # RAG 检索增强模块
│   ├── __init__.py                 # 包初始化
│   ├── loader.py                   # PDF 文件扫描与发现
│   ├── parser.py                   # PDF 文本解析 + LLM 元数据提取
│   ├── splitter.py                 # 文本分块（滑动窗口 + 重叠）
│   ├── embedder.py                 # 向量索引构建、保存、加载（含指纹校验）
│   ├── retriever.py                # 多策略检索（关键词 + 向量混合 + Reranker）
│   └── graphrag.py                 # ★ GraphRAG 实体共现图构建与可视化
│
├── tools/                          # 工具模块集（10 个工具）
│   ├── __init__.py                 # 工具模块索引
│   ├── analyze_tool.py             # 单篇论文结构化分析（7 维度规则 + LLM 增强）
│   ├── compare_tool.py             # 多篇论文比较（共同主题/方法/数据/发现/启示）
│   ├── outline_tool.py             # 综述提纲自动生成（6 章结构化框架）
│   ├── export_tool.py              # Markdown 报告导出（含富文本章节）
│   ├── graph_tool.py               # ★ 知识图谱构建与可视化导出（HTML + PNG）
│   ├── summarize_tool.py           # LLM 结构化摘要生成（背景-方法-结果-意义）
│   ├── translate_tool.py           # 中英学术文本互译
│   ├── recommend_tool.py           # 基于向量相似度的相关论文推荐
│   └── bibtex_tool.py              # BibTeX 引用条目生成
│
├── core/                           # 工作流编排核心
│   ├── __init__.py                 # 包初始化
│   ├── agent.py                    # ★ CityScholarAgent 主编排器（所有功能入口）
│   ├── memory.py                   # 多轮对话记忆 + 论文指代消解
│   ├── prompts.py                  # LLM 提示模板集中管理
│   └── workflow.py                 # 多步流程编排（Planner/State 模式）
│
├── security/                       # 安全防护模块
│   ├── __init__.py                 # 包初始化
│   └── guard.py                    # 输入注入检测、内容清理、频率限制、输出审查
│
├── mcp/                            # MCP (Model Context Protocol) 接口
│   ├── __init__.py                 # 包初始化
│   ├── tools.py                    # MCP 工具定义（7 个工具）与处理器
│   └── server.py                   # MCP 服务器实现
│
├── scripts/                        # 辅助脚本
│   ├── build_course_notebooks.py   # 课程 Jupyter Notebook 构建脚本
│   └── notebook_utf8_guard.py      # Notebook UTF-8 编码守护
│
├── notebooks/                      # 课程 Jupyter Notebook
│   ├── 00_课程总览.ipynb / .md      # 课程总览
│   ├── 01_最小科研助教智能体.ipynb/.md # 第1周：最小智能体
│   ├── 02_工具调用与模块化.ipynb/.md  # 第2周：工具调用
│   ├── 03_多步流程编排.ipynb/.md     # 第3周：流程编排
│   └── 04_本地知识库与RAG.ipynb/.md  # 第4周：RAG 检索
│
├── raw_papers/                     # 论文 PDF 存放目录（版权材料已 gitignore）
│   └── .gitkeep                    # 占位文件，请将 PDF 论文放入此目录
│
├── outputs/                        # 运行时生成的报告输出（已 gitignore）
│
├── .env.example                    # 环境变量配置模板
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明（本文件）
└── .gitignore                      # Git 忽略文件
```

---

## 主要模块说明

### 1. `llm/` — LLM 客户端抽象层

提供统一的 LLM 调用接口，支持多家服务商：

| 组件 | 文件 | 说明 |
|------|------|------|
| 抽象基类 | `base.py` | 定义 `BaseLLMClient`，含 `chat()` 和 `embed_texts()` 接口 |
| DeepSeek | `deepseek_client.py` | 使用 OpenAI 兼容的 `/v1/chat/completions` 端点 |
| DashScope | `dashscope_adapter.py` | 将底层 `DashScopeClient` 适配为统一接口 |
| 工厂函数 | `factory.py` | 按优先级自动选择：DeepSeek > DashScope(qwen-plus/max) |

**优先级规则**：
- `chat_client`（主对话）：DeepSeek > DashScope qwen-plus
- `analysis_client`（分析/长上下文）：DeepSeek > DashScope qwen-max
- `embedding_client`（向量化）：仅 DashScope text-embedding-v3

### 2. `rag/` — 本地知识库与 RAG

| 组件 | 文件 | 说明 |
|------|------|------|
| 论文加载 | `loader.py` | 扫描 `raw_papers/` 下的 PDF 文件 |
| PDF 解析 | `parser.py` | 提取文本、划分页面、LLM 提取元数据（标题/作者/DOI 等） |
| 向量索引 | `embedder.py` | 批量向量化、索引持久化、指纹校验（避免重复构建） |
| 检索器 | `retriever.py` | 关键词检索 + 向量语义检索（可调权重混合）+ LLM Reranker 精排 + 查询扩展 |

**检索流程**：
```
用户问题 → 查询扩展(LLM) → 关键词粗召回 → 向量语义召回
         → 混合加权 → LLM Reranker 精排 → Top-K 结果
```

### 3. `tools/` — 工具模块集

| 工具 | 文件 | 输入 | 输出 |
|------|------|------|------|
| 论文分析 | `analyze_tool.py` | 论文字符串/片段 | 7 维度结构化分析 |
| 多篇比较 | `compare_tool.py` | ≥2 篇论文 | 共同主题/方法/数据/发现/启示对比 |
| 提纲生成 | `outline_tool.py` | 多篇论文 + 主题 | 6 章结构化综述提纲 |
| Markdown 导出 | `export_tool.py` | 工作流中间结果 | 富文本 `.md` 报告（各章节含实质性文字分析） |
| **知识图谱** | **`graph_tool.py`** | 比较结果 + 提纲 | **交互式 HTML + 可选 PNG** |
| 摘要生成 | `summarize_tool.py` | 论文全文 + LLM | 结构化摘要（JSON） |
| 翻译 | `translate_tool.py` | 学术文本 | 中→英 / 英→中 |
| 论文推荐 | `recommend_tool.py` | 目标论文 ID | 相似论文 Top-N |
| BibTeX | `bibtex_tool.py` | 论文元数据 | BibTeX 引用条目 |

**`graph_tool.py` 支持的节点类型**：
- 🟦 **论文节点** — 论文名 + 研究问题/对象
- 🟠 **共同主题** — 跨论文共同关注的话题
- 🟢 **研究方法** — 各论文使用的方法
- 🟣 **数据来源** — 各论文的数据基础
- 🟡 **研究发现** — 各论文的核心结论
- 🔴 **综合启示** — 对城市治理/规划的启示
- 🩵 **提纲章节** — 综述提纲的章节结构

### 4. `core/` — 工作流编排核心

#### `core/agent.py` — CityScholarAgent 主编排器

整个系统的中枢，整合所有模块功能。包含：
- 知识库构建与管理（`build_knowledge_base`）
- 多策略检索管道（`retrieve_chunks_for_question`）
- 问答生成与来源标注（`answer`，含 `_verify_and_enhance_citations`）
- 论文分析与 LLM 增强（`analyze_paper` + `_llm_enhanced_analysis`）
- 多论文比较（`compare_papers` + `_llm_compare_papers`）
- 综述提纲生成（`generate_review_outline`）
- 一键工作流执行（`run_review_workflow`）
- GraphRAG 图谱构建与图搜索（`build_entity_graph` + `graph_search`）
- 图谱可视化导出（`export_graph_visualization`）
- 学术翻译、论文推荐、BibTeX 导出、研究空白分析等

#### `core/memory.py` — 对话记忆管理

- 多轮对话历史存储（保留最近 N 轮）
- 论文指代消解：自动识别「这篇」「第一篇」等指代表达
- 论文级上下文追踪：记录当前讨论的论文、方法、主题
- 记忆增强的 LLM 上下文构建

#### `core/prompts.py` — 提示模板管理

集中管理所有 LLM 提示模板：
- 问答系统/任务提示（含来源标注要求）
- 研究空白分析提示
- LLM 增强分析提示（JSON 输出约束）
- LLM 多论文比较提示（9 维度 JSON）
- 记忆增强的论文上下文构建

#### `core/workflow.py` — 工作流编排

实现 Planner → State → 逐步执行的工作流模式：

```
计划 (WorkflowPlan)
  ├── 步骤1: select_papers    — 定位纳入论文
  ├── 步骤2: compare_papers   — 多篇论文比较
  ├── 步骤3: generate_outline — 生成综述提纲
  ├── 步骤4: export_markdown  — 导出富文本 Markdown 报告
  └── 步骤5: export_graph     — 导出可视化知识图谱
```

### 5. `rag/graphrag.py` — GraphRAG 实体共现图

基于论文块构建实体共现网络：

- **实体提取**：方法（methods）、概念（concepts）、技术（techniques）、城市（cities）
- **共现图构建**：论文块内共现实体自动建边
- **图增强检索**：从命中块出发扩展邻居实体，扩大检索覆盖面
- **三种可视化导出**：
  - Mermaid — 纯文本图谱代码（可在 Markdown 渲染器中查看）
  - Graphviz DOT — 标准图描述语言
  - PNG — 静态图片（需 Graphviz 命令行工具）

### 6. `security/` — 安全防护

`SecurityGuard` 提供多层安全防护：

| 防护层 | 说明 |
|--------|------|
| 输入注入检测 | 检测提示注入、越狱、角色切换等攻击模式 |
| 内容清理 | 清理高风险指令关键词 |
| 频率限制 | 60 秒窗口内最多 30 次请求 |
| 输出审查 | 检测 API Key 泄露、敏感信息泄露 |
| 审计日志 | 记录所有安全事件，支持事后追溯 |

### 7. `mcp/` — MCP 服务器

标准 Model Context Protocol 实现，对外暴露 7 个工具：

| 工具名 | 功能 |
|--------|------|
| `search_papers` | 按关键词搜索论文 |
| `answer_question` | 基于本地知识库的检索问答 |
| `analyze_paper` | 单篇论文 7 维度结构化分析 |
| `summarize_paper` | LLM 结构化摘要生成 |
| `compare_papers` | 多篇论文对比分析 |
| `list_papers` | 列出所有可用论文 |
| `graph_search` | GraphRAG 图增强检索 |

MCP 服务器支持标准 JSON-RPC 协议，可接入 Claude Desktop 等 MCP 客户端。

---

## 快速开始

### 1. 环境要求

- Python 3.10 或更高版本
- （可选）虚拟环境

### 2. 安装依赖

```bash
# 必选依赖（PDF 解析）
pip install -r requirements.txt

# 可选：静态 PNG 图谱导出
pip install matplotlib networkx
```

### 3. 配置 API Key

项目通过环境变量加载 API Key，推荐使用 `.env` 文件：

**步骤一：复制环境变量模板**

```bash
cp .env.example .env
```

**步骤二：编辑 `.env`，填入你的真实 Key**

```bash
# .env 文件内容示例：
DASHSCOPE_API_KEY=你的-阿里云-通义千问-Key
DEEPSEEK_API_KEY=你的-DeepSeek-Key
```

**步骤三（可选）：你也可以直接在系统环境变量中设置**

```bash
# Windows (CMD)
set DASHSCOPE_API_KEY=你的Key
set DEEPSEEK_API_KEY=你的Key

# macOS / Linux
export DASHSCOPE_API_KEY="你的Key"
export DEEPSEEK_API_KEY="你的Key"
```

> ⚠️ **安全提醒**：`.env` 文件已被 `.gitignore` 排除，不会被提交到 Git。切勿将 API Key 硬编码在源代码中。
> 
> **LLM 选择**：对话与分析优先使用 DeepSeek，向量嵌入仅使用 DashScope。至少需要配置一个对话模型的 Key。

### 4. 放入论文 PDF

将待分析的 PDF 论文放入 `raw_papers/` 目录。该项目支持城市韧性、城市治理、城市规划等领域的学术论文。

> **注意**：由于论文 PDF 受版权保护，仓库中不包含原始论文文件。请自行收集相关论文放入该目录。

### 5. 启动交互式 CLI

```bash
python App.py
```

启动后将显示欢迎横幅与交互式提示，支持以下命令：

```
💬 直接输入问题          → 检索问答（自动混合检索 + Reranker）
📄 summarize [N/关键词]  → 生成论文结构化摘要
🔬 analyze [N/关键词]    → 单篇论文深度分析（7 维度）
⚖️  compare [1,2]        → 多篇论文深度对比（LLM 增强 9 维度）
📋 outline [1,2]::主题   → 综述提纲生成
🔄 workflow [1,2]::主题  → 一键工作流（对比→提纲→导出 Markdown）
🔍 gaps [1,2,3]          → 研究空白与未来方向分析
📎 recommend [N]          → 相似论文推荐
🌐 translate 文本         → 中英学术翻译
📎 bibtex [N]             → 导出 BibTeX 引用
🕸️  graph build           → 构建论文实体共现图
🕸️  graph <问题>          → 图增强检索问答
🕸️  graph export [png]    → 导出可视化图谱（PNG / Mermaid / DOT）
📋 papers / list          → 列出所有可用论文
🔎 search <关键词>         → 按关键词搜索论文
🔧 build_index            → 构建向量索引
🧠 memory                 → 查看对话记忆状态
📊 stats                  → 查看系统详细状态
❓ help / 帮助             → 显示完整命令帮助
🚪 exit / quit            → 退出程序
```

---

## 运行方式

### 方式一：各模块独立运行（Demo 演示）

每个工具模块都包含 `run_xxx_demo()` 函数，可直接运行：

```bash
# PDF 加载器演示
python -m rag.loader

# PDF 解析器演示
python -m rag.parser

# 单篇论文分析演示
python -m tools.analyze_tool

# 多篇论文比较演示
python -m tools.compare_tool

# 综述提纲生成演示
python -m tools.outline_tool

# Markdown 导出演示
python -m tools.export_tool

# ★ 知识图谱可视化演示（导出 HTML + PNG）
python -m tools.graph_tool
```

### 方式二：完整工作流编排

通过 `core/workflow.py` 提供的函数进行完整流程：

```python
from llm.factory import create_llm_clients
from core.workflow import (
    build_default_workflow_plan,
    format_workflow_plan,
    export_workflow_result,
    export_workflow_graph,
)

# 1. 创建 LLM 客户端
providers = create_llm_clients(
    dashscope_api_key="your-key",
    deepseek_api_key="your-key",
)

# 2. 构建工作流计划
plan = build_default_workflow_plan(
    topic="城市韧性与安全治理",
    targets=["paper_a.pdf", "paper_b.pdf", "paper_c.pdf"],
)
print(format_workflow_plan(plan))

# 3. 导出 Markdown 报告
artifact = export_workflow_result(
    output_dir="outputs",
    topic="城市韧性与安全治理",
    selected_papers=[...],
    comparison_text="...",
    outline_text="...",
    step_logs=["步骤1", "步骤2", ...],
)

# 4. 导出知识图谱（交互式 HTML）
html_path, png_path = export_workflow_graph(
    output_dir="outputs",
    topic="城市韧性与安全治理",
    comparison=comparison_result,
    outline=outline_result,
)
print(f"图谱已导出至：{html_path}")
```

### 方式三：Jupyter Notebook 交互式学习

项目提供了 5 个课程配套 Notebook：

| Notebook | 内容 | 对应周 |
|----------|------|--------|
| `00_课程总览.ipynb` | 课程总览、学习路线 | — |
| `01_最小科研助教智能体.ipynb` | 最小智能体构建 | 第 1 周 |
| `02_工具调用与模块化.ipynb` | 工具调用与模块化 | 第 2 周 |
| `03_多步流程编排.ipynb` | 多步流程编排 | 第 3 周 |
| `04_本地知识库与RAG.ipynb` | 知识库与 RAG | 第 4 周 |

```bash
cd notebooks
jupyter notebook
```

---

## 运行示例

### 示例 1：PDF 加载器

```bash
$ python -m rag.loader

Loader Demo
论文目录：/path/to/project/raw_papers
发现 PDF 数量：17

1. 2025 - Advancements in the application of large language models...
2. Allam 等 - 2022 - Unpacking the '15-minute city' via 6G, IoT...
3. Chelleri和Baravikova - 2021 - Understandings of urban resilience...
...
```

### 示例 2：单篇论文结构化分析

```bash
$ python -m tools.analyze_tool

结构化学术分析结果：
论文名称：demo_paper.pdf
文档编号：demo_paper
1. 研究问题：文本中未明确识别出研究问题...
2. 研究对象：某沿海超大城市的社区更新项目
3. 方法：研究采用问卷调查、POI 数据分析与多元回归方法。
4. 数据来源：数据来源包括 2023 年社区问卷、城市开放 POI 数据和统计年鉴
5. 主要结论：设施步行可达性提升能够显著改善居民对社区更新的评价
6. 局限性：样本主要集中于中心城区，外部可推广性仍然受限
7. 对城市治理/规划/安全的启示：将十五分钟生活圈与道路安全整治协同推进
```

### 示例 3：多篇论文比较

```bash
$ python -m tools.compare_tool

多篇论文比较结果：
比较主题：城市韧性与安全治理
纳入论文数量：2
纳入论文：
1. 《paper_a.pdf》
   研究问题：城市韧性如何评估？
   研究对象：沿海城市群
   方法：指标评价与空间分析方法
2. 《paper_b.pdf》
   研究问题：城市安全治理与韧性提升
   研究对象：内陆都市圈
   方法：问卷调查与回归分析

共同主题：
- 多篇论文都涉及「城市韧性」相关议题。
- 多篇论文都涉及「安全治理」相关议题。

方法比较：
- 《paper_a.pdf》：指标评价与空间分析方法
- 《paper_b.pdf》：问卷调查与回归分析方法

综合启示：
- 《paper_a.pdf》提示：提升跨区域协同治理能力
- 《paper_b.pdf》提示：公共服务规划与安全治理协同推进
```

### 示例 4：知识图谱导出

```bash
$ python -m tools.graph_tool

============================================================
知识图谱：「城市韧性与安全治理 — 知识图谱」
节点总数：21
边总数：39
节点分布：
  - 论文：3 个
  - 共同主题：3 个
  - 研究方法：3 个
  - 数据来源：3 个
  - 研究发现：3 个
  - 综合启示：3 个
  - 提纲章节：3 个

核心论文：
  - paper_a
    研究问题：城市韧性如何评估？
  - paper_b
    研究问题：城市安全治理如何影响韧性？
  - paper_c
    研究问题：数字化如何提升城市韧性？
============================================================

[OK] 交互式 HTML 图谱已导出：outputs/knowledge_graph_demo.html
[OK] 静态 PNG 图谱已导出：outputs/knowledge_graph_demo.png

[TIP] 用浏览器打开 HTML 文件即可交互式浏览知识图谱。
   支持节点拖拽、缩放、搜索、点击高亮邻居、截图导出等功能。
```

生成的知识图谱 HTML 文件用浏览器打开后，可以看到：

- 📊 **交互式网络图**：节点（论文/主题/方法/数据/发现/启示/章节）和边（关系连线）
- 🔍 **搜索过滤**：按节点名称搜索并定位
- 🖱️ **点击高亮**：点击节点高亮其所有邻居连接
- 📸 **截图导出**：工具栏一键导出 PNG 截图
- ⏯️ **物理引擎**：可暂停/恢复力导向布局动画
- 📋 **节点详情**：侧边栏显示选中节点的元数据信息

### 示例 5：完整工作流 Markdown 报告

运行完整工作流后，`outputs/` 目录下会生成包含实质性文字分析的 Markdown 报告：

```markdown
# 城市韧性研究综述

## 综述摘要

本报告围绕「城市韧性研究综述」这一主题，基于对 8 篇城市研究领域论文的系统性分析，
从研究问题、理论基础、研究方法、数据来源、核心发现等多个维度进行了深度比较与综合评述。
报告进一步提供了结构化的综述提纲，为后续撰写完整的文献综述提供了框架参考。

## 纳入论文

本次综述共纳入 8 篇论文：

1. **LLMs in urban studies A systematic review.pdf**
2. **Understandings of urban resilience meanings.pdf**
3. **Effects of spatial structure on carbon emissions.pdf**
4. **Unpacking the 15-minute city via 6G IoT and digital twins.pdf**
5. **Perspectives of resilience in mega-event studies.pdf**
6. **Spatio-temporal evolution of resilience Chengdu-Chongqing.pdf**
7. **Assessment model for urban resilience based on PSR-BP-GA.pdf**
8. **Urban resilience and livability of European smart cities.pdf**

## 多篇论文比较分析

本部分对纳入的 8 篇论文进行了多维度结构化比较，涵盖研究问题、理论基础、
研究方法、数据来源、核心发现、局限性及互补性等关键维度，
旨在揭示论文间的共识、分歧与协同关系。

### 研究问题异同
八篇论文从技术工具创新、概念框架反思和实证因果分析等不同角度切入...

### 研究方法对比
系统综述法、质性访谈、空间计量模型、机器学习与神经网络等多种方法形成互补...

## 综述提纲

基于多篇论文的比较分析结果，围绕「城市韧性研究综述」这一主题，
以下综述提纲为撰写系统性的文献综述提供了结构化框架。

### 1. 研究背景与问题提出
- 城市韧性研究缘起与发展
- 当前城市安全治理的现实需求

### 2. 核心概念与理论基础
- 韧性理论的演进与主要流派
- 城市治理与规划的交叉视角

## 综合小结

本工作流通过对 8 篇论文的多维度比较与提纲梳理，
系统呈现了「城市韧性研究综述」领域的研究现状与知识结构。
上述比较分析与综述提纲可作为文献综述撰写的基础框架，
建议结合具体研究需求进一步深化各章节内容。

## 工作流执行日志
- 步骤 1：已选中 8 篇论文。
- 步骤 2：已完成多篇论文比较。
- 步骤 3：已生成综述提纲。
- 步骤 4：已导出 Markdown 报告到 outputs/城市韧性研究综述.md
```

---

## 知识图谱可视化

项目包含两套图谱系统：

### 1. 知识图谱（`graph_tool.py`）— 论文关系可视化

基于多论文比较结果构建，用于工作流报告。

#### 导出格式

| 格式 | 文件 | 特点 | 依赖 |
|------|------|------|------|
| **交互式 HTML** | `*_知识图谱.html` | 可拖拽/缩放/搜索/高亮/截图 | **无额外依赖**（vis.js CDN） |
| 静态 PNG | `*_知识图谱.png` | 传统论文插图格式 | matplotlib + networkx |

#### HTML 图谱功能

打开 `.html` 文件后可以使用：

- 🖱️ **节点拖拽** — 自由调整节点位置
- 🔍 **滚轮缩放** — 放大/缩小视图
- 🔎 **搜索过滤** — 在侧边栏输入关键词定位节点
- 👆 **点击高亮** — 单击节点高亮其所有关联邻居
- 📋 **详情面板** — 显示选中节点的元数据
- 📸 **PNG 截图** — 工具栏按钮一键导出当前视图为 PNG
- ⏯️ **物理引擎控制** — 暂停/恢复节点自动布局

#### 图谱中的关系类型

```
论文 ←→ 论文    (同主题共现)
论文  → 主题    (涉及)
论文  → 方法    (使用)
论文  → 数据    (基于)
论文  → 发现    (得出)
论文  → 启示    (启示)
章节  → 论文    (参考)
```

### 2. 实体共现图（`graphrag.py`）— GraphRAG 实体网络

基于全库论文块构建，用于增强检索。实体类型包括：方法、概念、技术、城市。

```bash
# CLI 中构建并导出
graph build                      # 构建实体共现图
graph stats                      # 查看图谱统计
graph export                     # 导出为 PNG 图片（默认）
graph export mermaid             # 打印 Mermaid 代码（可复制到 Markdown 编辑器渲染）
graph export dot                 # 导出 Graphviz DOT 文件
graph <问题>                     # 图增强检索问答
```

---

## 依赖说明

### 必选依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `pypdf` | ≥5.1.0 | PDF 文本提取 |

### 可选依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `matplotlib` | ≥3.8 | 知识图谱静态 PNG 导出 |
| `networkx` | ≥3.2 | 图谱数据结构（PNG 导出用） |
| `pdfplumber` | ≥0.11 | 更好的 PDF 表格提取 |
| `PyMuPDF` | ≥1.24 | 更快的 PDF 解析与版面分析 |
| `pytesseract` | ≥0.3 | OCR 扫描版 PDF |
| `pdf2image` | ≥1.17 | PDF 转图片（OCR 前置步骤） |

> **设计原则**：项目的核心功能（LLM 调用、PDF 解析、论文分析、知识图谱 HTML 导出）均使用 Python 标准库，最小化外部依赖。交互式图谱通过浏览器端 CDN 加载 vis.js，无需额外 Python 包。

---

## 技术架构

```
┌──────────────────────────────────────────────────────────────┐
│                        App.py (CLI)                          │
│                   交互式命令行主入口（20+ 命令）                 │
├──────────────────────────────────────────────────────────────┤
│                      core/agent.py                           │
│              CityScholarAgent 主编排器（所有功能中枢）          │
├──────────────────────────────────────────────────────────────┤
│  core/            │  tools/          │  rag/       │  llm/    │
│  ├─ workflow.py   │  ├─ analyze      │  ├─ loader  │  ├─ base │
│  ├─ memory.py     │  ├─ compare      │  ├─ parser  │  ├─ deep │
│  ├─ prompts.py    │  ├─ outline      │  ├─ split   │  ├─ dash │
│                   │  ├─ export       │  ├─ embed   │  └─ fact │
│                   │  ├─ graph ★      │  ├─ retrieve│          │
│                   │  ├─ summarize    │  └─ graphrag│          │
│                   │  ├─ translate    │             │          │
│                   │  ├─ recommend    │             │          │
│                   │  └─ bibtex       │             │          │
├──────────────────────────────────────────────────────────────┤
│  security/guard.py  │  mcp/server.py  │  llm_dashscope.py     │
│  安全防护（4层）      │  MCP 服务器      │  DashScope 原生客户端  │
├──────────────────────────────────────────────────────────────┤
│  raw_papers/                   │  outputs/                    │
│  (存放论文 PDF，已 gitignore)      │  (运行时生成，已 gitignore)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 路线图

- [x] 最小智能体骨架
- [x] 工具调用与模块化（分析/比较/提纲/导出/摘要/翻译/推荐/BibTeX）
- [x] 多步流程编排（Planner/State 模式）
- [x] 本地知识库与 RAG（关键词 + 向量混合检索 + Reranker）
- [x] 知识图谱可视化导出（交互式 HTML + 静态 PNG）
- [x] GraphRAG 实体共现图（图增强检索）
- [x] MCP 服务器（标准 Model Context Protocol 接口）
- [x] 安全防护（注入检测 + 频率限制 + 内容审查）
- [x] 多轮对话记忆与论文指代消解
- [x] 富文本 Markdown 报告（各章节含实质性文字分析）
- [ ] LangChain / LangGraph 工作流集成
- [ ] Web 界面（Gradio / Streamlit）
- [ ] 多模态支持（图表识别、地图分析）
- [ ] 自动文献综述生成（含引用管理）

---

## 许可证

MIT License
