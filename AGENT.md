# AGENTS.md

## 项目名称
CityScholar-Agent v2.1 —— 多 LLM 城市研究论文科研助手

## 项目定位
面向城市治理、城市规划、城市安全方向的 AI 论文科研助手。
围绕本地论文知识库，实现检索问答、论文分析、多篇比较、综述提纲、
一键工作流、GraphRAG 图增强检索、知识图谱可视化、MCP 服务器等完整科研闭环。

## 架构概览

```
App.py (CLI 入口，20+ 命令)
├── config.py (集中配置：API Key、路径、参数)
├── llm/ (LLM 抽象层)
│   ├── base.py (BaseLLMClient 抽象基类)
│   ├── deepseek_client.py (DeepSeek 客户端，urllib 直连)
│   ├── dashscope_adapter.py (DashScope 适配器)
│   └── factory.py (客户端工厂，自动选择最优供应商)
├── llm_dashscope.py (DashScope 原生客户端，chat + embedding)
├── core/
│   ├── agent.py (CityScholarAgent 主编排器，所有功能中枢)
│   ├── memory.py (多轮对话记忆 + 论文指代消解)
│   ├── prompts.py (LLM 提示模板集中管理)
│   └── workflow.py (多步工作流编排：Planner/State 模式)
├── rag/
│   ├── loader.py (PDF 文件扫描与发现)
│   ├── parser.py (PDF 解析 + LLM 元数据提取)
│   ├── splitter.py (文本分块：滑动窗口 + 重叠)
│   ├── embedder.py (向量索引构建/加载/指纹校验)
│   ├── retriever.py (混合检索 + 查询扩展 + LLM Reranker)
│   └── graphrag.py (GraphRAG 实体共现图 + 图增强检索)
├── tools/ (10 个工具模块)
│   ├── analyze_tool.py (单篇论文 7 维度结构化分析)
│   ├── compare_tool.py (多篇论文比较)
│   ├── outline_tool.py (综述提纲生成)
│   ├── export_tool.py (富文本 Markdown 报告导出)
│   ├── graph_tool.py (知识图谱构建 + HTML/PNG 可视化导出)
│   ├── summarize_tool.py (LLM 结构化摘要)
│   ├── translate_tool.py (中英学术翻译)
│   ├── recommend_tool.py (相似论文推荐)
│   └── bibtex_tool.py (BibTeX 引用导出)
├── security/
│   └── guard.py (输入注入检测 + 频率限制 + 输出审查 + 审计日志)
├── mcp/
│   ├── tools.py (MCP 工具定义，7 个标准工具)
│   └── server.py (MCP 服务器，JSON-RPC 协议)
└── data/
    ├── raw_papers/ (PDF 论文，约 40 篇)
    └── processed/ (向量索引缓存)
```

## 核心能力矩阵

| 能力 | 模块 | 说明 |
|------|------|------|
| 本地 PDF 论文管理 | rag/loader.py + parser.py | pypdf 提取文本 + LLM 提取元数据（标题/作者/年份/期刊/DOI/摘要/关键词） |
| 知识库构建 | rag/splitter.py + embedder.py | 滑动窗口分块 + DashScope embedding 向量化 + 指纹校验防重复 |
| RAG 混合检索 | rag/retriever.py | 查询扩展 → 关键词粗召回 + 向量语义召回 → 混合加权 → LLM Reranker → Top-K |
| 检索问答 | core/agent.py | 来源标注 + 依据片段验证 + 不确定性说明 |
| 单篇论文分析 | tools/analyze_tool.py | 规则引擎 7 维度提取 + LLM 增强分析（研究类型/方法标签/质量评估） |
| 多篇论文比较 | tools/compare_tool.py | 规则比较（兜底）+ LLM 增强 9 维度深度对比（JSON 结构化输出） |
| 综述提纲生成 | tools/outline_tool.py | 6 章结构化综述提纲自动生成 |
| 一键工作流 | core/workflow.py + tools/export_tool.py | 论文选择 → 比较 → 提纲 → 富文本 Markdown 报告 → 知识图谱导出 |
| GraphRAG 图增强 | rag/graphrag.py | 实体提取（方法/概念/技术/城市）→ 共现图构建 → 邻居扩展检索 |
| 知识图谱可视化 | tools/graph_tool.py | 交互式 HTML（vis.js，7 种节点类型）+ 静态 PNG（matplotlib+networkx） |
| 实体图谱导出 | rag/graphrag.py | Mermaid 代码打印 / DOT 文件 / PNG 图片（Graphviz） |
| 结构化摘要 | tools/summarize_tool.py | LLM 生成 背景-方法-发现-意义 四段摘要 |
| 中英互译 | tools/translate_tool.py | 学术文本翻译，自动检测方向 |
| 相似论文推荐 | tools/recommend_tool.py | 基于 embedding 余弦相似度的论文推荐（Top-N） |
| BibTeX 导出 | tools/bibtex_tool.py | 单篇/批量 BibTeX 引用条目生成 |
| 研究空白识别 | core/agent.py | 跨论文分析已有共识、研究空白与未来方向 |
| 多轮对话记忆 | core/memory.py | 上下文追踪 + 论文指代消解（"这篇""第一篇"） |
| 安全防护 | security/guard.py | 4 层防护：注入检测 + 内容清理 + 频率限制(30次/60s) + 输出审查 |
| MCP 服务器 | mcp/server.py | 标准 JSON-RPC MCP 协议，7 个工具，可接入 Claude Desktop |

## LLM 供应商

- **DeepSeek** (deepseek-chat): 主对话与分析模型，OpenAI 兼容接口
- **DashScope/通义千问** (qwen-plus/qwen-max): 分析备用模型 + text-embedding-v3 向量化
- 所有 LLM 客户端使用原生 `urllib` 直连 API，无需安装额外 SDK
- 优先级规则：chat → DeepSeek > qwen-plus，analysis → DeepSeek > qwen-max，embedding → 仅 DashScope

## 开发原则

1. 优先保证代码清晰、稳定、易维护
2. 模块单一职责，不把多个功能混在一个文件中
3. 命名直观，注释讲清楚"为什么这样做"
4. 保留必要的错误处理，优雅降级（无 API Key 时仍可运行规则模式）
5. 优先使用简单、成熟、常见的 Python 方案

## 当前版本边界

已实现：
- 完整的本地论文助手 CLI（20+ 命令）
- 规则 + LLM 双重分析管道（分析/比较/提纲）
- 一键工作流（比较→提纲→Markdown 报告→知识图谱）
- RAG 混合检索（关键词 + 向量 + Reranker）
- GraphRAG 实体共现图 + 图增强检索
- 知识图谱 HTML/PNG 可视化导出
- MCP 服务器（标准协议，7 个工具）
- 安全防护（注入检测 + 频率限制 + 内容审查）
- 多轮对话记忆 + 论文指代消解
- 结构化摘要 / 中英翻译 / 论文推荐 / BibTeX 导出

暂不扩展：
- 多智能体协作（Multi-Agent）
- Web 前端界面（Gradio / Streamlit）
- 在线论文抓取与检索
- 大规模数据库（SQLite / PostgreSQL）
- 多模态支持（图表识别、地图分析）
