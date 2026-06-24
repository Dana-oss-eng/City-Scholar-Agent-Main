"""MCP (Model Context Protocol) 服务端 —— JSON-RPC 2.0 over stdio。

将 CityScholar-Agent 暴露为 MCP 工具服务，供 MCP 客户端（如 Claude Desktop）调用。

用法：
    python -m mcp.server                     # 独立运行（stdio 模式）
    python mcp/server.py                     # 直接运行
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 通过环境变量注入 API Key
# 方式一：在系统环境变量中设置 DASHSCOPE_API_KEY 和 DEEPSEEK_API_KEY
# 方式二：将下方占位符替换为你的真实 Key（注意：不要将真实 Key 提交到公开仓库）
# API Key 请通过环境变量或 .env 文件设置，切勿硬编码在此处
# 参见 .env.example 了解所需的环境变量

from config import get_app_config
from core.agent import CityScholarAgent
from llm.factory import create_llm_clients
from mcp.tools import TOOL_DEFINITIONS, handle_tool_call


SERVER_NAME = "cityscholar-mcp"
SERVER_VERSION = "2.0.0"


class MCPServer:
    """最小 MCP JSON-RPC 2.0 stdio 服务端。"""

    def __init__(self):
        self.agent: CityScholarAgent | None = None
        self._initialized = False

    def _init_agent(self) -> None:
        config = get_app_config()
        providers = create_llm_clients(
            dashscope_api_key=config["ds_api_key"],
            dashscope_base_url=config["ds_base_url"],
            dashscope_embedding_model=config["ds_embed_model"],
            deepseek_api_key=config["deepseek_api_key"],
            deepseek_base_url=config["deepseek_base_url"],
            deepseek_model=config["deepseek_model"],
        )
        self.agent = CityScholarAgent(
            raw_papers_dir=Path(config["raw_papers_dir"]),
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 150)),
            top_k=int(config.get("top_k", 5)),
        )
        self.agent.chat_client = providers.chat_client
        self.agent.analysis_client = providers.analysis_client
        self.agent.build_knowledge_base()

        # 尝试加载元数据和向量索引
        if self.agent.chat_client:
            try:
                self.agent.enrich_metadata()
            except Exception:
                pass

        if providers.embedding_client:
            try:
                self.agent.embedding_client = providers.embedding_client
                self.agent.embedding_model_name = config["ds_embed_model"]
                self.agent.embedding_dimensions = int(config.get("ds_embed_dim", 512))
                self.agent.prepare_embedding_index(
                    client=providers.embedding_client,
                    model_name=self.agent.embedding_model_name,
                    dimensions=self.agent.embedding_dimensions,
                    processed_data_dir=Path(config["processed_data_dir"]),
                    build_if_missing=False, force_rebuild=False,
                )
            except Exception:
                pass

    def handle_request(self, request: dict) -> dict | None:
        """处理单个 JSON-RPC 请求，返回响应或 None（通知）。"""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return self._response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        if method == "notifications/initialized":
            self._initialized = True
            return None  # 通知无需响应

        if method == "tools/list":
            return self._response(req_id, {"tools": TOOL_DEFINITIONS})

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            content = handle_tool_call(tool_name, arguments, self.agent)
            return self._response(req_id, {"content": content})

        if method == "ping":
            return self._response(req_id, {})

        return self._error(req_id, -32601, f"未知方法: {method}")

    def _response(self, req_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def send(self, message: dict) -> None:
        """将响应写入 stdout。"""
        sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def run(self) -> None:
        """启动 stdio 主循环。"""
        self._init_agent()
        # 通知上层 MCP 日志（通过 stderr，不干扰 stdio 协议）
        sys.stderr.write(f"[MCP] {SERVER_NAME} v{SERVER_VERSION} started\n")
        sys.stderr.write(f"[MCP] {len(self.agent.ensure_knowledge_base_ready().documents)} papers loaded\n")
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = self.handle_request(request)
            if response is not None:
                self.send(response)


# ====== 主入口 ======

if __name__ == "__main__":
    server = MCPServer()
    server.run()
