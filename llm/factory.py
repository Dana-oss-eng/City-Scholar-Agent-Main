"""LLM 客户端工厂函数 —— 根据配置创建对应的客户端实例。"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm.base import BaseLLMClient, LLMConfig
from llm.deepseek_client import DeepSeekClient
from llm.dashscope_adapter import DashScopeAdapter


@dataclass
class LLMProviders:
    """统一管理所有 LLM 服务商客户端。"""
    chat_client: BaseLLMClient | None = None          # 主对话模型（优先）
    analysis_client: BaseLLMClient | None = None      # 分析模型（长上下文，高质量）
    embedding_client: BaseLLMClient | None = None     # 向量模型（仅 DashScope 支持）


def create_llm_clients(
    dashscope_api_key: str = "",
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    dashscope_chat_model: str = "qwen-plus",
    dashscope_analysis_model: str = "qwen-max",
    dashscope_embedding_model: str = "text-embedding-v3",
    dashscope_timeout: int = 45,
    deepseek_api_key: str = "",
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_model: str = "deepseek-chat",
) -> LLMProviders:
    """根据提供的 API Key 创建可用的 LLM 客户端集合。

    优先级规则：
    - chat_client: DeepSeek > DashScope(qwen-plus)
    - analysis_client: DeepSeek > DashScope(qwen-max)
    - embedding_client: 仅 DashScope
    """

    providers = LLMProviders()

    # DeepSeek（优先用于 chat 和 analysis）
    if deepseek_api_key:
        ds_config = LLMConfig(
            provider="deepseek",
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
            chat_model=deepseek_model,
            timeout_sec=60,
        )
        ds_client = DeepSeekClient(ds_config)
        providers.chat_client = ds_client
        providers.analysis_client = ds_client

    # DashScope（备选 chat/analysis，以及唯一的 embedding 提供商）
    if dashscope_api_key:
        ds_chat_config = LLMConfig(
            provider="dashscope",
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
            chat_model=dashscope_chat_model,
            timeout_sec=dashscope_timeout,
        )
        ds_chat = DashScopeAdapter(ds_chat_config)

        if providers.chat_client is None:
            providers.chat_client = ds_chat

        # analysis 模型使用 qwen-max
        ds_analysis_config = LLMConfig(
            provider="dashscope",
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
            chat_model=dashscope_analysis_model,
            timeout_sec=dashscope_timeout,
        )
        ds_analysis = DashScopeAdapter(ds_analysis_config)
        if providers.analysis_client is None:
            providers.analysis_client = ds_analysis

        # embedding 始终使用 DashScope
        ds_emb_config = LLMConfig(
            provider="dashscope",
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
            chat_model=dashscope_embedding_model,
            timeout_sec=dashscope_timeout,
        )
        providers.embedding_client = DashScopeAdapter(ds_emb_config)

    return providers
