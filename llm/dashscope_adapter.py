"""DashScope LLM 客户端适配器 —— 封装现有的 llm_dashscope 模块以符合 BaseLLMClient 接口。"""

from __future__ import annotations

from llm.base import BaseLLMClient, LLMConfig
from llm_dashscope import DashScopeClient


class DashScopeAdapter(BaseLLMClient):
    """将现有的 DashScopeClient 适配为 BaseLLMClient 接口。"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = DashScopeClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout_sec=config.timeout_sec,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: dict[str, str] | None = None,
    ) -> str:
        return self._client.chat(
            model=model or self.config.chat_model or "qwen-plus",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str = "",
        dimensions: int = 0,
    ) -> list[list[float]]:
        return self._client.embed_texts(
            model=model or "text-embedding-v3",
            texts=texts,
            dimensions=dimensions or 128,
        )
