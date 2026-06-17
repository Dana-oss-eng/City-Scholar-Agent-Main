"""LLM 客户端抽象基类与配置。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """单家 LLM 服务商的连接配置。"""
    provider: str
    api_key: str
    base_url: str
    chat_model: str = ""
    timeout_sec: int = 60


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类，定义 chat 与 embed 两个核心接口。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """发送对话请求，返回模型回复文本。"""
        ...

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str = "",
        dimensions: int = 0,
    ) -> list[list[float]]:
        """批量文本向量化。默认抛出 NotImplementedError，子类可选实现。"""
        raise NotImplementedError(f"{self.config.provider} 不支持 embedding 接口")

    @property
    def enabled(self) -> bool:
        return bool(self.config.api_key)
