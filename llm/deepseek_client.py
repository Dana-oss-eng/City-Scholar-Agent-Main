"""DeepSeek API 客户端（OpenAI 兼容接口）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from llm.base import BaseLLMClient, LLMConfig


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端，使用 OpenAI 兼容的 /v1/chat/completions 端点。"""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: dict[str, str] | None = None,
    ) -> str:
        model = model or self.config.chat_model or "deepseek-chat"
        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {error_body[:500]}") from exc
        except Exception as exc:
            raise RuntimeError(f"DeepSeek API 调用失败: {exc}") from exc
