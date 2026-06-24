"""本模块作用：智能体安全防护 —— 输入注入检测、内容消毒、频率限制、输出审查。

所有安全策略均以"不阻断正常工作流"为前提，可疑行为记录日志并可选拦截。
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field


# ====== 注入检测模式 ======

# 常见 prompt injection 模式
INJECTION_PATTERNS = [
    # 指令覆盖
    r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|messages?)",
    r"(?:you\s+are\s+now|now\s+you\s+are|act\s+as\s+(?:a|an)|pretend\s+(?:to\s+be|you\s+are))",
    r"(?:(?:new|override)\s+system\s+(?:prompt|instruction|role))",
    # 角色劫持
    r"(?:DAN\s|developer\s*mode|jailbreak|角色扮演)",
    r"(?:\|\s*(?:system|assistant|user)\s*[:：])",
    # 数据提取尝试
    r"(?:print|output|display|show|tell\s+me)\s+(?:your\s+)?(?:system\s+(?:prompt|instruction|message)|API\s+key|secret)",
    r"(?:what\s+(?:is|are)\s+your\s+(?:instructions?|system\s+prompt))",
    # 代码注入
    r"(?:```(?:python|bash|sh|js)\s*\n.*(?:os\.system|subprocess|exec|eval|__import__))",
    r"(?:\$\{.*\}|\{\{.*\}\})",  # 模板注入
]

# 可疑关键词（权重低，仅记录）
SUSPICIOUS_KEYWORDS = [
    "ignore instructions", "bypass", "jailbreak",
    "system prompt", "你的系统提示词", "你的指令",
    "/dev/null", "/bin/bash", "cmd.exe",
]

# 输入长度限制
MAX_INPUT_LENGTH = 10000
MAX_OUTPUT_LENGTH = 50000

# 频率限制
RATE_LIMIT_WINDOW = 60       # 时间窗口（秒）
RATE_LIMIT_MAX_REQUESTS = 30  # 窗口内最大请求数


@dataclass
class AuditLog:
    """安全审计日志条目。"""
    timestamp: float
    event_type: str      # "injection_detected" | "rate_limited" | "input_truncated" | "output_truncated"
    severity: str        # "low" | "medium" | "high"
    details: str
    user_input_snippet: str = ""


@dataclass
class SecurityGuard:
    """智能体安全防护门 —— 在输入进入 Agent 前和输出返回用户前执行安全策略。"""

    audit_logs: list[AuditLog] = field(default_factory=list)
    _request_times: list[float] = field(default_factory=list)
    _blocked_count: int = 0

    def check_input(self, user_input: str) -> tuple[str, list[str]]:
        """对用户输入执行安全检查。

        Returns:
            (sanitized_input, warnings) — 消毒后的输入 + 警告信息列表
        """
        warnings: list[str] = []

        # 1. 长度检查（截断而非阻断）
        if len(user_input) > MAX_INPUT_LENGTH:
            user_input = user_input[:MAX_INPUT_LENGTH]
            warnings.append(f"输入过长，已截断至 {MAX_INPUT_LENGTH} 字符。")
            self._log("input_truncated", "low", f"truncated from {len(user_input)} chars")

        # 2. 注入模式检测
        lowered = user_input.lower()
        injection_score = 0
        for pattern in INJECTION_PATTERNS:
            matches = re.findall(pattern, lowered, flags=re.IGNORECASE)
            if matches:
                injection_score += len(matches) * 20
                warnings.append(f"检测到可疑注入模式：{matches[0][:60]}...")
                self._log("injection_detected", "high",
                         f"pattern matched: {matches[0][:80]}",
                         user_input[:200])

        # 3. 可疑关键词
        for kw in SUSPICIOUS_KEYWORDS:
            if kw.lower() in lowered:
                injection_score += 5
                self._log("injection_detected", "medium",
                         f"suspicious keyword: {kw}", user_input[:200])

        # 高分数注入：标记但不阻断（可在配置中改为阻断）
        if injection_score >= 40:
            warnings.insert(0, "⚠️ 检测到高风险 prompt injection 尝试，请求已记录。")
            self._blocked_count += 1

        # 4. 基础消毒：移除 null 字节、控制字符
        user_input = user_input.replace("\x00", "")
        user_input = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", user_input)

        return user_input, warnings

    def check_rate_limit(self) -> tuple[bool, str]:
        """检查请求频率是否超限。

        Returns:
            (allowed, message) — 是否放行 + 提示信息
        """
        now = time.time()
        # 清理过期记录
        self._request_times = [t for t in self._request_times if now - t < RATE_LIMIT_WINDOW]

        if len(self._request_times) >= RATE_LIMIT_MAX_REQUESTS:
            self._log("rate_limited", "medium",
                     f"{len(self._request_times)} requests in {RATE_LIMIT_WINDOW}s")
            return False, f"请求频率过高（{RATE_LIMIT_WINDOW}s 内超过 {RATE_LIMIT_MAX_REQUESTS} 次），请稍后再试。"

        self._request_times.append(now)
        return True, ""

    def check_output(self, output: str) -> tuple[str, list[str]]:
        """对 Agent 输出执行安全检查。

        Returns:
            (sanitized_output, warnings)
        """
        warnings: list[str] = []

        # 1. 长度保护
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n\n[输出过长，已截断]"
            warnings.append(f"输出过长，已截断至 {MAX_OUTPUT_LENGTH} 字符。")

        # 2. 基础敏感信息泄露检测
        api_key_pattern = r"(?:sk-[a-zA-Z0-9]{20,}|[a-zA-Z0-9]{32,})"
        if re.search(api_key_pattern, output):
            output = re.sub(api_key_pattern, "[REDACTED]", output)
            warnings.append("输出中检测到疑似 API Key，已脱敏。")
            self._log("injection_detected", "high", "potential API key in output")

        return output, warnings

    def _log(self, event_type: str, severity: str, details: str, snippet: str = "") -> None:
        self.audit_logs.append(AuditLog(
            timestamp=time.time(),
            event_type=event_type, severity=severity,
            details=details, user_input_snippet=snippet,
        ))

    def get_summary(self) -> str:
        """返回安全摘要报告。"""
        lines = [
            "🔒 安全防护状态：",
            f"  审计日志数：{len(self.audit_logs)}",
            f"  已拦截高风险请求：{self._blocked_count}",
            f"  当前窗口请求数：{len(self._request_times)}/{RATE_LIMIT_MAX_REQUESTS}",
        ]
        if self.audit_logs:
            high = sum(1 for a in self.audit_logs if a.severity == "high")
            medium = sum(1 for a in self.audit_logs if a.severity == "medium")
            lines.append(f"  高风险事件：{high} | 中风险事件：{medium}")
            if high:
                lines.append("  最近高风险事件：")
                for a in self.audit_logs[-3:]:
                    if a.severity == "high":
                        lines.append(f"    - [{a.event_type}] {a.details[:80]}")
        return "\n".join(lines)

    def reset(self) -> None:
        """重置防护状态（保留审计日志）。"""
        self._request_times.clear()
        self._blocked_count = 0
