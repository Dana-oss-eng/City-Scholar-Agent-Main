"""本模块作用：中英学术文本互译，支持论文标题、摘要、段落的翻译。"""


_TRANSLATE_CN2EN = """将以下中文学术文本翻译为英文，保持学术风格和专业术语准确。

中文原文：
{text}

英文翻译："""

_TRANSLATE_EN2CN = """将以下英文学术文本翻译为中文，保持学术风格和专业术语准确。

英文原文：
{text}

中文翻译："""


def translate(text: str, chat_fn, direction: str = "auto") -> str:
    """翻译学术文本。

    Args:
        text: 待翻译文本
        chat_fn: LLM chat 函数
        direction: "cn2en" / "en2cn" / "auto"（自动检测）
    """
    if chat_fn is None:
        return "LLM 未启用，无法翻译。"

    if direction == "auto":
        # 简单检测：中文占比高则中→英，否则英→中
        cn_chars = sum(1 for c in text if "一" <= c <= "鿿")
        direction = "cn2en" if cn_chars > len(text) * 0.3 else "en2cn"

    prompt = _TRANSLATE_CN2EN if direction == "cn2en" else _TRANSLATE_EN2CN

    try:
        return chat_fn(
            messages=[{"role": "user", "content": prompt.format(text=text[:5000])}],
            temperature=0.2, max_tokens=1500,
        )
    except Exception as exc:
        return f"翻译失败：{exc}"
