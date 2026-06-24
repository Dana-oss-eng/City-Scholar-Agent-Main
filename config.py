"""本模块作用：集中管理 CityScholar-Agent 的多 LLM 供应商配置与运行参数。"""

import os
from pathlib import Path

# ── 加载 .env 文件 ──
# 优先使用 python-dotenv 库，不可用时回退到手动解析
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        with open(_ENV_PATH, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val


BASE_DIR = Path(__file__).resolve().parent
PROJECT_NAME = "CityScholar-Agent"
DATA_DIR = BASE_DIR / "data"
RAW_PAPERS_DIR = BASE_DIR / "raw_papers"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

# ====== DashScope（阿里云通义千问：chat + embedding） ======
DS_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DS_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
DS_CHAT_MODEL = os.getenv("DASHSCOPE_ANSWER_MODEL", "qwen-plus").strip()
DS_ANALYSIS_MODEL = os.getenv("DASHSCOPE_ANALYSIS_MODEL", "qwen-max").strip()
DS_EMBED_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3").strip()
DS_TIMEOUT = int(os.getenv("DASHSCOPE_TIMEOUT_SEC", "45"))
DS_EMBED_DIM = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "512"))

# ====== DeepSeek（chat 优先） ======
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# ====== 知识库参数 ======
CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("KB_TOP_K", "5"))


def get_app_config() -> dict:
    """返回应用当前使用的完整配置字典。"""
    return {
        "project_name": PROJECT_NAME,
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "raw_papers_dir": RAW_PAPERS_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "output_dir": OUTPUT_DIR,
        # DashScope
        "ds_api_key": DS_API_KEY,
        "ds_base_url": DS_BASE_URL,
        "ds_chat_model": DS_CHAT_MODEL,
        "ds_analysis_model": DS_ANALYSIS_MODEL,
        "ds_embed_model": DS_EMBED_MODEL,
        "ds_timeout": DS_TIMEOUT,
        "ds_embed_dim": DS_EMBED_DIM,
        # DeepSeek
        "deepseek_api_key": DEEPSEEK_API_KEY,
        "deepseek_base_url": DEEPSEEK_BASE_URL,
        "deepseek_model": DEEPSEEK_MODEL,
        # Knowledge base
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        # Derived
        "llm_enabled": bool(DS_API_KEY or DEEPSEEK_API_KEY),
        "embedding_enabled": bool(DS_API_KEY),
    }
