"""
utils/config.py
----------------
مكان واحد يقرأ فيه كل الإعدادات من ملف .env.
كل الملفات التانية (00_ ... 09_) بتستورد من هنا بدل ما كل ملف
يعمل load_dotenv() ويقرأ os.environ لوحده.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")


def _get(name: str, default=None):
    """
    بيدور على القيمة بالترتيب:
    1. متغيرات البيئة / .env (شغال محليًا زي ما هو)
    2. Streamlit Cloud Secrets (st.secrets) — لو الأب شغال جوه Streamlit
       Community Cloud وحاطط المفاتيح في "Secrets" بدل .env (اللي متسجلش
       على GitHub عمدًا لحمايته)
    3. default
    """
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # مش شغالين جوه Streamlit، أو مفيش secrets.toml — عادي، نكمل

    return default


# ---------- Model provider ----------
# "openai" أو "gemini"
MODEL_PROVIDER = _get("MODEL_PROVIDER", "openai").lower()

# ---------- OpenAI ----------
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")

# ---------- Gemini (بديل مجاني، من غير بطاقة ائتمان) ----------
GEMINI_API_KEY = _get("GEMINI_API_KEY")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-flash-latest")

# ---------- Embeddings ----------
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------- Chroma ----------
CHROMA_PERSIST_DIR = str(ROOT_DIR / _get("CHROMA_PERSIST_DIR", "vector_store/chroma_db"))
CHROMA_COLLECTION_NAME = _get("CHROMA_COLLECTION_NAME", "langchain_docs")

# ---------- Data collection ----------
# ملاحظة: dcs.langchain.com و مواقع تانية كتير بتعمل redirect. الكود بيتابعه
# لوحده. SOURCES_CONFIG_PATH بيحدد ملف فيه ليستة كل المواقع المطلوب جمعها.
SOURCES_CONFIG_PATH = ROOT_DIR / _get("SOURCES_CONFIG_FILE", "sources.json")

# احتفظنا بيهم كـ fallback لو حد شغل الكود القديم اللي بيقرأ متغير واحد بس
DOCS_BASE_URL = _get("DOCS_BASE_URL", "https://python.langchain.com/docs/introduction/")
ALLOWED_PATH_PREFIX = _get("ALLOWED_PATH_PREFIX", "https://docs.langchain.com/oss/python/")
MAX_PAGES = int(_get("MAX_PAGES", 150))

# ---------- Retrieval ----------
TOP_K = int(_get("TOP_K", 5))
HYBRID_ALPHA = float(_get("HYBRID_ALPHA", 0.5))  # وزن الـ semantic مقابل lexical

# ---------- Paths ----------
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_METADATA_DIR = ROOT_DIR / "data" / "metadata"
EVALUATION_DIR = ROOT_DIR / "evaluation"

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_METADATA_DIR, EVALUATION_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


def require_provider_key():
    """ينده قبل أي نداء للـ LLM عشان ياخد error واضح بدل ما يفشل جوه المكتبة."""
    if MODEL_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY مش موجود. اعمل نسخة من .env.example باسم .env "
            "واملأ فيه المفتاح بتاعك، أو حط MODEL_PROVIDER=gemini لو عايز تستخدم Gemini بدل."
        )
    if MODEL_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY مش موجود. خد مفتاح مجاني من https://aistudio.google.com/apikey "
            "وحطه في .env."
        )
    if MODEL_PROVIDER not in ("openai", "gemini"):
        raise RuntimeError(
            f"MODEL_PROVIDER='{MODEL_PROVIDER}' مش معروف. المسموح بس 'openai' أو 'gemini'."
        )
