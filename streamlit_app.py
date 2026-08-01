"""
streamlit_app.py
--------------------
Phase 8 (Frontend) — AI Learning Hub

واجهة premium على شكل ChatGPT/Claude/Perplexity: chat bubbles, glass cards
لكل مصدر، تبويبات (Sources / Retrieved Chunks / Details)، sidebar فيه
إعدادات + فلترة الـ knowledge base + تاريخ المحادثة.

ملف واحد بس (مقصود) — أسهل في النسخ والصيانة من نسخة الملفات المتعددة.

تشغيل:
    streamlit run streamlit_app.py
"""

import json
import re
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path

import streamlit as st

# بعض إصدارات بايثون/الـ launchers الجديدة (زي Python 3.14) مش بتضيف
# مجلد السكريبت نفسه لـ sys.path تلقائيًا زي القديم، فـ import_module
# بتاعت الموديولات المرقّمة (06_retrieve_context...) كانت بتفشل. السطر ده
# بيضمن إن المجلد ده دايمًا موجود في مسار البحث، أيًا كان إصدار بايثون.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, EMBEDDING_MODEL, EVALUATION_DIR, GEMINI_MODEL
from utils.config import HYBRID_ALPHA, MODEL_PROVIDER, OPENAI_MODEL, TOP_K

retrieve_module = import_module("06_retrieve_context")
context_module = import_module("06b_build_context")
generate_module = import_module("07_generate_answer")


# ──────────────────────────────────────────────────────────────────────────
# صفحة + إعدادات عامة
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Learning Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# خريطة كل مصدر: أيقونة + اسم عرض + لون مميز (structure = information: كل
# مصدر ليه هوية لونية ثابتة، تتكرر من الـ sidebar لحد الـ citation chip
# جوه الإجابة، عشان "من فين جت المعلومة دي" يبقى واضح بصريًا فورًا)
SITE_META = {
    "langchain": {"icon": "🦜", "label": "LangChain", "color": "#818cf8"},
    "langgraph": {"icon": "🕸️", "label": "LangGraph", "color": "#fb7185"},
    "huggingface_transformers": {"icon": "🤗", "label": "HF Transformers", "color": "#f5a623"},
    "huggingface_tokenizers": {"icon": "🤗", "label": "HF Tokenizers", "color": "#fbbf24"},
    "huggingface_tasks": {"icon": "🤗", "label": "HF Tasks", "color": "#fdba74"},
    "sentence_transformers": {"icon": "🧬", "label": "Sentence-Transformers", "color": "#22d3ee"},
    "sentence_transformers_semantic_search": {
        "icon": "🧬", "label": "SBERT Semantic Search", "color": "#67e8f9",
    },
    "faiss": {"icon": "🔍", "label": "FAISS", "color": "#a78bfa"},
    "scikit_learn": {"icon": "🔬", "label": "scikit-learn", "color": "#fb923c"},
    "pytorch": {"icon": "🔥", "label": "PyTorch", "color": "#ee4c2c"},
    "tensorflow": {"icon": "📐", "label": "TensorFlow", "color": "#ff9800"},
    "keras": {"icon": "⌨️", "label": "Keras", "color": "#d00000"},
    "spacy": {"icon": "⚡", "label": "spaCy", "color": "#f472b6"},
    "nltk": {"icon": "📖", "label": "NLTK", "color": "#34d399"},
}
DEFAULT_SITE_META = {"icon": "📄", "label": "Other", "color": "#94a3b8"}


def site_meta(site_name: str) -> dict:
    return SITE_META.get(site_name, {**DEFAULT_SITE_META, "label": site_name or "Other"})


# ──────────────────────────────────────────────────────────────────────────
# CSS — كل الطلاء البصري: glass panels, gradients, animations, typography
# ──────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root{
  --bg-0:#0a0d16;
  --bg-1:#0e1220;
  --panel:rgba(24,29,46,.66);
  --panel-solid:#171c2c;
  --border:rgba(255,255,255,.09);
  --border-strong:rgba(255,255,255,.17);
  --text-0:#f3f5f9;
  --text-1:#c2c9db;
  --text-2:#8b93a8;
  --accent-0:#5b62f0;
  --accent-1:#8b90f7;
  --accent-glow:rgba(91,98,240,.30);
  --success:#3ddb96;
  --radius-lg:16px;
  --radius-md:11px;
  --radius-sm:8px;
}

@media (prefers-reduced-motion: reduce){
  *{ animation-duration:.001ms !important; transition-duration:.001ms !important; }
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 620px at 14% -8%, rgba(91,98,240,.12), transparent 60%),
    radial-gradient(900px 560px at 100% 8%, rgba(61,219,150,.05), transparent 55%),
    var(--bg-0);
  font-family:'Inter', sans-serif;
  color:var(--text-0);
  font-size:16px;
}

h1,h2,h3,h4, .hub-brand-title{ font-family:'Sora', sans-serif; letter-spacing:-.01em; }
code, .mono{ font-family:'JetBrains Mono', monospace; }

/* قراءة أوضح: نص الشات وأي فقرة عادية بحجم وتباعد سطر مريح */
p, li, [data-testid="stMarkdownContainer"] p{
  font-size:1rem !important; line-height:1.72 !important; color:var(--text-0);
}
[data-testid="stMarkdownContainer"] li{ line-height:1.65 !important; margin-bottom:.15rem; }

/* مهم جدًا: مش بنخفي الـ header ولا الـ toolbar خالص، لأن زرار فتح/قفل
   السايدبار عايش جواهم، ومكانه بيختلف شوية بين نسخ Streamlit. أي محاولة
   نخفيهم بتسبب بالظبط المشكلة اللي حصلت: السايدبار بيتقفل ومفيش زرار
   يرجّعه. بنكتفي بإخفاء #MainMenu و footer بس (عناصر ثابتة وآمنة من
   إصدار لإصدار)، ونلوّن الـ header بس عشان يتماشى مع الثيم الداكن. */
#MainMenu, footer{ visibility:hidden; }
[data-testid="stDecoration"]{ display:none; }
header[data-testid="stHeader"]{
  background:var(--bg-0) !important;
  border-bottom:1px solid var(--border);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(11,15,26,.98), rgba(7,10,18,.98));
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] *{ font-family:'Inter', sans-serif; }
[data-testid="stSidebarUserContent"]{ padding-top:1.1rem; }

.hub-brand{
  display:flex; align-items:center; gap:.65rem;
  padding:.25rem .1rem 1rem .1rem;
  border-bottom:1px solid var(--border);
  margin-bottom:1rem;
}
.hub-brand-icon{
  width:40px; height:40px; border-radius:12px; flex:none;
  display:flex; align-items:center; justify-content:center;
  font-family:'Sora', sans-serif; font-size:.92rem; font-weight:800; letter-spacing:.02em;
  color:#fff;
  background:linear-gradient(135deg, var(--accent-0), var(--success));
  box-shadow:0 6px 18px var(--accent-glow);
}
.hub-brand-title{ font-size:1.1rem; font-weight:700; color:var(--text-0); line-height:1.2; }
.hub-brand-sub{ font-size:.78rem; color:var(--text-2); margin-top:.1rem; }

.hub-section-label{
  font-size:.74rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--text-2); margin:1.2rem 0 .5rem 0;
}

/* nav radio -> pill list */
[data-testid="stSidebar"] div[role="radiogroup"]{ gap:.15rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label{
  border-radius:10px; padding:.45rem .6rem; transition:background .15s ease;
  width:100%;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{ background:rgba(255,255,255,.04); }
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]{
  background:linear-gradient(90deg, rgba(91,98,240,.24), rgba(91,98,240,.05));
  border:1px solid rgba(91,98,240,.4);
}

/* ---------- Buttons ---------- */
.stButton>button{
  border-radius:10px !important; border:1px solid var(--border-strong) !important;
  background:var(--panel-solid) !important; color:var(--text-0) !important;
  font-weight:600 !important; transition:all .15s ease !important;
}
.stButton>button:hover{
  border-color:var(--accent-1) !important; transform:translateY(-1px);
  box-shadow:0 8px 20px rgba(0,0,0,.35);
}
.hub-danger button{ color:#fca5a5 !important; border-color:rgba(252,165,165,.25) !important; }
.hub-danger button:hover{ border-color:#f87171 !important; }

/* ---------- Chat header ---------- */
.hub-header{
  display:flex; align-items:center; gap:.9rem; padding:.4rem 0 1.1rem 0;
  border-bottom:1px solid var(--border); margin-bottom:1.3rem;
}
.hub-header-title{ font-size:1.7rem; font-weight:800; margin:0; display:flex; align-items:center; gap:.55rem; }
.hub-header-mark{
  display:inline-flex; align-items:center; justify-content:center;
  width:34px; height:34px; border-radius:10px; flex:none;
  font-family:'Sora', sans-serif; font-size:.8rem; font-weight:800; color:#fff;
  background:linear-gradient(135deg, var(--accent-0), var(--success));
  box-shadow:0 4px 14px var(--accent-glow);
}
.hub-header-title .grad{
  background:linear-gradient(90deg, var(--accent-0), var(--success));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hub-header-sub{ color:var(--text-1); font-size:.9rem; margin-top:.1rem; }

/* ---------- Stats strip (reduces the "empty" first impression) ---------- */
.stats-strip{ display:flex; gap:.7rem; flex-wrap:wrap; margin-bottom:1.3rem; }
.stat-pill{
  display:flex; flex-direction:column; gap:.1rem;
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:.6rem 1rem; min-width:130px; backdrop-filter:blur(10px);
  transition:transform .15s ease, border-color .15s ease;
}
.stat-pill:hover{ transform:translateY(-2px); border-color:var(--accent-1); }
.stat-num{ font-family:'JetBrains Mono', monospace; font-size:1.25rem; font-weight:700; color:var(--text-0); }
.stat-label{ font-size:.7rem; color:var(--text-2); text-transform:uppercase; letter-spacing:.06em; }

/* ---------- Suggested question chips (empty chat state) ---------- */
.suggested-label{ color:var(--text-1); font-size:.85rem; margin:.3rem 0 .6rem 0; font-weight:600; }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"]{
  background:var(--panel) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius-lg) !important;
  backdrop-filter:blur(14px);
  padding:.9rem 1.1rem !important;
  margin-bottom:.85rem !important;
  animation:hubFadeUp .32s ease both;
}
@keyframes hubFadeUp{
  from{ opacity:0; transform:translateY(8px); }
  to{ opacity:1; transform:translateY(0); }
}

/* ---------- Citation chip inside answer text ---------- */
.cite-chip{
  display:inline-flex; align-items:center; gap:.3rem;
  font-family:'JetBrains Mono', monospace; font-size:.72rem;
  padding:.08rem .5rem; margin:0 .1rem; border-radius:999px;
  background:rgba(255,255,255,.05); border:1px solid var(--border-strong);
  color:var(--text-1); vertical-align:middle;
}
.cite-dot{ width:6px; height:6px; border-radius:50%; flex:none; }

/* ---------- Source cards ---------- */
.src-card{
  display:flex; align-items:center; gap:.75rem;
  background:var(--panel-solid); border:1px solid var(--border);
  border-left:3px solid var(--accent-0);
  border-radius:var(--radius-md); padding:.65rem .9rem; margin-bottom:.5rem;
  transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}
.src-card:hover{
  transform:translateX(3px); border-color:var(--border-strong);
  box-shadow:0 8px 22px rgba(0,0,0,.32);
}
.src-icon{ font-size:1.2rem; flex:none; }
.src-body{ flex:1; min-width:0; }
.src-title{ font-weight:600; font-size:.88rem; color:var(--text-0); }
.src-link{ font-size:.74rem; color:var(--text-2); text-decoration:none; word-break:break-all; }
.src-link:hover{ color:var(--accent-1); }
.src-badge{
  font-family:'JetBrains Mono', monospace; font-size:.68rem;
  padding:.15rem .5rem; border-radius:999px; flex:none;
  background:rgba(52,211,153,.12); color:#6ee7b7; border:1px solid rgba(52,211,153,.25);
}

/* ---------- Generic glass card (Knowledge Base / Analytics / About) ---------- */
.glass-card{
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius-lg);
  padding:1.1rem 1.3rem; backdrop-filter:blur(14px); margin-bottom:.9rem;
}
.kb-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(220px,1fr)); gap:.85rem; }
.kb-card{
  background:var(--panel-solid); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:1rem; border-top:3px solid var(--accent-0); transition:transform .15s ease;
}
.kb-card:hover{ transform:translateY(-3px); }
.kb-card .kb-icon{ font-size:1.5rem; }
.kb-card .kb-name{ font-weight:700; margin-top:.35rem; }
.kb-card .kb-stat{ color:var(--text-1); font-size:.82rem; margin-top:.15rem; }

.empty-state{
  text-align:center; padding:2.4rem 1rem; color:var(--text-1);
  border:1px dashed var(--border-strong); border-radius:var(--radius-lg);
}
.empty-state .emoji{ font-size:2rem; }

/* ---------- Tabs (Sources / Retrieved Chunks / Details) ---------- */
[data-baseweb="tab-list"]{ gap:.3rem; border-bottom:1px solid var(--border) !important; }
[data-baseweb="tab"]{
  color:var(--text-1) !important; font-weight:600 !important; font-size:.84rem !important;
}
[aria-selected="true"][data-baseweb="tab"]{ color:var(--accent-1) !important; }
[data-baseweb="tab-highlight"]{ background-color:var(--accent-0) !important; }

/* ---------- Chat input ---------- */
[data-testid="stChatInput"]{
  border-radius:14px !important; border:1px solid var(--border-strong) !important;
  background:var(--panel-solid) !important;
}

/* ---------- Misc ---------- */
hr{ border-color:var(--border) !important; }
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-thumb{ background:#242b3d; border-radius:8px; }
:focus-visible{ outline:2px solid var(--accent-1) !important; outline-offset:2px; }

@media (max-width: 640px){
  .hub-header-title{ font-size:1.35rem; }
  .kb-grid{ grid-template-columns:1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retriever…")
def load_retriever():
    return retrieve_module.HybridRetriever()


@st.cache_data(show_spinner=False)
def available_sites() -> list[str]:
    if not DATA_RAW_DIR.exists():
        return list(SITE_META.keys())
    dirs = [d.name for d in DATA_RAW_DIR.iterdir() if d.is_dir() and any(d.glob("*.txt"))]
    return sorted(dirs) or list(SITE_META.keys())


@st.cache_data(show_spinner=False)
def knowledge_base_stats() -> dict:
    """Returns {site: {"files": n}} from data/raw/, used on the Knowledge Base page."""
    stats = {}
    if not DATA_RAW_DIR.exists():
        return stats
    for d in DATA_RAW_DIR.iterdir():
        if d.is_dir():
            n = len(list(d.glob("*.txt")))
            if n:
                stats[d.name] = {"files": n}
    return stats


@st.cache_data(show_spinner=False)
def corpus_stats() -> dict:
    """Quick counts for the stats strip under the header: total chunks,
    distinct sites, distinct source pages — used to make the app feel
    substantive even before the user asks anything."""
    chunks_path = DATA_PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        return {"chunks": 0, "sites": 0, "pages": 0}

    sites = set()
    pages = set()
    n_chunks = 0
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            n_chunks += 1
            try:
                rec = json.loads(line)
                sites.add(rec["metadata"].get("site", ""))
                pages.add(rec["metadata"].get("file_name", ""))
            except Exception:
                continue
    return {"chunks": n_chunks, "sites": len(sites), "pages": len(pages)}


def render_answer_with_chips(answer_text: str, context_chunks: list[dict]) -> str:
    """بتحوّل [Source: title] جوه نص الإجابة لـ chip ملوّن حسب المصدر."""
    title_to_site = {
        c["metadata"].get("title", "").strip().lower(): c["metadata"].get("site", "")
        for c in context_chunks
    }

    def _replace(match):
        title = match.group(1).strip()
        site = title_to_site.get(title.lower(), "")
        meta = site_meta(site)
        return (
            f'<span class="cite-chip">'
            f'<span class="cite-dot" style="background:{meta["color"]}"></span>{title}'
            f"</span>"
        )

    escaped = (
        answer_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return re.sub(r"\[Source:\s*([^\]]+?)\s*\]", _replace, escaped)


def run_query(query: str, style: str, top_k: int, selected_sites: set):
    retriever = load_retriever()
    candidates = retriever.retrieve(query, top_k=top_k)
    if selected_sites:
        candidates = [c for c in candidates if c["metadata"].get("site") in selected_sites]
    prefer_code = generate_module.STYLE_PREFER_CODE.get(style)  # None = زي القديم بالظبط
    context_package = context_module.build_context(candidates, prefer_code=prefer_code)
    result = generate_module.generate_answer(query, context_package, style=style)
    return result, context_package


# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="hub-brand">
            <div class="hub-brand-icon">AI</div>
            <div>
                <div class="hub-brand-title">AI Learning Hub</div>
                <div class="hub-brand-sub">Your AI Learning Companion</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hub-section-label">Navigation</div>', unsafe_allow_html=True)
    nav_options = {
        "chat": "💬  Chat",
        "compare": "⚖️  Compare Modes",
        "kb": "📚  Knowledge Base",
        "analytics": "📊  Analytics",
        "settings": "⚙️  Settings",
        "about": "ℹ️  About",
    }
    page = st.radio(
        "nav", list(nav_options.keys()), format_func=lambda k: nav_options[k],
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown('<div class="hub-section-label">Settings</div>', unsafe_allow_html=True)
    model_label = GEMINI_MODEL if MODEL_PROVIDER == "gemini" else OPENAI_MODEL
    st.selectbox("LLM Provider", [f"{MODEL_PROVIDER} · {model_label}"], disabled=True)
    prompt_style = st.selectbox(
        "Answer Mode",
        ["better", "code_examples", "explanation_only", "code_only", "strict", "weak"],
        index=0,
        format_func=lambda s: {
            "better": "Balanced (explanation + citations)",
            "code_examples": "Explanation + code examples",
            "explanation_only": "Explanation only (no code)",
            "code_only": "Code only",
            "strict": "Strict (explanation + confidence score)",
            "weak": "Weak (minimal rules)",
        }[s],
    )
    top_k = st.slider("Top K (retrieved chunks)", min_value=1, max_value=10, value=TOP_K)

    st.markdown('<div class="hub-section-label">Knowledge Base</div>', unsafe_allow_html=True)
    sites = available_sites()
    selected_sites = set()
    for s in sites:
        meta = site_meta(s)
        checked = st.checkbox(f"{meta['icon']} {meta['label']}", value=True, key=f"site_{s}")
        if checked:
            selected_sites.add(s)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hub-danger">', unsafe_allow_html=True)
    if st.button("🗑️  Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# تأكيد إن الـ vector store جاهز
# ──────────────────────────────────────────────────────────────────────────

try:
    load_retriever()
    retriever_ready = True
except FileNotFoundError as e:
    retriever_ready = False
    retriever_error = str(e)


# ──────────────────────────────────────────────────────────────────────────
# Header (shown on every page) + stats strip
# ──────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hub-header">
        <div>
            <div class="hub-header-title"><span class="hub-header-mark">AI</span> <span class="grad">Learning Hub</span></div>
            <div class="hub-header-sub">Ask anything about AI, NLP, RAG, LangChain, FAISS, Transformers and more…</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not retriever_ready:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="emoji">🧩</div>
            <h3>No vector store found</h3>
            <p>{retriever_error}</p>
            <p>Run the pipeline in order: <code>00 → 01 → 02 → 03 → 04 → 05</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

_stats = corpus_stats()
st.markdown(
    f"""
    <div class="stats-strip">
        <div class="stat-pill"><span class="stat-num">{_stats['chunks']:,}</span><span class="stat-label">Chunks indexed</span></div>
        <div class="stat-pill"><span class="stat-num">{_stats['pages']:,}</span><span class="stat-label">Source pages</span></div>
        <div class="stat-pill"><span class="stat-num">{_stats['sites']}</span><span class="stat-label">Knowledge sources</span></div>
        <div class="stat-pill"><span class="stat-num">{len(st.session_state.messages)//2}</span><span class="stat-label">Questions asked</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Page: Chat
# ──────────────────────────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "What is FAISS?",
    "How does long-term memory work in LangChain agents?",
    "What is tokenization?",
    "What's the difference between BM25 and embeddings?",
]


def process_query(query_text: str):
    st.session_state.messages.append(
        {"role": "user", "content": query_text, "timestamp": datetime.now().strftime("%H:%M")}
    )
    with st.spinner("Searching sources…"):
        result, context_package = run_query(query_text, prompt_style, top_k, selected_sites)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "context_chunks": context_package["chunks"],
            "style": prompt_style,
            "top_k": top_k,
            "total_words": context_package["total_words"],
            "dropped_outdated": context_package["dropped_outdated"],
            "dropped_duplicates": context_package["dropped_duplicates"],
            "timestamp": datetime.now().strftime("%H:%M"),
        }
    )
    st.rerun()


if page == "chat":
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                chips = render_answer_with_chips(msg["content"], msg.get("context_chunks", []))
                st.markdown(chips, unsafe_allow_html=True)

                sources = msg.get("sources", [])
                chunks = msg.get("context_chunks", [])
                tab_sources, tab_chunks, tab_details = st.tabs(
                    ["Sources", "Retrieved Chunks", "Details"]
                )

                with tab_sources:
                    if not sources:
                        st.caption("No sources were cited in this answer.")
                    for s in sources:
                        site = next(
                            (c["metadata"].get("site", "") for c in chunks
                             if c["metadata"].get("title", "").strip().lower()
                             == s["title"].strip().lower()),
                            "",
                        )
                        meta = site_meta(site)
                        st.markdown(
                            f"""
                            <div class="src-card" style="border-left-color:{meta['color']}">
                                <div class="src-icon">{meta['icon']}</div>
                                <div class="src-body">
                                    <div class="src-title">{s['title']}</div>
                                    <a class="src-link" href="{s['url']}" target="_blank">{s['url']}</a>
                                </div>
                                <div class="src-badge">{meta['label']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with tab_chunks:
                    if not chunks:
                        st.caption("No chunks were retrieved.")
                    for i, c in enumerate(chunks, 1):
                        meta = site_meta(c["metadata"].get("site", ""))
                        with st.expander(
                            f"{meta['icon']} #{i} — {c['metadata'].get('title', '')} "
                            f"(score={c.get('hybrid_score', 0):.3f})"
                        ):
                            st.caption(
                                f"lexical={c.get('lexical_score', 0):.3f} · "
                                f"semantic={c.get('semantic_score', 0):.3f}"
                            )
                            st.write(c["text"])

                with tab_details:
                    st.markdown(
                        f"""
                        - **Answer mode:** `{msg.get('style', '-')}`
                        - **Top K:** `{msg.get('top_k', '-')}`
                        - **Context words:** `{msg.get('total_words', '-')}`
                        - **Dropped (outdated / duplicates):**
                          `{msg.get('dropped_outdated', 0)}` / `{msg.get('dropped_duplicates', 0)}`
                        - **Timestamp:** `{msg.get('timestamp', '-')}`
                        """
                    )

    query = st.chat_input("Ask your question here…")

    if query and query.strip():
        process_query(query.strip())

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="emoji">💭</div>
                <h3>Start with a question</h3>
                <p>Try one of the suggestions below, or type your own in the box at the bottom.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="suggested-label">Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, sq in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(sq, use_container_width=True, key=f"suggested_{i}"):
                process_query(sq)


# ──────────────────────────────────────────────────────────────────────────
# Page: Compare Modes — 2 chat boxes side by side (code vs explanation)
# ──────────────────────────────────────────────────────────────────────────

elif page == "compare":
    if "compare_explanation" not in st.session_state:
        st.session_state.compare_explanation = []
    if "compare_code" not in st.session_state:
        st.session_state.compare_code = []

    st.markdown(
        """
        <div class="empty-state" style="padding:1.1rem 1rem; margin-bottom:1.2rem;">
            <p style="margin:0;">Ask one question, get two answers side by side —
            one <b>explanation only</b>, one <b>code examples</b> — from two
            separate context builders and prompts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def render_compare_message(msg: dict, accent: str):
        if msg["role"] == "user":
            st.markdown(
                f"""<div class="msg-row" style="margin-bottom:.6rem;">
                    <div style="font-weight:700; font-size:.85rem; color:var(--text-1);">🧑 You</div>
                </div>
                <div style="color:var(--text-0); margin-bottom:.9rem;">{msg['content']}</div>""",
                unsafe_allow_html=True,
            )
        else:
            chips = render_answer_with_chips(msg["content"], msg.get("context_chunks", []))
            st.markdown(
                f"""<div class="msg-row" style="margin-bottom:.6rem;">
                    <div style="font-weight:700; font-size:.85rem; color:{accent};">✨ AI</div>
                </div>
                <div style="color:var(--text-0); margin-bottom:1.1rem;">{chips}</div>""",
                unsafe_allow_html=True,
            )
            n_sources = len(msg.get("sources", []))
            st.caption(f"📎 {n_sources} source(s) cited")

    col_explain, col_code = st.columns(2)

    with col_explain:
        st.markdown(
            '<div class="hub-section-label" style="color:#67e8f9;">💡 Explanation Only</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="glass-card" style="min-height:220px;">', unsafe_allow_html=True)
        if not st.session_state.compare_explanation:
            st.caption("No messages yet.")
        for msg in st.session_state.compare_explanation:
            render_compare_message(msg, "#67e8f9")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_code:
        st.markdown(
            '<div class="hub-section-label" style="color:#fb923c;">💻 Code Examples</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="glass-card" style="min-height:220px;">', unsafe_allow_html=True)
        if not st.session_state.compare_code:
            st.caption("No messages yet.")
        for msg in st.session_state.compare_code:
            render_compare_message(msg, "#fb923c")
        st.markdown("</div>", unsafe_allow_html=True)

    compare_query = st.chat_input(
        "Ask a question — you'll get both an explanation and a code example…",
        key="compare_input",
    )

    if compare_query and compare_query.strip():
        q = compare_query.strip()
        ts = datetime.now().strftime("%H:%M")

        st.session_state.compare_explanation.append({"role": "user", "content": q, "timestamp": ts})
        st.session_state.compare_code.append({"role": "user", "content": q, "timestamp": ts})

        with st.spinner("Generating both answers…"):
            result_exp, ctx_exp = run_query(q, "explanation_only", top_k, selected_sites)
            result_code, ctx_code = run_query(q, "code_examples", top_k, selected_sites)

        st.session_state.compare_explanation.append(
            {
                "role": "assistant",
                "content": result_exp["answer"],
                "sources": result_exp["sources"],
                "context_chunks": ctx_exp["chunks"],
                "timestamp": ts,
            }
        )
        st.session_state.compare_code.append(
            {
                "role": "assistant",
                "content": result_code["answer"],
                "sources": result_code["sources"],
                "context_chunks": ctx_code["chunks"],
                "timestamp": ts,
            }
        )
        st.rerun()

elif page == "kb":
    stats = knowledge_base_stats()
    if not stats:
        st.markdown(
            """
            <div class="empty-state">
                <div class="emoji">📦</div>
                <h3>No data collected yet</h3>
                <p>Run <code>python 00_data_collection.py</code> to collect the sources.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        cards = ""
        for site, s in sorted(stats.items()):
            meta = site_meta(site)
            cards += f"""
            <div class="kb-card" style="border-top-color:{meta['color']}">
                <div class="kb-icon">{meta['icon']}</div>
                <div class="kb-name">{meta['label']}</div>
                <div class="kb-stat">{s['files']} pages collected</div>
            </div>
            """
        st.markdown(f'<div class="kb-grid">{cards}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Page: Analytics
# ──────────────────────────────────────────────────────────────────────────

elif page == "analytics":
    metrics_path = EVALUATION_DIR / "metrics_report.csv"
    if not metrics_path.exists():
        st.markdown(
            """
            <div class="empty-state">
                <div class="emoji">📊</div>
                <h3>No evaluation data yet</h3>
                <p>Run <code>python 08_evaluate_retrieval.py</code> to generate a comparison
                of BM25 vs Embeddings vs Hybrid.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        import pandas as pd

        df = pd.read_csv(metrics_path)
        metric_cols = [c for c in df.columns if c not in ("query", "retriever")]
        summary = df.groupby("retriever")[metric_cols].mean()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Retriever comparison (average across all questions)")
        st.bar_chart(summary)
        st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Page: Settings
# ──────────────────────────────────────────────────────────────────────────

elif page == "settings":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Current pipeline settings (from `.env`)")
    st.markdown(
        f"""
        | Setting | Value |
        |---|---|
        | Model provider | `{MODEL_PROVIDER}` |
        | Model | `{model_label}` |
        | Embedding model | `{EMBEDDING_MODEL}` |
        | Hybrid alpha | `{HYBRID_ALPHA}` |
        | Top K (default) | `{TOP_K}` |
        """
    )
    st.caption("To change these permanently, edit the `.env` file and restart the app.")
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Page: About
# ──────────────────────────────────────────────────────────────────────────

elif page == "about":
    st.markdown(
        """
        <div class="glass-card">
        <h4>About this project</h4>
        <p>AI Learning Hub is a RAG assistant that answers AI/NLP/RAG questions
        from real documentation (LangChain, Sentence-Transformers, FAISS,
        scikit-learn, NLTK), citing its source for every answer.</p>
        <p><b>Pipeline:</b> Scraping → Preprocessing → Chunking → Embeddings →
        Vector Store (ChromaDB) → Hybrid Retrieval (BM25 + Embeddings) →
        Context Building → Generation (OpenAI/Gemini).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
