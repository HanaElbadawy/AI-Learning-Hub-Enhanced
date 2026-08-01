"""
07_generate_answer.py
------------------------
Phase 7 — Answer Generation
"""

import re
import sys
import time

from utils.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
    require_provider_key,
)
from utils.logging_utils import get_logger, log_failure

logger = get_logger("07_generate_answer")


PROMPT_TEMPLATES = {
    "weak": """Answer the following question using the context below.

Context:
{context}

Question: {question}
Answer:""",
    "better": """You are a helpful assistant answering questions about AI, NLP, and RAG \
using ONLY the provided context.

Rules:
- Answer only using facts present in the context below.
- If the context does not contain the answer, say clearly that you don't have enough information.
- Write the answer as flowing, natural paragraphs (like a knowledgeable person explaining \
the topic), NOT as a bullet-point list. Only use a bulleted/numbered list if the content \
is genuinely a sequence of steps or a set of clearly parallel items the user asked to \
compare — plain explanations should read as connected sentences.
- Cite the source title for every claim you make, in the form [Source: <title>], placed \
naturally at the end of the sentence it supports — the title ONLY, do not include the \
URL inside the brackets.
- Do not use any outside/prior knowledge.

Context:
{context}

Question: {question}

Answer (flowing paragraphs, with inline citations):""",
    "strict": """You are a grounded RAG assistant. You must answer using ONLY the context below.

Context:
{context}

Question: {question}

Respond in exactly this two-part format:
1) ANSWER: <your answer, with inline [Source: <title>] citations for every claim — \
title ONLY inside the brackets, never the URL>
2) CONFIDENCE: <"high" if the context clearly answers the question, "low" if it is \
partial or missing, followed by a one-sentence reason>

If the context has no relevant information at all, ANSWER must say so explicitly and \
CONFIDENCE must be "low".""",
    "code_examples": """You are a helpful assistant answering questions about AI, NLP, and RAG \
using ONLY the provided context.

Rules:
- Answer only using facts present in the context below.
- If the context does not contain the answer, say clearly that you don't have enough information.
- Give a short explanation in flowing prose, AND include relevant code snippet(s) from the \
context (in fenced code blocks, e.g. ```python ... ```) whenever the context contains code \
that illustrates the answer. Prefer showing real code over describing it in words.
- Cite the source title for every claim/snippet, in the form [Source: <title>] — the \
title ONLY, no URL inside the brackets. For code blocks, put the citation as a comment on \
the line right above the code (e.g. "# [Source: <title>]").
- Do not use any outside/prior knowledge.

Context:
{context}

Question: {question}

Answer (explanation + code examples, with citations):""",
    "explanation_only": """You are a helpful assistant answering questions about AI, NLP, and RAG \
using ONLY the provided context.

Rules:
- Answer only using facts present in the context below.
- If the context does not contain the answer, say clearly that you don't have enough information.
- Explain the concept in plain, flowing prose ONLY. Do NOT include any code, code blocks, \
function names as code, or command-line snippets — even if the context contains code, \
describe in words what it does instead of showing it.
- Write as connected sentences/paragraphs, not bullet points, unless describing a genuine \
sequence of steps in words.
- Cite the source title for every claim, in the form [Source: <title>] — title ONLY, no URL.
- Do not use any outside/prior knowledge.

Context:
{context}

Question: {question}

Answer (plain-language explanation, no code, with citations):""",
    "code_only": """You are a code-focused assistant answering questions about AI, NLP, and RAG \
using ONLY the provided context.

Rules:
- Respond with ONLY code — extracted or adapted from the context below. No paragraphs of \
explanation before or after the code.
- You may include a single one-line comment above the code block if needed for context \
(e.g. "# Creating an agent with tools"), but nothing more.
- Put the source citation as a code comment directly above the relevant code, in the form \
"# [Source: <title>]" — title ONLY, no URL.
- If the context does not contain any code relevant to the question, respond with exactly \
one comment line: "# Not enough code examples in the available context for this question."
- Do not use any outside/prior knowledge, and do not invent code that isn't grounded in \
the context.

Context:
{context}

Question: {question}

Answer (code only):""",
}

# كل وضع، هل يفضّل context مليان كود ولا نثر بس؟ None = مفيش تفضيل (زي القديم)
STYLE_PREFER_CODE = {
    "code_examples": True,
    "code_only": True,
    "explanation_only": False,
}


def build_prompt(question: str, context_text: str, style: str = "better") -> str:
    if style not in PROMPT_TEMPLATES:
        raise ValueError(f"style لازم يكون واحد من {list(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[style].format(context=context_text, question=question)


CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+?)\s*\]", re.IGNORECASE)


def extract_cited_titles(answer_text: str) -> set[str]:
    """بتستخرج كل عناوين المصادر اللي الموديل استشهد بيها فعليًا جوه
    نص الإجابة، بالشكل [Source: <title>]. بنرجعها lowercase عشان المقارنة
    تبقى case-insensitive.

    دفاع إضافي: لو الموديل (بالغلط) حط الـ URL جوه نفس القوس زي
    "[Source: title | url]" أو "[Source: title, url]"، بنقص أي حاجة بعد
    أول | أو , أو مسافة+http عشان نستخرج التيتل بس ونضمن المطابقة."""
    raw_matches = CITATION_PATTERN.findall(answer_text)
    titles = set()
    for m in raw_matches:
        cleaned = re.split(r"\s*[|,]\s*|\s+https?://", m, maxsplit=1)[0]
        titles.add(cleaned.strip().lower())
    return titles


CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE:\s*[\"']?(high|low)", re.IGNORECASE)


def extract_model_confidence(answer_text: str) -> str | None:
    """بتستخرج CONFIDENCE: high/low من إجابات أسلوب 'strict' بس. لو الأسلوب
    مش strict أو الموديل ماكتبهاش، بترجع None (الواجهة بترجع لتقدير تاني)."""
    m = CONFIDENCE_PATTERN.search(answer_text)
    return m.group(1).lower() if m else None


def estimate_confidence_from_retrieval(chunks: list[dict]) -> str:
    """تقدير احتياطي لو مفيش CONFIDENCE صريح من الموديل، مبني على أعلى
    hybrid_score في الـ chunks المسترجعة. ده تقدير مش قياس دقيق."""
    if not chunks:
        return "low"
    top_score = max(c.get("hybrid_score", 0.0) for c in chunks)
    if top_score >= 0.55:
        return "high"
    if top_score >= 0.35:
        return "medium"
    return "low"


def _call_openai(prompt: str, temperature: float = 0.2) -> tuple[str, dict | None]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = response.choices[0].message.content

    usage = None
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return text, usage


def _call_gemini(prompt: str, temperature: float = 0.2) -> tuple[str, dict | None]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    text = response.text

    usage = None
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        # أسماء الحقول دي ممكن تتغيّر مع نسخ الـ SDK، فبنقراها بأمان
        prompt_tokens = getattr(um, "prompt_token_count", None)
        completion_tokens = getattr(um, "candidates_token_count", None)
        total_tokens = getattr(um, "total_token_count", None)
        if prompt_tokens is not None or total_tokens is not None:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    return text, usage


def call_llm(prompt: str, temperature: float = 0.2) -> tuple[str, dict | None]:
    """بتوجّه النداء لـ OpenAI أو Gemini حسب MODEL_PROVIDER في .env.
    بترجع (answer_text, usage_dict_or_none)."""
    if MODEL_PROVIDER == "gemini":
        return _call_gemini(prompt, temperature)
    return _call_openai(prompt, temperature)


def generate_answer(
    question: str,
    context_package: dict,
    style: str = "better",
    temperature: float = 0.2,
) -> dict:
    """
    context_package: ناتج build_context() من 06b_build_context.py
    بيرجّع dict فيها answer, sources, prompt_style, raw_prompt, generation_time,
    usage (tokens لو الـ API رجّعها), confidence.
    """
    require_provider_key()

    if not context_package["chunks"]:
        log_failure(
            query=question,
            failed_layer="context",
            notes="مفيش chunks بعد الـ context building — الـ retrieval رجّع فاضي أو كله outdated/مكرر",
        )
        return {
            "answer": "معنديش معلومات كفاية في المصادر المتاحة عشان أجاوب على السؤال ده.",
            "sources": [],
            "prompt_style": style,
            "generation_time": 0.0,
            "usage": None,
            "confidence": "low",
            "error": True,
        }

    prompt = build_prompt(question, context_package["context_text"], style)

    t_start = time.perf_counter()
    try:
        answer_text, usage = call_llm(prompt, temperature)
    except Exception as e:  # أي فشل في نداء الـ API (أي provider)
        logger.error(f"فشل نداء {MODEL_PROVIDER}: {e}")
        log_failure(query=question, failed_layer="generation", notes=str(e))
        return {
            "answer": "حصل خطأ أثناء توليد الإجابة. حاول تاني.",
            "sources": [],
            "prompt_style": style,
            "generation_time": time.perf_counter() - t_start,
            "usage": None,
            "confidence": "low",
            "error": True,
        }
    generation_time = time.perf_counter() - t_start

    cited_titles = extract_cited_titles(answer_text)

    if cited_titles:
        # بس المصادر اللي الموديل فعلاً استشهد بيها جوه الإجابة
        relevant_chunks = [
            c
            for c in context_package["chunks"]
            if c["metadata"].get("title", "").strip().lower() in cited_titles
        ]
    else:
        # مفيش citations (زي أسلوب "weak")، فنسيب كل الـ context كمرجع عام
        relevant_chunks = context_package["chunks"]

    sources = sorted(
        {
            (c["metadata"].get("title", ""), c["metadata"].get("source", ""))
            for c in relevant_chunks
        }
    )

    confidence = extract_model_confidence(answer_text) or estimate_confidence_from_retrieval(
        context_package["chunks"]
    )

    return {
        "answer": answer_text,
        "sources": [{"title": t, "url": u} for t, u in sources],
        "prompt_style": style,
        "raw_prompt": prompt,
        "generation_time": generation_time,
        "usage": usage,
        "confidence": confidence,
        "error": False,
    }


def main():
    from importlib import import_module

    retrieve_module = import_module("06_retrieve_context")
    context_module = import_module("06b_build_context")

    args = sys.argv[1:]
    style = "better"
    for a in list(args):
        if a.startswith("--mode="):
            style = a.split("=", 1)[1]
            args.remove(a)

    query = " ".join(args) or "What is retrieval augmented generation?"

    retriever = retrieve_module.HybridRetriever()
    candidates = retriever.retrieve(query, top_k=TOP_K)
    prefer_code = STYLE_PREFER_CODE.get(style)
    context_package = context_module.build_context(candidates, prefer_code=prefer_code)

    result = generate_answer(query, context_package, style=style)

    print(f"\n=== ANSWER (mode={style}) ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"- {s['title']}: {s['url']}")


if __name__ == "__main__":
    main()
