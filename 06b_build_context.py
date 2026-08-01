"""
06b_build_context.py
-----------------------
Phase 6b — Context Building (جديد، مبني على Lab 8 Part 2 و Lab 9)

الـ retrieval بيرجّع "candidate evidence" مش context جاهز للاستخدام.
لو استخدمنا نتيجة الـ retrieval زي ما هي جوه الـ prompt ممكن يحصل:
- مصدر قديم (is_current=False) يتاخد كأنه حالي
- نفس المعلومة تتكرر من أكتر من chunk
- الـ prompt يكبر ويعدي الـ token budget

الملف ده بياخد نتيجة retrieve() ويرجّع "context package" نضيف:
    candidate chunks
      → filter outdated
      → remove near-duplicates
      → sort by (hybrid_score, recency)
      → apply word budget
      → label كل chunk بمصدره

تشغيل تجريبي:
    python 06b_build_context.py "What is BM25?"
"""

import re
import sys

from difflib import SequenceMatcher

from utils.config import TOP_K
from utils.logging_utils import get_logger

logger = get_logger("06b_build_context")

WORD_BUDGET = 1200  # أقصى عدد كلمات في الـ context الكلي اللي هيدخل الـ prompt
DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # لو نسبة التشابه أعلى من كده، نعتبرهم تكرار

# أنماط بتدل على إن السطر ده "كود" مش نثر عادي (heuristic بسيط، مش parser
# حقيقي — كفاية عشان نرتّب الأولوية، مش عشان نفصل بدقة 100%)
CODE_LINE_PATTERN = re.compile(
    r"```"
    r"|^\s*(def |class |import |from \s*\w+ import|const |let |var |function\s*\("
    r"|return |for \(|if \(|=>|;\s*$|^\s*[\}\{]\s*$)",
    re.MULTILINE,
)


def _is_near_duplicate(text_a: str, text_b: str) -> bool:
    ratio = SequenceMatcher(None, text_a, text_b).ratio()
    return ratio >= DUPLICATE_SIMILARITY_THRESHOLD


def filter_outdated(chunks: list[dict]) -> list[dict]:
    return [c for c in chunks if c["metadata"].get("is_current", True)]


def remove_near_duplicates(chunks: list[dict]) -> list[dict]:
    kept = []
    for chunk in chunks:
        is_dup = any(_is_near_duplicate(chunk["text"], kept_chunk["text"]) for kept_chunk in kept)
        if not is_dup:
            kept.append(chunk)
    return kept


def sort_by_relevance(chunks: list[dict]) -> list[dict]:
    # الأحدث بيتقدم لو فيه تعادل في الـ score (recency كـ tie-breaker)
    return sorted(
        chunks,
        key=lambda c: (c.get("hybrid_score", 0.0),),
        reverse=True,
    )


def code_density(text: str) -> float:
    """نسبة السطور اللي شكلها 'كود' من إجمالي سطور الـ chunk. رقم من 0
    (نثر بالكامل) لـ 1 (كود بالكامل)."""
    lines = text.splitlines() or [text]
    code_lines = sum(1 for line in lines if CODE_LINE_PATTERN.search(line))
    return code_lines / max(len(lines), 1)


def rerank_by_code_preference(chunks: list[dict], prefer_code: bool | None) -> list[dict]:
    """
    الـ 'context builder التاني': بيعيد ترتيب الـ chunks حسب نسبة الكود
    فيها، بالإضافة لـ hybrid_score الأصلي — مش بديل عنه.

    - prefer_code=True  → الأجزاء اللي فيها كود بتتقدّم (مفيد لأوضاع
      "code examples" و"code only")
    - prefer_code=False → الأجزاء اللي فيها كود بتتأخر، النثر بيتقدّم
      (مفيد لوضع "explanation only")
    - prefer_code=None  → مفيش تغيير خالص (السلوك الافتراضي القديم،
      عشان الكود القديم اللي بينادي build_context() من غير الباراميتر
      ده يفضل شغال زي ما هو تمامًا)
    """
    if prefer_code is None:
        return chunks

    def score(c):
        base = c.get("hybrid_score", 0.0)
        cd = code_density(c["text"])
        bonus = cd if prefer_code else -cd
        return base + 0.4 * bonus

    return sorted(chunks, key=score, reverse=True)


def apply_word_budget(chunks: list[dict], budget: int) -> list[dict]:
    kept = []
    total_words = 0
    for chunk in chunks:
        n_words = len(chunk["text"].split())
        if total_words + n_words > budget and kept:
            break
        kept.append(chunk)
        total_words += n_words
    return kept


def label_chunk(chunk: dict) -> dict:
    meta = chunk["metadata"]
    # مهم: القوس [Source: title] لازم يفضل title بس، مطابق تمامًا لتعليمات
    # الـ prompt في 07_generate_answer.py. لو حطينا الـ URL جوه نفس القوس،
    # الموديل بيقلّد الشكل ده وقت الاستشهاد، وده بيكسر مطابقة الاستشهادات
    # بعدين. الـ URL بنحطه في سطر منفصل بدل كده.
    title = meta.get("title", "unknown")
    url = meta.get("source", "")
    label = f"[Source: {title}]\n(URL: {url})"
    if not meta.get("is_current", True):
        label += " [OUTDATED]"
    chunk["label"] = label
    return chunk


def build_context(
    candidate_chunks: list[dict],
    word_budget: int = WORD_BUDGET,
    prefer_code: bool | None = None,
) -> dict:
    """
    بياخد نتيجة HybridRetriever.retrieve() ويرجّع context package:
        {
          "chunks": [...],       # الـ chunks النهائية بعد كل الفلترة
          "context_text": "...", # نص جاهز يتحط في الـ prompt
          "total_words": int,
          "dropped_outdated": int,
          "dropped_duplicates": int,
        }

    prefer_code: اختياري. True/False لتفعيل "context builder التاني"
    (بيرتب حسب نسبة الكود في كل chunk). None = السلوك الافتراضي زي ما هو
    (مفيش تغيير للكود القديم اللي بينادي الدالة من غير الباراميتر ده).
    """
    n_before_filter = len(candidate_chunks)
    current_chunks = filter_outdated(candidate_chunks)
    n_after_filter = len(current_chunks)

    deduped_chunks = remove_near_duplicates(current_chunks)
    n_after_dedupe = len(deduped_chunks)

    ordered_chunks = sort_by_relevance(deduped_chunks)
    ordered_chunks = rerank_by_code_preference(ordered_chunks, prefer_code)
    budgeted_chunks = apply_word_budget(ordered_chunks, word_budget)
    labeled_chunks = [label_chunk(c) for c in budgeted_chunks]

    context_text = "\n\n".join(f"{c['label']}\n{c['text']}" for c in labeled_chunks)
    total_words = sum(len(c["text"].split()) for c in labeled_chunks)

    return {
        "chunks": labeled_chunks,
        "context_text": context_text,
        "total_words": total_words,
        "dropped_outdated": n_before_filter - n_after_filter,
        "dropped_duplicates": n_after_filter - n_after_dedupe,
    }


def main():
    from importlib import import_module

    retrieve_module = import_module("06_retrieve_context")
    query = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"

    retriever = retrieve_module.HybridRetriever()
    candidates = retriever.retrieve(query, top_k=TOP_K)

    context = build_context(candidates)

    logger.info(
        f"Context package: {len(context['chunks'])} chunk, "
        f"{context['total_words']} كلمة، "
        f"اتشال {context['dropped_outdated']} outdated و {context['dropped_duplicates']} مكرر"
    )
    print("\n" + context["context_text"])


if __name__ == "__main__":
    main()
