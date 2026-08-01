"""
09_log_failures.py
----------------------
Phase 9 — Failure Analysis (جديد، مبني على Lab 9)

بيشغّل الـ pipeline الكامل (retrieve → context → generate) على أسئلة
evaluation/eval_queries.json، وأي سؤال تفشل فيه أي مرحلة، بيتسجل صف
في evaluation/failure_log.csv موضّح فيه:
    query, failed_layer (retrieval/context/prompt/generation), expected, got, notes

الفكرة الأساسية من Lab 9: "Fix the layer, not the symptom" —
يعني قبل ما تصلح، لازم تعرف المشكلة حصلت في أنهي طبقة بالظبط.

تشغيل:
    python 09_log_failures.py

ملاحظة: فيه فاصل ~4.5 ثانية بين كل سؤال وسؤال (احترام rate limit الـ
free tier)، فتشغيل الملف على 18 سؤال هياخد دقيقة ونص تقريبًا بدل ثواني.
"""

import json
import time
from importlib import import_module

from utils.config import EVALUATION_DIR, TOP_K
from utils.logging_utils import get_logger, log_failure

logger = get_logger("09_log_failures")

# Free tier عادةً بيدّي 15 طلب/دقيقة. الفاصل ده (~4.5 ثانية) بيخلينا تحت
# الحد بأمان (~13 طلب/دقيقة) حتى لو الأسئلة كتير، من غير ما نستهلك وقت
# زيادة عن اللازم.
GENERATION_DELAY_SECONDS = 4.5


def _title_matches(chunk_title: str, relevant_titles: list[str]) -> bool:
    chunk_title = chunk_title.lower()
    return any(rt.lower() in chunk_title for rt in relevant_titles)


def main():
    eval_path = EVALUATION_DIR / "eval_queries.json"
    if not eval_path.exists():
        logger.warning(f"{eval_path} مش موجود.")
        return

    with open(eval_path, encoding="utf-8") as f:
        eval_queries = json.load(f)

    retrieve_module = import_module("06_retrieve_context")
    context_module = import_module("06b_build_context")
    generate_module = import_module("07_generate_answer")

    retriever = retrieve_module.HybridRetriever()

    n_failures = 0
    for item in eval_queries:
        query = item["query"]
        relevant_titles = item["relevant_titles"]

        # --- طبقة الـ retrieval ---
        candidates = retriever.retrieve(query, top_k=TOP_K)
        retrieved_titles = [c["metadata"].get("title", "") for c in candidates]
        found_relevant = any(_title_matches(t, relevant_titles) for t in retrieved_titles)

        if not found_relevant:
            log_failure(
                query=query,
                failed_layer="retrieval",
                expected=", ".join(relevant_titles),
                got=", ".join(retrieved_titles) or "(فاضي)",
                notes="ولا عنوان من المتوقعين ظهر في أول top_k نتايج",
            )
            n_failures += 1
            continue  # مفيش داعي نكمل للـ context/generation لو الـ retrieval فشل

        # --- طبقة الـ context ---
        context_package = context_module.build_context(candidates)
        if not context_package["chunks"]:
            log_failure(
                query=query,
                failed_layer="context",
                expected=", ".join(relevant_titles),
                notes="كل الـ candidates اتشالوا في الفلترة (outdated/duplicates/budget)",
            )
            n_failures += 1
            continue

        # --- طبقة الـ generation ---
        result = generate_module.generate_answer(query, context_package, style="better")
        if result.get("error"):
            # generate_answer() سجّلت الفشل ده بنفسها (log_failure) وقت ما حصل،
            # هنا بنعدّه بس في الملخص عشان الرقم المطبوع يبقى مطابق للملف فعليًا
            n_failures += 1
        elif "معنديش معلومات" in result["answer"]:
            log_failure(
                query=query,
                failed_layer="generation",
                expected=", ".join(relevant_titles),
                got=result["answer"][:200],
                notes="فيه context لكن الموديل رفض/فشل يجاوب",
            )
            n_failures += 1

        time.sleep(GENERATION_DELAY_SECONDS)  # نحترم rate limit الـ free tier

    logger.info(f"خلصنا. {n_failures} فشل من أصل {len(eval_queries)} سؤال.")
    logger.info(f"التفاصيل في {EVALUATION_DIR / 'failure_log.csv'}")


if __name__ == "__main__":
    main()
