"""
02_preprocessing.py
---------------------
Phase 2 — Preprocessing

بيقرأ data/processed/documents.jsonl، وينضف كل page_content:
- إزالة سطور التكرار (نفس السطر ده كتير في docs: "Copy Page", "Search"...)
- إزالة whitespace زيادة
- إزالة الصفحات الفارغة أو القصيرة جدًا بعد التنضيف

ملاحظة مهمة (زي ما اتعلمنا في Lab 5):
مبنعملش lowercasing ولا إزالة punctuation هنا، لأن ده هيضر الـ retrieval
والـ LLM answer لاحقًا. التنضيف هنا structural (نضافة الصفحة) مش
linguistic (تبسيط الكلمات) - ده الفرق بين preprocessing لـ classification
و preprocessing لـ RAG.

تشغيل:
    python 02_preprocessing.py
"""

import json
import re

from utils.config import DATA_PROCESSED_DIR
from utils.logging_utils import get_logger

logger = get_logger("02_preprocessing")

# سطور شائعة في أي صفحة documentation ومالهاش قيمة كـ محتوى
BOILERPLATE_LINES = {
    "copy page", "search", "github", "edit this page", "was this page helpful?",
    "on this page", "previous", "next", "table of contents", "skip to main content",
}

MIN_CONTENT_LENGTH = 200  # حروف. أقل من كده يبقى الصفحة مش مفيدة


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate_lines(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if line.strip().lower() not in BOILERPLATE_LINES]
    return "\n".join(kept)


def remove_duplicate_consecutive_lines(text: str) -> str:
    lines = text.split("\n")
    deduped = []
    prev = None
    for line in lines:
        if line.strip() != prev:
            deduped.append(line)
        prev = line.strip()
    return "\n".join(deduped)


def clean_document_text(text: str) -> str:
    text = remove_boilerplate_lines(text)
    text = remove_duplicate_consecutive_lines(text)
    text = normalize_whitespace(text)
    return text


def main():
    in_path = DATA_PROCESSED_DIR / "documents.jsonl"
    if not in_path.exists():
        logger.warning(f"{in_path} مش موجود. شغل 01_documents.py الأول.")
        return

    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    cleaned_records = []
    dropped = 0
    for rec in records:
        cleaned_text = clean_document_text(rec["page_content"])
        if len(cleaned_text) < MIN_CONTENT_LENGTH:
            dropped += 1
            continue
        rec["page_content"] = cleaned_text
        cleaned_records.append(rec)

    out_path = DATA_PROCESSED_DIR / "documents_clean.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"اتنضف {len(cleaned_records)} document (اتشال {dropped} فاضي/قصير).")
    logger.info(f"اتحفظوا في {out_path}")


if __name__ == "__main__":
    main()
