"""
03_chunking.py
----------------
Phase 3 — Chunking

بيقسّم كل document لـ chunks متداخلة (overlapping fixed-size chunks) —
نفس الأسلوب اللي اتعلمناه في Lab 9 — عشان نتجنب مشاكل زي:
- rule/exception split
- procedure split
- symptom/fix split

كل chunk بياخد metadata من الـ document الأصلي + ID فريد + رقم ترتيبه
جوه الصفحة، عشان نقدر بعدين نرجّعه لمكانه الأصلي.

تشغيل:
    python 03_chunking.py
"""

import json

from utils.config import DATA_PROCESSED_DIR
from utils.logging_utils import get_logger

logger = get_logger("03_chunking")

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # نتجنب infinite loop لو overlap >= chunk_size

    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def main():
    in_path = DATA_PROCESSED_DIR / "documents_clean.jsonl"
    if not in_path.exists():
        logger.warning(f"{in_path} مش موجود. شغل 02_preprocessing.py الأول.")
        return

    documents = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    all_chunks = []
    for doc_idx, doc in enumerate(documents):
        text_chunks = chunk_text(doc["page_content"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for chunk_idx, chunk_content in enumerate(text_chunks):
            chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_content,
                    "search_text": f"{doc['metadata'].get('title', '')}. {chunk_content}",
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_idx,
                        "is_current": True,  # افتراضيًا كل صفحة متجمعة دلوقتي = current
                    },
                }
            )

    out_path = DATA_PROCESSED_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        f"اتعمل {len(all_chunks)} chunk من {len(documents)} document "
        f"(chunk_size={CHUNK_SIZE_WORDS} كلمة, overlap={CHUNK_OVERLAP_WORDS} كلمة)."
    )
    logger.info(f"اتحفظوا في {out_path}")


if __name__ == "__main__":
    main()
