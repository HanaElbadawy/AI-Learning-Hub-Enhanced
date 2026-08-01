"""
04_embeddings.py
-------------------
Phase 4 — Embeddings

بيحوّل كل chunk["search_text"] لـ dense vector باستخدام SentenceTransformer
(نفس الموديل all-MiniLM-L6-v2 اللي استخدمناه في Lab 7 و Lab 8).

normalize_embeddings=True عشان cosine similarity = dot product (أسرع).

الناتج: data/processed/chunk_embeddings.npy (مصفوفة NumPy)
        + بنفس ترتيب الصفوف اللي في chunks.jsonl

تشغيل:
    python 04_embeddings.py
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import DATA_PROCESSED_DIR, EMBEDDING_MODEL
from utils.logging_utils import get_logger

logger = get_logger("04_embeddings")


def main():
    in_path = DATA_PROCESSED_DIR / "chunks.jsonl"
    if not in_path.exists():
        logger.warning(f"{in_path} مش موجود. شغل 03_chunking.py الأول.")
        return

    chunks = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"بنحمّل موديل الـ embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["search_text"] for c in chunks]
    logger.info(f"بنعمل embed لـ {len(texts)} chunk...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    out_path = DATA_PROCESSED_DIR / "chunk_embeddings.npy"
    np.save(out_path, embeddings)

    logger.info(f"Shape: {embeddings.shape} (rows=chunks, cols=dimensions)")
    logger.info(f"اتحفظوا في {out_path}")


if __name__ == "__main__":
    main()
