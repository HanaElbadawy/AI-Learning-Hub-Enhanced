"""
05_create_vector_store.py
----------------------------
Phase 5 — Vector Store (ChromaDB)

بياخد chunks.jsonl + chunk_embeddings.npy ويخزنهم في ChromaDB persistent
collection، عشان منحتاجش نعيد حساب الـ embeddings كل مرة.

اخترنا ChromaDB (مش FAISS) عشان بيسمحلنا نخزن الـ metadata
(source, title, is_current...) جوه نفس الـ collection ونعمل filter
عليها وقت البحث — ده هيفيدنا في 06b_build_context.py.

تشغيل:
    python 05_create_vector_store.py
"""

import json

import chromadb
import numpy as np

from utils.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, DATA_PROCESSED_DIR
from utils.logging_utils import get_logger

logger = get_logger("05_create_vector_store")

CHROMA_BATCH_SIZE = 500  # chroma بترفض batches كبيرة جدًا مرة واحدة


def main():
    chunks_path = DATA_PROCESSED_DIR / "chunks.jsonl"
    embeddings_path = DATA_PROCESSED_DIR / "chunk_embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        logger.warning(
            "chunks.jsonl أو chunk_embeddings.npy مش موجودين. "
            "شغل 03_chunking.py و 04_embeddings.py الأول."
        )
        return

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    embeddings = np.load(embeddings_path)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"عدد الـ chunks ({len(chunks)}) مش متطابق مع عدد الـ embeddings "
            f"({embeddings.shape[0]}). شغل 04_embeddings.py تاني."
        )

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # لو الـ collection موجودة من قبل، نمسحها ونعمل واحدة جديدة (rebuild نضيف)
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info(f"اتمسحت نسخة قديمة من collection: {CHROMA_COLLECTION_NAME}")

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = dict(c["metadata"])
        # Chroma metadata لازم قيمها تكون str/int/float/bool بس
        meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}
        metadatas.append(meta)

    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
        logger.info(f"اتضاف batch {start}-{min(end, len(chunks))}")

    logger.info(
        f"تم بناء الـ vector store بـ {collection.count()} chunk في {CHROMA_PERSIST_DIR}"
    )


if __name__ == "__main__":
    main()
