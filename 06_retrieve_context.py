"""
06_retrieve_context.py
-------------------------
Phase 6 — Retrieval (Hybrid: BM25 + Embeddings)

زي ما شفنا في Lab 8: لا الـ lexical (BM25) ولا الـ semantic (embeddings)
كويس في كل حاجة لوحده:
- BM25 قوي في الكلمات المضبوطة (function names, error codes...)
- Embeddings قوي في الترادف والمعنى

فبنعمل hybrid: نجيب مرشحين من الاتنين، ننرمل السكورات، نجمعهم بوزن
HYBRID_ALPHA (0 = BM25 بس, 1 = embeddings بس, 0.5 = بالنص بالنص).

الملف ده بيصدّر دالة retrieve() تستخدمها باقي الملفات (06b, streamlit_app).
لو شغلته لوحده هيعمل demo query من الـ command line.

تشغيل:
    python 06_retrieve_context.py "What is BM25?"
"""

import json
import sys
import time

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DATA_PROCESSED_DIR,
    EMBEDDING_MODEL,
    HYBRID_ALPHA,
    TOP_K,
)
from utils.logging_utils import get_logger

logger = get_logger("06_retrieve_context")


class HybridRetriever:
    def __init__(self):
        self._load_chunks()
        self._build_bm25_index()
        self._connect_chroma()
        self._embedding_model = None  # lazy load، عشان الاستيراد يبقى سريع

    def _load_chunks(self):
        chunks_path = DATA_PROCESSED_DIR / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} مش موجود. شغل 03_chunking.py الأول.")

        self.chunks_by_id = {}
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks_by_id[chunk["chunk_id"]] = chunk

    def _build_bm25_index(self):
        self.chunk_ids = list(self.chunks_by_id.keys())
        tokenized_corpus = [
            self.chunks_by_id[cid]["search_text"].lower().split() for cid in self.chunk_ids
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _connect_chroma(self):
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = client.get_collection(CHROMA_COLLECTION_NAME)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def _bm25_candidates(self, query: str, top_n: int) -> dict:
        tokenized_query = query.lower().split()
        scores = np.array(self.bm25.get_scores(tokenized_query))
        top_idx = np.argsort(scores)[::-1][:top_n]
        norm_scores = self._min_max_normalize(scores[top_idx])
        return {self.chunk_ids[i]: float(s) for i, s in zip(top_idx, norm_scores)}

    def _embedding_candidates(self, query: str, top_n: int) -> dict:
        query_vector = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].tolist()

        results = self.collection.query(query_embeddings=[query_vector], n_results=top_n)
        ids = results["ids"][0]
        distances = np.array(results["distances"][0])  # cosine distance، أصغر = أقرب
        similarities = 1 - distances
        norm_scores = self._min_max_normalize(similarities)
        return {cid: float(s) for cid, s in zip(ids, norm_scores)}

    def retrieve(self, query: str, top_k: int = TOP_K, alpha: float = HYBRID_ALPHA) -> list[dict]:
        """
        alpha: وزن الـ semantic. (1-alpha) وزن الـ BM25.
        بيرجّع list of dicts فيها chunk + hybrid_score، مرتبة تنازليًا.

        بعد كل نداء، self.last_timing بيتحدّث بـ
        {"embedding_time": ..., "retrieval_time": ...} بالثانية — مفيدة
        للواجهة (تبويب Details) من غير ما نغيّر شكل الـ return الأصلي.
        """
        t_start = time.perf_counter()
        candidate_pool_size = max(top_k * 4, 20)

        bm25_scores = self._bm25_candidates(query, candidate_pool_size)

        t_embed_start = time.perf_counter()
        embed_scores = self._embedding_candidates(query, candidate_pool_size)
        t_embed_end = time.perf_counter()

        all_ids = set(bm25_scores) | set(embed_scores)
        combined = []
        for cid in all_ids:
            lexical = bm25_scores.get(cid, 0.0)
            semantic = embed_scores.get(cid, 0.0)
            hybrid_score = alpha * semantic + (1 - alpha) * lexical
            chunk = dict(self.chunks_by_id[cid])
            chunk["hybrid_score"] = hybrid_score
            chunk["lexical_score"] = lexical
            chunk["semantic_score"] = semantic
            combined.append(chunk)

        combined.sort(key=lambda c: c["hybrid_score"], reverse=True)
        result = combined[:top_k]

        self.last_timing = {
            "embedding_time": t_embed_end - t_embed_start,
            "retrieval_time": time.perf_counter() - t_start,
        }
        return result


def main():
    query = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"
    retriever = HybridRetriever()
    results = retriever.retrieve(query)

    logger.info(f'أفضل {len(results)} chunk للسؤال: "{query}"\n')
    for i, r in enumerate(results, 1):
        print(f"--- #{i} | hybrid={r['hybrid_score']:.3f} "
              f"(lexical={r['lexical_score']:.3f}, semantic={r['semantic_score']:.3f}) ---")
        print(f"Title: {r['metadata'].get('title')}")
        print(f"Source: {r['metadata'].get('source')}")
        print(r["text"][:300].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
