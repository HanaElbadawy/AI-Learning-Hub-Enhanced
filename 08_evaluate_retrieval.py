"""
08_evaluate_retrieval.py
----------------------------
Phase 8 — Retrieval Evaluation (جديد، مبني على Lab 7 و Lab 8)

بيشتغل offline (مش وقت سؤال المستخدم الحقيقي). بياخد evaluation/eval_queries.json
(كل سطر فيه query + relevant_titles معروفة مسبقًا)، وبيحسب لكل واحد من:
BM25-only, Embeddings-only, Hybrid:

- Precision@K
- Recall@K
- Hit Rate@K
- MRR (Mean Reciprocal Rank)

الهدف: نعرف نقارن الطرق بالأرقام مش بالإحساس، ونعرف نحدد الـ alpha
الأفضل لـ HYBRID_ALPHA.

تشغيل:
    python 08_evaluate_retrieval.py
"""

import json
from importlib import import_module

import pandas as pd

from utils.config import EVALUATION_DIR, TOP_K
from utils.logging_utils import get_logger

logger = get_logger("08_evaluate_retrieval")

K = TOP_K


def _title_matches(chunk_title: str, relevant_titles: list[str]) -> bool:
    chunk_title = chunk_title.lower()
    return any(rt.lower() in chunk_title for rt in relevant_titles)


def evaluate_single_query(retrieved_chunks: list[dict], relevant_titles: list[str]) -> dict:
    hits = [
        _title_matches(c["metadata"].get("title", ""), relevant_titles) for c in retrieved_chunks
    ]

    n_relevant_retrieved = sum(hits)
    precision = n_relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0
    # بنفترض أقل حاجة "relevant" واحدة موجودة في الداتا لكل عنوان متوقع (تبسيط تعليمي)
    recall = n_relevant_retrieved / max(len(relevant_titles), 1)
    hit_rate = 1.0 if n_relevant_retrieved > 0 else 0.0

    reciprocal_rank = 0.0
    for rank, is_hit in enumerate(hits, start=1):
        if is_hit:
            reciprocal_rank = 1.0 / rank
            break

    return {
        f"precision@{K}": precision,
        f"recall@{K}": recall,
        f"hit_rate@{K}": hit_rate,
        "reciprocal_rank": reciprocal_rank,
    }


def main():
    eval_path = EVALUATION_DIR / "eval_queries.json"
    if not eval_path.exists():
        logger.warning(f"{eval_path} مش موجود.")
        return

    with open(eval_path, encoding="utf-8") as f:
        eval_queries = json.load(f)

    retrieve_module = import_module("06_retrieve_context")
    retriever = retrieve_module.HybridRetriever()

    rows = []
    for item in eval_queries:
        query = item["query"]
        relevant_titles = item["relevant_titles"]

        for method_name, alpha in [("bm25_only", 0.0), ("embeddings_only", 1.0), ("hybrid", None)]:
            kwargs = {"top_k": K}
            if alpha is not None:
                kwargs["alpha"] = alpha
            results = retriever.retrieve(query, **kwargs)
            metrics = evaluate_single_query(results, relevant_titles)
            rows.append({"query": query, "retriever": method_name, **metrics})

    df = pd.DataFrame(rows)
    summary = df.groupby("retriever")[
        [f"precision@{K}", f"recall@{K}", f"hit_rate@{K}", "reciprocal_rank"]
    ].mean().sort_values(by="reciprocal_rank", ascending=False)

    out_path = EVALUATION_DIR / "metrics_report.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"تفاصيل كل سؤال اتحفظت في {out_path}")
    print("\n=== ملخص مقارنة الـ retrievers (متوسط على كل الأسئلة) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
