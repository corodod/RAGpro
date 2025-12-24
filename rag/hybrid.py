# rag/hybrid.py
from typing import List, Dict, Optional

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

    def search(
        self,
        *,
        query: str,
        bm25_top_n: int,
        dense_top_n: int,
        final_top_k: int,
    ) -> List[Dict]:
        # 1️⃣ BM25 — candidate generation
        bm25_res = self.bm25.search(query, top_k=bm25_top_n)
        candidate_ids = [r["chunk_id"] for r in bm25_res]

        # 2️⃣ Dense — semantic filtering
        dense_res = self.dense.rerank_candidates(
            query,
            candidate_ids,
            top_k=dense_top_n,
        )

        # прокидываем bm25_score
        bm25_scores = {r["chunk_id"]: r["bm25_score"] for r in bm25_res}
        for r in dense_res:
            r["bm25_score"] = bm25_scores.get(r["chunk_id"], 0.0)

        # 3️⃣ Cross-encoder scoring (без отбора)
        if self.reranker is not None:
            dense_res = self.reranker.score(query, dense_res)

            dense_res = sorted(
                dense_res,
                key=lambda x: x["ce_score"],
                reverse=True,
            )

        # 4️⃣ Финальный top-k
        return dense_res[:final_top_k]
