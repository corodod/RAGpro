# rag/hybrid.py
from typing import List, Dict, Optional, Iterable

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
        rewrites: Iterable[str],
        bm25_top_n: int,
        dense_top_n: int,
        final_top_k: int,
    ) -> List[Dict]:

        # =====================================================
        # 1️⃣ BM25 aggregation (q0 ≠ rewrites)
        # =====================================================
        bm25_scores = {}
        bm25_hits = {}

        # --- q0: основной скор ---
        for r in self.bm25.search(query, top_k=bm25_top_n):
            cid = r["chunk_id"]
            bm25_scores[cid] = r["bm25_score"]
            bm25_hits[cid] = bm25_hits.get(cid, 0) + 1

        # --- rewrites: только факт попадания ---
        for q in rewrites:
            for r in self.bm25.search(q, top_k=bm25_top_n):
                cid = r["chunk_id"]
                bm25_hits[cid] = bm25_hits.get(cid, 0) + 1

        if not bm25_hits:
            return []

        # =====================================================
        # 2️⃣ Candidate filtering (anti-noise)
        # =====================================================
        candidate_ids = [
            cid for cid in bm25_hits
            if bm25_scores.get(cid, 0.0) > 0.0 or bm25_hits[cid] >= 2
        ]

        if not candidate_ids:
            return []

        # =====================================================
        # 3️⃣ Dense rerank (ONLY q0)
        # =====================================================
        dense_res = self.dense.rerank_candidates(
            query,
            candidate_ids,
            top_k=dense_top_n,
        )

        for r in dense_res:
            r["bm25_score"] = bm25_scores.get(r["chunk_id"], 0.0)
            r["bm25_hits"] = bm25_hits.get(r["chunk_id"], 0)

        # =====================================================
        # 4️⃣ Cross-encoder (ONLY q0)
        # =====================================================
        if self.reranker is not None:
            dense_res = self.reranker.score(query, dense_res)
            dense_res = sorted(
                dense_res,
                key=lambda x: x["ce_score"],
                reverse=True,
            )

        # =====================================================
        # 5️⃣ Final top-k
        # =====================================================
        return dense_res[:final_top_k]