# rag/hybrid.py
from __future__ import annotations

from typing import List, Dict, Optional, Iterable

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker


class HybridRetriever:
    """
    Single-step retrieval module.

    Responsibilities:
    1) BM25(q0 + rewrites)
    2) Dense rerank (q0 only)
    3) Cross-Encoder scoring (q0 only)

    Does NOT handle:
    - entity expansion
    - coverage selection
    - multi-step logic
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

    # =========================================================
    # Public API
    # =========================================================

    def search(
        self,
        *,
        query: str,
        rewrites: Iterable[str],
        bm25_top_n: int,
        dense_top_n: int,
        final_top_k: int,
        t_strong: Optional[float] = None,
    ) -> List[Dict]:
        """
        Base retrieval with optional confidence gate.

        If t_strong is provided and max CE >= t_strong:
            returns ONLY strong chunks
        """

        candidates = self._base_retrieval(
            query=query,
            rewrites=rewrites,
            bm25_top_n=bm25_top_n,
            dense_top_n=dense_top_n,
        )

        if not candidates:
            return []

        # -----------------------------
        # Confidence gate (CE arbiter)
        # -----------------------------
        if t_strong is not None and self.reranker is not None:
            max_ce = candidates[0].get("ce_score")

            if max_ce is not None and max_ce >= t_strong:
                strong = [
                    r for r in candidates
                    if r.get("ce_score", -1e9) >= t_strong
                ]
                if strong:
                    return strong[:final_top_k]

        return candidates[:final_top_k]

    # =========================================================
    # Internal
    # =========================================================

    def _base_retrieval(
        self,
        *,
        query: str,
        rewrites: Iterable[str],
        bm25_top_n: int,
        dense_top_n: int,
    ) -> List[Dict]:

        bm25_scores: dict[str, float] = {}
        bm25_hits: dict[str, int] = {}

        # --- BM25(q0) ---
        for r in self.bm25.search(query, top_k=bm25_top_n):
            cid = r["chunk_id"]
            bm25_scores[cid] = r["bm25_score"]
            bm25_hits[cid] = bm25_hits.get(cid, 0) + 1

        # --- BM25(rewrites) — recall only ---
        for q in rewrites:
            for r in self.bm25.search(q, top_k=bm25_top_n):
                cid = r["chunk_id"]
                bm25_hits[cid] = bm25_hits.get(cid, 0) + 1

        if not bm25_hits:
            return []

        candidate_ids = [
            cid for cid in bm25_hits
            if bm25_scores.get(cid, 0.0) > 0.0 or bm25_hits[cid] >= 2
        ]

        if not candidate_ids:
            return []

        # --- Dense rerank (q0 only) ---
        dense_res = self.dense.rerank_candidates(
            query,
            candidate_ids,
            top_k=dense_top_n,
        )

        for r in dense_res:
            cid = r["chunk_id"]
            r["bm25_score"] = bm25_scores.get(cid, 0.0)
            r["bm25_hits"] = bm25_hits.get(cid, 0)

        # --- Cross-Encoder (q0 only) ---
        if self.reranker is not None:
            dense_res = self.reranker.score(query, dense_res)
            dense_res.sort(key=lambda x: x["ce_score"], reverse=True)

        return dense_res