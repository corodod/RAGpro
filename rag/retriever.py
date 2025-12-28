# rag/retriever.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector


# ================= CONFIG =================
@dataclass
class RetrieverConfig:
    # --- base retrieval ---
    bm25_top_n: int = 300
    dense_top_n: int = 100
    final_top_k: int = 20

    # --- rewrites ---
    use_rewrites: bool = True
    n_rewrites: int = 2
    rewrite_min_cosine: float = 0.75

    # --- cross-encoder ---
    use_cross_encoder: bool = True
    ce_strong_threshold: Optional[float] = 11.2

    # --- entity expansion ---
    use_entity_expansion: bool = True
    entity_bm25_top_n: int = 150
    entity_dense_top_n: int = 50

    # --- coverage ---
    use_coverage: bool = True


# ================= RETRIEVER =================
class Retriever:
    """
    Unified retrieval entry point.
    ALL retrieval logic lives here.
    """

    def __init__(
        self,
        *,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        rewriter: Optional[QueryRewriter] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        coverage_selector: Optional[CoverageSelector] = None,
        config: RetrieverConfig,
        debug: bool = False,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rewriter = rewriter
        self.entity_extractor = entity_extractor
        self.coverage_selector = coverage_selector
        self.cfg = config
        self.debug = debug

    # --------------------------------------------------

    def retrieve(self, question: str) -> List[Dict]:
        q0 = question

        # ================= Rewrites =================
        rewrites: Iterable[str] = []
        if self.cfg.use_rewrites and self.rewriter is not None:
            rewrites = self.rewriter.rewrite(
                q0,
                n_rewrites=self.cfg.n_rewrites,
                min_cosine=self.cfg.rewrite_min_cosine,
            )

        if self.debug:
            print("\n" + "=" * 80)
            print("[QUERY]", q0)
            print("[REWRITES]", rewrites or "(none)")

        # ================= Base retrieval =================
        candidates = self._base_retrieval(
            query=q0,
            rewrites=rewrites,
            bm25_top_n=self.cfg.bm25_top_n,
            dense_top_n=self.cfg.dense_top_n,
        )

        if not candidates:
            return []

        # ================= CE confidence gate =================
        if (
            self.cfg.use_cross_encoder
            and self.cfg.ce_strong_threshold is not None
            and self.reranker is not None
        ):
            max_ce = candidates[0].get("ce_score")
            if max_ce is not None and max_ce >= self.cfg.ce_strong_threshold:
                if self.debug:
                    print("[PATH] STRONG_CE_EXIT", max_ce)
                strong = [
                    r for r in candidates
                    if r.get("ce_score", -1e9) >= self.cfg.ce_strong_threshold
                ]
                return self._strip(strong[: self.cfg.final_top_k])

        # ================= Entity expansion =================
        if self.cfg.use_entity_expansion and self.entity_extractor is not None:
            entities = self.entity_extractor.extract(q0)

            if self.debug:
                print("[PATH] ENTITY_EXPANSION")
                print("[ENTITIES]", entities or "(none)")

            expanded = []
            for e in entities:
                expanded.extend(
                    self._base_retrieval(
                        query=e,
                        rewrites=[],
                        bm25_top_n=self.cfg.entity_bm25_top_n,
                        dense_top_n=self.cfg.entity_dense_top_n,
                    )
                )

            by_id = {r["chunk_id"]: r for r in candidates + expanded}
            candidates = list(by_id.values())

        # ================= Coverage =================
        if self.cfg.use_coverage and self.coverage_selector is not None:
            q_emb = self.dense.encode_query(q0)
            candidates = self.dense.rerank_candidates(
                q0,
                [c["chunk_id"] for c in candidates],
                top_k=len(candidates),
            )
            candidates = self.coverage_selector.select(
                query_emb=q_emb,
                candidates=candidates,
                emb_key="dense_emb",
            )

        return self._strip(candidates[: self.cfg.final_top_k])

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

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

        # --- BM25(rewrites) ---
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

        # --- Dense rerank ---
        dense_res = self.dense.rerank_candidates(
            query,
            candidate_ids,
            top_k=dense_top_n,
        )

        for r in dense_res:
            cid = r["chunk_id"]
            r["bm25_score"] = bm25_scores.get(cid, 0.0)
            r["bm25_hits"] = bm25_hits.get(cid, 0)

        # --- Cross-Encoder ---
        if self.reranker is not None:
            dense_res = self.reranker.score(query, dense_res)
            dense_res.sort(key=lambda x: x["ce_score"], reverse=True)

        return dense_res

    @staticmethod
    def _strip(chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c.pop("dense_emb", None)
        return chunks