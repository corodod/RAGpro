# rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector


# ================= CONFIG =================
@dataclass
class RetrieverConfig:
    # --- recall ---
    bm25_top_n: int = 700
    dense_recall_top_n: int = 350

    # --- entity-only recall limits ---
    bm25_top_n_entity: int = 0
    dense_recall_top_n_entity: int = 0

    # --- fusion (RRF) ---
    use_fusion: bool = True
    rrf_k: int = 80
    w_bm25: float = 1.0
    w_dense: float = 1.0
    # optional: include ranks from rewrites or only q0
    fusion_use_rewrites: bool = False
    # how many candidates to keep after fusion before expensive dense rerank
    fusion_top_n: int = 600

    # --- entity bias ---
    entity_bias: float = 1.2

    # --- dense ranking ---
    dense_stage1_top_n: int = 300
    dense_stage2_top_n: int = 200

    # --- final ---
    final_top_k: int = 20

    # --- rewrites ---
    use_rewrites: bool = False
    n_rewrites: int = 2
    rewrite_min_cosine: float = 0.75

    # --- cross-encoder ---
    use_cross_encoder: bool = False
    ce_strong_threshold: Optional[float] = None
    ce_top_n: int = 100

    # --- entity fallback ---
    use_entity_expansion: bool = True
    entity_bm25_top_n: int = 50
    entity_dense_recall_top_n: int = 50
    entity_top_n_per_entity: int = 5
    base_top_x: int = 5

    # --- coverage ---
    use_coverage: bool = False


# ================= RETRIEVER =================
class Retriever:
    """
    Hybrid retriever with configurable components:
      - BM25 + Dense recall
      - Optional rewrites
      - Optional RRF fusion
      - Anchored dense ranking
      - Optional Cross-Encoder
      - Entity fallback
      - Optional coverage selection
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

    @staticmethod
    def _rrf(rrf_k: int, rank: int) -> float:
        return 1.0 / (rrf_k + rank)

    # --------------------------------------------------

    def retrieve(self, question: str) -> List[Dict]:
        q0 = question

        # ========== entities (used only in fallback / optional recall) ==========
        entities: List[str] = (
            self.entity_extractor.extract(q0)
            if self.entity_extractor is not None
            else []
        )

        # ========== rewrites ==========
        rewrites: List[str] = []
        if self.cfg.use_rewrites and self.rewriter is not None:
            rewrites = self.rewriter.rewrite(
                q0,
                n_rewrites=self.cfg.n_rewrites,
                min_cosine=self.cfg.rewrite_min_cosine,
            )

        Q = [q0] + rewrites

        # Which queries participate in recall / fusion
        if self.cfg.use_fusion and self.cfg.fusion_use_rewrites:
            Q_fusion = Q
        else:
            Q_fusion = [q0]

        # ========== recall ==========
        cand: Dict[str, Dict] = {}

        def ensure(cid: str, hit: Dict):
            if cid not in cand:
                cand[cid] = {
                    "chunk_id": cid,
                    "title": hit.get("title", ""),
                    "text": hit.get("text", ""),
                    "bm25_rank": None,
                    "dense_rank": None,
                    "dense_q0": None,
                    "dense_multi": None,
                    "dense_emb": None,
                    # used only if fusion is enabled
                    "source": set(),
                    "fused_score": 0.0,
                }

        # --- BM25 ---
        for q in Q_fusion:
            for i, r in enumerate(
                self.bm25.search(q, top_k=self.cfg.bm25_top_n),
                start=1,
            ):
                cid = r["chunk_id"]
                ensure(cid, r)
                cand[cid]["bm25_rank"] = (
                    i if cand[cid]["bm25_rank"] is None
                    else min(cand[cid]["bm25_rank"], i)
                )
                cand[cid]["source"].add("query")

        # --- Dense ANN ---
        for q in Q_fusion:
            for i, r in enumerate(
                self.dense.search(q, top_k=self.cfg.dense_recall_top_n),
                start=1,
            ):
                cid = r["chunk_id"]
                ensure(cid, r)
                cand[cid]["dense_rank"] = (
                    i if cand[cid]["dense_rank"] is None
                    else min(cand[cid]["dense_rank"], i)
                )
                cand[cid]["source"].add("query")

        if not cand:
            return []

        # ========== optional fusion ==========
        if self.cfg.use_fusion:
            for c in cand.values():
                fs = 0.0
                if c["bm25_rank"] is not None:
                    fs += self.cfg.w_bm25 * self._rrf(self.cfg.rrf_k, c["bm25_rank"])
                if c["dense_rank"] is not None:
                    fs += self.cfg.w_dense * self._rrf(self.cfg.rrf_k, c["dense_rank"])
                if "entity" in c["source"]:
                    fs *= self.cfg.entity_bias
                c["fused_score"] = fs

            cand = dict(
                sorted(
                    cand.items(),
                    key=lambda x: x[1]["fused_score"],
                    reverse=True,
                )[: self.cfg.fusion_top_n]
            )

        candidate_ids = list(cand.keys())

        # ========== dense rerank ==========
        scored_q0 = self.dense.rerank_candidates(
            q0,
            candidate_ids,
            top_k=len(candidate_ids),
            return_embeddings=self.cfg.use_coverage,
        )

        for r in scored_q0:
            c = cand[r["chunk_id"]]
            c["dense_q0"] = float(r["dense_score"])
            if "dense_emb" in r:
                c["dense_emb"] = r["dense_emb"]

        for q in Q:
            scored = self.dense.rerank_candidates(
                q,
                candidate_ids,
                top_k=len(candidate_ids),
                return_embeddings=False,
            )
            for r in scored:
                c = cand[r["chunk_id"]]
                c["dense_multi"] = max(
                    c["dense_multi"] or float("-inf"),
                    float(r["dense_score"]),
                )

        # ========== two-stage selection ==========
        C1 = sorted(cand.values(), key=lambda x: x["dense_q0"], reverse=True)
        C1 = C1[: self.cfg.dense_stage1_top_n]

        C2 = sorted(C1, key=lambda x: x["dense_multi"], reverse=True)
        C2 = C2[: self.cfg.dense_stage2_top_n]

        # ========== optional CE ==========
        if self.cfg.use_cross_encoder and self.reranker is not None:
            C2 = self.reranker.score(q0, C2[: self.cfg.ce_top_n])
            C2.sort(key=lambda x: x.get("ce_score", -1e9), reverse=True)

        if self.cfg.use_cross_encoder and self.cfg.ce_strong_threshold is not None:
            strong = [
                c for c in C2
                if c.get("ce_score", -1e9) >= self.cfg.ce_strong_threshold
            ]
            if strong:
                return self._strip(strong[: self.cfg.final_top_k])

        # ========== entity fallback ==========
        base = C2[: self.cfg.base_top_x]
        if not self.cfg.use_entity_expansion or not entities:
            return self._strip(base[: self.cfg.final_top_k])

        ent_pool = {c["chunk_id"]: c for c in base}
        for e in entities:
            for r in (
                self.bm25.search(e, self.cfg.entity_bm25_top_n)
                + self.dense.search(e, self.cfg.entity_dense_recall_top_n)
            )[: self.cfg.entity_top_n_per_entity]:
                if r["chunk_id"] not in ent_pool:
                    ent_pool[r["chunk_id"]] = {
                        "chunk_id": r["chunk_id"],
                        "title": r.get("title", ""),
                        "text": r.get("text", ""),
                        "dense_emb": None,
                    }

        pool = list(ent_pool.values())

        # ========== coverage ==========
        if self.cfg.use_coverage and self.coverage_selector is not None:
            pool_ids = [c["chunk_id"] for c in pool]
            pool = self.dense.rerank_candidates(
                q0,
                pool_ids,
                top_k=len(pool_ids),
                return_embeddings=True,
            )
            pool = self.coverage_selector.select(
                query_emb=self.dense.encode_query(q0),
                candidates=pool,
                emb_key="dense_emb",
            )

        return self._strip(pool[: self.cfg.final_top_k])

    # --------------------------------------------------
    @staticmethod
    def _strip(chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c.pop("dense_emb", None)
        return chunks
