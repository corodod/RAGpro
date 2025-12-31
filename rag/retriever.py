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
    bm25_top_n: int = 400
    dense_recall_top_n: int = 100

    # --- fusion (RRF) ---
    use_fusion: bool = True
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_dense: float = 1.0
    # optional: include ranks from rewrites or only q0
    fusion_use_rewrites: bool = True

    # how many candidates to keep after fusion before expensive dense rerank
    fusion_top_n: int = 800

    # --- dense ranking ---
    dense_stage1_top_n: int = 250   # q0-only precision stage
    dense_stage2_top_n: int = 200   # multi-q rescue stage

    # --- final ---
    final_top_k: int = 20

    # --- rewrites ---
    use_rewrites: bool = True
    n_rewrites: int = 2
    rewrite_min_cosine: float = 0.75

    # --- cross-encoder ---
    use_cross_encoder: bool = True
    ce_strong_threshold: Optional[float] = 3.1
    ce_top_n: int = 100  # (optional) cap CE scoring to top-N after dense_stage2

    # --- entity fallback ---
    use_entity_expansion: bool = True
    entity_bm25_top_n: int = 150
    entity_dense_recall_top_n: int = 30
    entity_top_n_per_entity: int = 3
    base_top_x: int = 3

    # --- coverage ---
    use_coverage: bool = True


# ================= RETRIEVER =================
class Retriever:
    """
    Hybrid retriever with:
      - Entity-first recall
      - Multi-query recall (q0 + rewrites + entities)
      - RRF fusion with entity bias
      - Anchored dense ranking
      - CE on q0
      - Entity fallback + coverage
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
        # rank is 1-based
        return 1.0 / (rrf_k + rank)

    def retrieve(self, question: str) -> List[Dict]:
        q0 = question

        # ================= 0) Entity extraction (ONCE) =================
        entities: List[str] = []
        if self.entity_extractor is not None:
            entities = self.entity_extractor.extract(q0)

        # ================= 1) Rewrites =================
        rewrites: List[str] = []
        if self.cfg.use_rewrites and self.rewriter is not None:
            rewrites = self.rewriter.rewrite(
                q0,
                n_rewrites=self.cfg.n_rewrites,
                min_cosine=self.cfg.rewrite_min_cosine,
            )

        Q = [q0] + rewrites

        # ================= NEW: entity-first fusion queries =================
        Q_entity = entities if entities else []

        if self.cfg.fusion_use_rewrites:
            Q_fusion = Q + Q_entity
        else:
            Q_fusion = [q0] + Q_entity

        # ================= 2) Multi-query recall + rank tracking =================
        cand: Dict[str, Dict] = {}

        def ensure(cid: str, hit: Dict):
            if cid not in cand:
                cand[cid] = {
                    "chunk_id": cid,
                    "title": hit.get("title", ""),
                    "text": hit.get("text", ""),
                    "source": set(),          # NEW
                    "fused_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                    "dense_q0": None,
                    "dense_multi": None,
                    "dense_emb": None,
                }

        # ---------- BM25 recall ----------
        for q in Q_fusion:
            is_entity = q in Q_entity
            bm25_hits = self.bm25.search(q, top_k=self.cfg.bm25_top_n)
            for i, r in enumerate(bm25_hits, start=1):
                cid = r["chunk_id"]
                ensure(cid, r)
                cand[cid]["source"].add("entity" if is_entity else "query")
                prev = cand[cid]["bm25_rank"]
                cand[cid]["bm25_rank"] = i if prev is None else min(prev, i)

        # ---------- Dense ANN recall ----------
        for q in Q_fusion:
            is_entity = q in Q_entity
            dense_hits = self.dense.search(q, top_k=self.cfg.dense_recall_top_n)
            for i, r in enumerate(dense_hits, start=1):
                cid = r["chunk_id"]
                ensure(cid, r)
                cand[cid]["source"].add("entity" if is_entity else "query")
                prev = cand[cid]["dense_rank"]
                cand[cid]["dense_rank"] = i if prev is None else min(prev, i)

        if not cand:
            return []

        # ================= 2.3) RRF fusion with entity bias =================
        if self.cfg.use_fusion:
            for c in cand.values():
                fs = 0.0
                if c["bm25_rank"] is not None:
                    fs += self.cfg.w_bm25 * self._rrf(self.cfg.rrf_k, c["bm25_rank"])
                if c["dense_rank"] is not None:
                    fs += self.cfg.w_dense * self._rrf(self.cfg.rrf_k, c["dense_rank"])

                # 🔥 Attribute-triggered bias
                if "entity" in c["source"]:
                    fs *= 1.2

                c["fused_score"] = fs

            fused_sorted = sorted(
                cand.values(),
                key=lambda x: x["fused_score"],
                reverse=True,
            )[: self.cfg.fusion_top_n]

            candidate_ids = [c["chunk_id"] for c in fused_sorted]
            cand = {c["chunk_id"]: c for c in fused_sorted}
        else:
            candidate_ids = list(cand.keys())

        # ================= 3) Dense scoring =================
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
                s = float(r["dense_score"])
                prev = c["dense_multi"]
                if prev is None or s > prev:
                    c["dense_multi"] = s

        # ================= 4) Two-stage dense rank =================
        C1 = sorted(cand.values(), key=lambda x: x["dense_q0"], reverse=True)
        C1 = C1[: self.cfg.dense_stage1_top_n]

        C2 = sorted(C1, key=lambda x: x["dense_multi"], reverse=True)
        C2 = C2[: self.cfg.dense_stage2_top_n]

        # ================= 5) Cross-Encoder =================
        if self.cfg.use_cross_encoder and self.reranker is not None:
            C2_for_ce = C2[: self.cfg.ce_top_n]
            C2_for_ce = self.reranker.score(q0, C2_for_ce)
            C2_for_ce.sort(key=lambda x: x.get("ce_score", -1e9), reverse=True)

            scored_ids = {c["chunk_id"] for c in C2_for_ce}
            rest = [c for c in C2 if c["chunk_id"] not in scored_ids]
            C2 = C2_for_ce + rest

        # ================= 6) Confidence gate =================
        strong = [
            c for c in C2
            if c.get("ce_score", -1e9) >= self.cfg.ce_strong_threshold
        ]
        if strong:
            return self._strip(strong[: self.cfg.final_top_k])

        # ================= 7) Entity fallback =================
        base_top = C2[: self.cfg.base_top_x]

        if not self.cfg.use_entity_expansion or self.entity_extractor is None:
            return self._strip(base_top[: self.cfg.final_top_k])

        ent_pool = {c["chunk_id"]: c for c in base_top}
        for e in entities:
            bm25 = self.bm25.search(e, top_k=self.cfg.entity_bm25_top_n)
            dense = self.dense.search(e, top_k=self.cfg.entity_dense_recall_top_n)
            for r in (bm25 + dense)[: self.cfg.entity_top_n_per_entity]:
                if r["chunk_id"] not in ent_pool:
                    ent_pool[r["chunk_id"]] = {
                        "chunk_id": r["chunk_id"],
                        "title": r.get("title", ""),
                        "text": r.get("text", ""),
                        "dense_emb": None,
                    }

        pool = list(ent_pool.values())

        # ================= 8) Coverage =================
        if self.cfg.use_coverage and self.coverage_selector is not None:
            q_emb = self.dense.encode_query(q0)
            pool_ids = [c["chunk_id"] for c in pool]
            pool = self.dense.rerank_candidates(
                q0,
                pool_ids,
                top_k=len(pool_ids),
                return_embeddings=True,
            )
            pool = self.coverage_selector.select(
                query_emb=q_emb,
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
