# rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

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
    bm25_top_n: int = 800
    dense_recall_top_n: int = 400

    # --- entity-only recall limits ---
    bm25_top_n_entity: int = 0
    dense_recall_top_n_entity: int = 0

    # --- fusion (RRF) ---
    use_fusion: bool = True
    rrf_k: int = 100
    w_bm25: float = 1.0
    w_dense: float = 1.0
    # optional: include ranks from rewrites or only q0
    fusion_use_rewrites: bool = False
    fusion_top_n: int = 700

    # --- entity bias ---
    entity_bias: float = 1.2

    # --- dense ranking ---
    dense_stage1_top_n: int = 500
    dense_stage2_top_n: int = 200

    # --- final ---
    final_top_k: int = 20

    # --- rewrites ---
    use_rewrites: bool = False
    n_rewrites: int = 2
    rewrite_min_cosine: float = 0.75

    # --- cross-encoder ---
    use_cross_encoder: bool = True
    ce_strong_threshold: Optional[float] = 3
    ce_top_n: int = 100

    # --- entity fallback ---
    use_entity_expansion: bool = True
    entity_bm25_top_n: int = 100
    entity_dense_recall_top_n: int = 30
    entity_top_n_per_entity: int = 7
    base_top_x: int = 7

    # --- coverage ---
    use_coverage: bool = False

    # coverage selector params (tunable)
    coverage_epsilon: float = 0.005
    coverage_max_chunks: int = 20
    coverage_alpha: float = 0.35

    # how wide pool is given to coverage (prevents "returns 1-2 chunks" collapse)
    coverage_pool_mult: int = 4        # pool = final_top_k * mult
    coverage_pool_min: int = 60        # minimum pool size

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

        # >>> LOG: храним последнюю диагностику
        self.last_debug: Dict = {}

    @staticmethod
    def _rrf(rrf_k: int, rank: int) -> float:
        return 1.0 / (rrf_k + rank)

    # --------------------------------------------------

    def retrieve(self, question: str) -> List[Dict]:
        q0 = question

        # ========== entities ==========
        entities: List[str] = (
            self.entity_extractor.extract(q0)
            if self.entity_extractor is not None
            else []
        )

        # >>> LOG: сущности
        self.last_debug = {
            "entities": entities,
            "gold_sources": [],
            "entity_hit": False,
        }

        # ========== rewrites ==========
        rewrites: List[str] = []
        if self.cfg.use_rewrites and self.rewriter is not None:
            rewrites = self.rewriter.rewrite(
                q0,
                n_rewrites=self.cfg.n_rewrites,
                min_cosine=self.cfg.rewrite_min_cosine,
            )

        Q = [q0] + rewrites
        Q_fusion = Q if (self.cfg.use_fusion and self.cfg.fusion_use_rewrites) else [q0]

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
            for i, r in enumerate(self.bm25.search(q, self.cfg.bm25_top_n), start=1):
                cid = r["chunk_id"]
                ensure(cid, r)
                cand[cid]["bm25_rank"] = i if cand[cid]["bm25_rank"] is None else min(cand[cid]["bm25_rank"], i)
                cand[cid]["source"].add("query")

        # --- Dense ---
        for q in Q_fusion:
            for i, r in enumerate(self.dense.search(q, self.cfg.dense_recall_top_n), start=1):
                cid = r["chunk_id"]
                ensure(cid, r)
                cand[cid]["dense_rank"] = i if cand[cid]["dense_rank"] is None else min(cand[cid]["dense_rank"], i)
                cand[cid]["source"].add("query")

        if not cand:
            return []

        # ========== fusion ==========
        if self.cfg.use_fusion:
            for c in cand.values():
                fs = 0.0
                if c["bm25_rank"] is not None:
                    fs += self._rrf(self.cfg.rrf_k, c["bm25_rank"])
                if c["dense_rank"] is not None:
                    fs += self._rrf(self.cfg.rrf_k, c["dense_rank"])
                c["fused_score"] = fs

            cand = dict(
                sorted(cand.items(), key=lambda x: x[1]["fused_score"], reverse=True)
                [: self.cfg.fusion_top_n]
            )

        candidate_ids = list(cand.keys())

        # ========== dense rerank ==========
        scored_q0 = self.dense.rerank_candidates(
            q0, candidate_ids, top_k=len(candidate_ids), return_embeddings=self.cfg.use_coverage
        )

        for r in scored_q0:
            c = cand[r["chunk_id"]]
            c["dense_q0"] = float(r["dense_score"])
            if "dense_emb" in r:
                c["dense_emb"] = r["dense_emb"]

        for q in Q:
            scored = self.dense.rerank_candidates(q, candidate_ids, top_k=len(candidate_ids))
            for r in scored:
                c = cand[r["chunk_id"]]
                c["dense_multi"] = max(c["dense_multi"] or float("-inf"), float(r["dense_score"]))

        # ========== selection ==========
        C1 = sorted(cand.values(), key=lambda x: x["dense_q0"], reverse=True)[: self.cfg.dense_stage1_top_n]
        C2 = sorted(C1, key=lambda x: x["dense_multi"], reverse=True)[: self.cfg.dense_stage2_top_n]

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
                cid = r["chunk_id"]
                if cid not in ent_pool:
                    ent_pool[cid] = {
                        "chunk_id": cid,
                        "title": r.get("title", ""),
                        "text": r.get("text", ""),
                        "dense_emb": None,
                    }
                # >>> LOG: entity hit
                ent_pool[cid].setdefault("source", set()).add("entity")

        # >>> LOG: был ли hit по entity вообще
        self.last_debug["entity_hit"] = bool(ent_pool)


        final_candidates = list(ent_pool.values())
        # ===== DENSE RERANK AFTER ENTITY EXPANSION =====
        entity_ids = [c["chunk_id"] for c in final_candidates]

        reranked = self.dense.rerank_candidates(
            q0,
            entity_ids,
            top_k=len(entity_ids),
            return_embeddings=self.cfg.use_coverage,
        )

        by_id = {c["chunk_id"]: c for c in final_candidates}

        for r in reranked:
            c = by_id[r["chunk_id"]]
            c["dense_q0"] = float(r["dense_score"])
            if "dense_emb" in r:
                c["dense_emb"] = r["dense_emb"]

        # теперь порядок по реальному score вопроса
        final_candidates = sorted(
            final_candidates,
            key=lambda x: x["dense_q0"],
            reverse=True,
        )

        # ========== OPTIONAL COVERAGE (FINAL STAGE) ==========
        if self.cfg.use_coverage and self.coverage_selector:

            # гарантируем, что coverage_selector инициализирован с параметрами из конфига
            coverage = CoverageSelector(
                epsilon=self.cfg.coverage_epsilon,
                max_chunks=self.cfg.coverage_max_chunks,
                alpha=self.cfg.coverage_alpha,
            )

            # формируем достаточно широкий пул для coverage
            pool_size = max(
                self.cfg.final_top_k * self.cfg.coverage_pool_mult,
                self.cfg.coverage_pool_min,
            )
            pool_for_coverage = [
                                    c for c in final_candidates
                                    if c.get("dense_emb") is not None
                                ][:pool_size]
            # считаем эмбеддинг запроса
            q_emb = self.dense.encode_query(q0)

            # coverage-aware отбор
            selected = coverage.select(
                query_emb=q_emb,
                candidates=pool_for_coverage,
                emb_key="dense_emb",
            )

            # 🔒 SAFETY: если coverage выбрал мало — добиваем top по dense_q0
            if len(selected) < self.cfg.final_top_k:
                used = {c["chunk_id"] for c in selected}
                for c in pool_for_coverage:
                    if len(selected) >= self.cfg.final_top_k:
                        break
                    if c["chunk_id"] not in used:
                        selected.append(c)
                        used.add(c["chunk_id"])

            final_candidates = selected

        # ========== FINAL ==========
        return self._strip(final_candidates[: self.cfg.final_top_k])

    # --------------------------------------------------
    @staticmethod
    def _strip(chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c.pop("dense_emb", None)
        return chunks
