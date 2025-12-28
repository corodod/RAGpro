# rag/retriever.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

from rag.hybrid import HybridRetriever
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector
from rag.dense import DenseRetriever


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
    This is the ONLY place where retrieval policy lives.
    """

    def __init__(
        self,
        *,
        hybrid: HybridRetriever,
        dense: DenseRetriever,
        rewriter: Optional[QueryRewriter] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        coverage_selector: Optional[CoverageSelector] = None,
        config: RetrieverConfig,
        debug: bool = False,
    ):
        self.hybrid = hybrid
        self.dense = dense
        self.rewriter = rewriter
        self.entity_extractor = entity_extractor
        self.coverage_selector = coverage_selector
        self.cfg = config
        self.debug = debug

    # --------------------------------------------------

    def retrieve(self, question: str) -> List[Dict]:
        q0 = question

        # ================= Rewrites =================
        rewrites = []
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
        base = self.hybrid.search(
            query=q0,
            rewrites=rewrites,
            bm25_top_n=self.cfg.bm25_top_n,
            dense_top_n=self.cfg.dense_top_n,
            final_top_k=self.cfg.final_top_k,
            t_strong=self.cfg.ce_strong_threshold
            if self.cfg.use_cross_encoder else None,
        )

        if not base:
            return []

        max_ce = base[0].get("ce_score")

        if (
            self.cfg.use_cross_encoder
            and self.cfg.ce_strong_threshold is not None
            and max_ce is not None
            and max_ce >= self.cfg.ce_strong_threshold
        ):
            if self.debug:
                print("[PATH] STRONG_CE_EXIT", max_ce)
            return self._strip(base)

        # ================= Entity expansion =================
        candidates = base

        if self.cfg.use_entity_expansion and self.entity_extractor is not None:
            entities = self.entity_extractor.extract(q0)

            if self.debug:
                print("[PATH] ENTITY_EXPANSION")
                print("[ENTITIES]", entities or "(none)")

            expanded = []
            for e in entities:
                expanded.extend(
                    self.hybrid.search(
                        query=e,
                        rewrites=[],
                        bm25_top_n=self.cfg.entity_bm25_top_n,
                        dense_top_n=self.cfg.entity_dense_top_n,
                        final_top_k=self.cfg.final_top_k,
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

        return self._strip(candidates)

    # --------------------------------------------------

    @staticmethod
    def _strip(chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c.pop("dense_emb", None)
        return chunks