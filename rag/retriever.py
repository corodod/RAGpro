# rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from rag.bm25 import BM25Retriever
from rag.coverage import CoverageSelector
from rag.dense import DenseRetriever
from rag.entities import EntityExtractor
from rag.reranker import CrossEncoderReranker
from rag.rewrite import QueryRewriter


# ================= CONFIG =================
@dataclass
class RetrieverConfig:
    # --- multi-query recall ---
    bm25_top_n: int = 300

    # dense recall per-query (FAISS search)
    dense_recall_top_n: int = 50

    # dense rerank pool size (after union + multi-q scoring)
    dense_top_n: int = 100

    # --- final ---
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
    entity_dense_recall_top_n: int = 30  # FAISS recall for entities (chunk-level)
    entity_top_n_per_entity: int = 2     # n_per_entity
    base_top_x: int = 2                  # x (fallback keeps top-x from main pass)

    # --- coverage ---
    use_coverage: bool = True


# ================= RETRIEVER =================
class Retriever:
    """
    Multi-query retrieval pipeline (q0 + rewrites), with:
      - BM25 recall per query
      - DenseSearch (FAISS) recall per query
      - Union by chunk_id with max aggregation
      - Multi-query dense ranking score = max DenseSim(q, c)
      - Cross-Encoder scoring ONLY on q0
      - Confidence gate (early exit)
      - Fallback: entity expansion + (optional) coverage selection

    Returns: list[Dict] chunks (chunk-level).
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

        # ================= Step 1: Rewrites =================
        rewrites: List[str] = []
        if self.cfg.use_rewrites and self.rewriter is not None:
            rewrites = self.rewriter.rewrite(
                q0,
                n_rewrites=self.cfg.n_rewrites,
                min_cosine=self.cfg.rewrite_min_cosine,
            )

        Q: List[str] = [q0] + list(rewrites)

        if self.debug:
            print("\n" + "=" * 80)
            print("[QUERY q0]", q0)
            print("[REWRITES]", rewrites or "(none)")
            print("[Q]", Q)

        # ================= Step 2: Multi-query recall =================
        # For each q in Q:
        #   B_q = BM25(q, bm25_top_n)
        #   D_q = DenseSearch(q, dense_recall_top_n)
        # Then union all candidates at chunk_id level.
        agg: Dict[str, Dict] = {}  # chunk_id -> aggregated record

        bm25_total = 0
        dense_total = 0

        for q in Q:
            # --- BM25(q) ---
            bm25_hits = self.bm25.search(q, top_k=self.cfg.bm25_top_n)
            bm25_total += len(bm25_hits)
            self._aggregate_hits(
                agg=agg,
                hits=bm25_hits,
                bm25_key="bm25_score",
                dense_key=None,
                hit_type="bm25",
            )

            # --- DenseSearch(q) ---
            dense_hits = self.dense.search(q, top_k=self.cfg.dense_recall_top_n)
            dense_total += len(dense_hits)
            self._aggregate_hits(
                agg=agg,
                hits=dense_hits,
                bm25_key=None,
                dense_key="dense_score",
                hit_type="dense",
            )

        if self.debug:
            print(f"[RECALL] bm25_hits_total={bm25_total} dense_hits_total={dense_total}")
            print(f"[UNION] unique_chunk_ids={len(agg)}")

        if not agg:
            return []

        # ================= Step 3+4: Multi-query dense ranking (max) =================
        # We want dense_rerank_score(c) = max_{q∈Q} DenseSim(q, c)
        # Since DenseRetriever can compute DenseSim(q, c) via rerank_candidates(query, candidate_ids),
        # we run it for each q over the SAME candidate set and aggregate max.
        candidate_ids = list(agg.keys())

        # We'll compute per-q dense scores (and optionally keep an embedding for coverage later).
        # To avoid storing huge embeddings, we only keep dense_emb for the query q0 run (enough for coverage).
        for qi, q in enumerate(Q):
            scored = self.dense.rerank_candidates(
                q,
                candidate_ids,
                top_k=len(candidate_ids),
                return_embeddings=(q == q0),  # only q0 needs dense_emb for coverage
            )
            # scored includes dense_score for all candidates (cosine), but only q0 run includes dense_emb.
            for r in scored:
                cid = r["chunk_id"]

                # max aggregation across queries
                prev = agg[cid].get("dense_max")
                cur = float(r.get("dense_score", 0.0))
                if prev is None or cur > prev:
                    agg[cid]["dense_max"] = cur
                    agg[cid]["dense_max_q"] = q  # debug / analysis

                # keep q0 embedding for coverage
                if q == q0 and "dense_emb" in r:
                    agg[cid]["dense_emb"] = r["dense_emb"]

        # Now we have dense_max for each candidate.
        ranked = list(agg.values())
        ranked.sort(key=lambda x: x.get("dense_max", -1e9), reverse=True)

        # truncate to dense_top_n for CE stage
        C2 = ranked[: self.cfg.dense_top_n]

        if self.debug:
            print(f"[DENSE-RANK] candidates_for_CE={len(C2)}")

        # ================= Step 5: Cross-Encoder scoring ONLY on q0 =================
        if self.cfg.use_cross_encoder and self.reranker is not None and C2:
            C2 = self.reranker.score(q0, C2)
            C2.sort(key=lambda x: x.get("ce_score", -1e9), reverse=True)

        # ================= Step 6: Confidence gate =================
        if (
            self.cfg.use_cross_encoder
            and self.cfg.ce_strong_threshold is not None
            and self.reranker is not None
            and C2
        ):
            strong = [
                c for c in C2
                if c.get("ce_score", -1e9) >= self.cfg.ce_strong_threshold
            ]
            if strong:
                if self.debug:
                    print(f"[GATE] strong_found={len(strong)} threshold={self.cfg.ce_strong_threshold}")
                return self._strip(strong[: self.cfg.final_top_k])

        # ================= Step 7: Fallback: entity expansion (no CE) =================
        # base_top = C3[:x] where C3 is CE-sorted list (or dense-sorted if CE is off)
        base_top = C2[: self.cfg.base_top_x]

        if not self.cfg.use_entity_expansion or self.entity_extractor is None:
            # no entities: optionally coverage over base_top (usually unnecessary)
            return self._strip(base_top[: self.cfg.final_top_k])

        entities = self.entity_extractor.extract(q0)

        if self.debug:
            print("[PATH] ENTITY_FALLBACK")
            print("[ENTITIES]", entities or "(none)")
            print(f"[BASE_TOP] x={self.cfg.base_top_x}")

        ent_pool: Dict[str, Dict] = {}
        for e in entities:
            # BM25(e)
            b_e = self.bm25.search(e, top_k=self.cfg.entity_bm25_top_n)

            # DenseSearch(e)
            d_e = self.dense.search(e, top_k=self.cfg.entity_dense_recall_top_n)

            # union for this entity, then rank by dense similarity to entity (max is trivial here)
            local: Dict[str, Dict] = {}
            self._aggregate_hits(local, b_e, bm25_key="bm25_score", dense_key=None, hit_type="bm25")
            self._aggregate_hits(local, d_e, bm25_key=None, dense_key="dense_score", hit_type="dense")

            # score dense similarity of entity to these candidates (for selecting top per entity)
            local_ids = list(local.keys())
            if local_ids:
                scored_e = self.dense.rerank_candidates(
                    e,
                    local_ids,
                    top_k=len(local_ids),
                    return_embeddings=False,  # embeddings not needed here
                )
                # map back dense_score
                for r in scored_e:
                    cid = r["chunk_id"]
                    local[cid]["dense_max"] = max(local[cid].get("dense_max", -1e9), float(r["dense_score"]))

                local_list = list(local.values())
                local_list.sort(key=lambda x: x.get("dense_max", -1e9), reverse=True)

                # take n per entity
                take = local_list[: self.cfg.entity_top_n_per_entity]
                for c in take:
                    cid = c["chunk_id"]
                    if cid not in ent_pool:
                        ent_pool[cid] = c

        Ent = list(ent_pool.values())

        # ================= Step 8: Final context assembly =================
        # Variant 2: coverage (stronger)
        pool = base_top + Ent

        if not pool:
            return []

        # If coverage enabled, we need dense_emb in candidates; ensure it exists:
        # - base_top has dense_emb (from q0 rerank loop above) if coverage is on
        # - entity candidates might not have dense_emb -> compute for pool if missing
        if self.cfg.use_coverage and self.coverage_selector is not None:
            # ensure embeddings for pool (q0-based)
            missing = [c["chunk_id"] for c in pool if "dense_emb" not in c]
            if missing:
                scored_missing = self.dense.rerank_candidates(
                    q0,
                    missing,
                    top_k=len(missing),
                    return_embeddings=True,
                )
                for r in scored_missing:
                    cid = r["chunk_id"]
                    # inject embedding back
                    for c in pool:
                        if c["chunk_id"] == cid:
                            c["dense_emb"] = r.get("dense_emb")

            q_emb = self.dense.encode_query(q0)
            final = self.coverage_selector.select(
                query_emb=q_emb,
                candidates=pool,
                emb_key="dense_emb",
            )
            return self._strip(final[: self.cfg.final_top_k])

        # fallback: no coverage
        return self._strip(pool[: self.cfg.final_top_k])

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _aggregate_hits(
        agg: Dict[str, Dict],
        hits: List[Dict],
        *,
        bm25_key: Optional[str],
        dense_key: Optional[str],
        hit_type: str,  # "bm25" | "dense" (for hits counter)
    ) -> None:
        """
        Aggregates a list of hits into `agg` by chunk_id using max aggregation:
          - bm25_max = max over all bm25_score occurrences
          - dense_max (initial recall score from FAISS) = max over all dense_score occurrences
          - bm25_hits / dense_hits counters
        """
        for r in hits:
            cid = r["chunk_id"]

            if cid not in agg:
                # Keep canonical fields from first occurrence
                agg[cid] = {
                    "chunk_id": cid,
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "bm25_max": None,
                    "dense_recall_max": None,  # score from DenseSearch() (not rerank)
                    "bm25_hits": 0,
                    "dense_hits": 0,
                }

            a = agg[cid]

            # hits counters
            if hit_type == "bm25":
                a["bm25_hits"] += 1
            elif hit_type == "dense":
                a["dense_hits"] += 1

            # bm25 max
            if bm25_key is not None and bm25_key in r:
                cur = float(r[bm25_key])
                prev = a.get("bm25_max")
                if prev is None or cur > prev:
                    a["bm25_max"] = cur

            # dense recall max (FAISS search score)
            if dense_key is not None and dense_key in r:
                cur = float(r[dense_key])
                prev = a.get("dense_recall_max")
                if prev is None or cur > prev:
                    a["dense_recall_max"] = cur

    @staticmethod
    def _strip(chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c.pop("dense_emb", None)
        return chunks
