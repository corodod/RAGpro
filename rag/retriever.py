# rag/retriever.py
from __future__ import annotations
from pathlib import Path
# =========================
# HYPERPARAMETERS
# =========================

# --- recall ---
BM25_TOP_N = 800
DENSE_RECALL_TOP_N = 400

# --- entity-only recall limits (currently unused in code, but kept for config) ---
BM25_TOP_N_ENTITY = 0
DENSE_RECALL_TOP_N_ENTITY = 0

# --- fusion (RRF) ---
USE_FUSION = True
RRF_K = 100
W_BM25 = 1.0
W_DENSE = 1.0
FUSION_USE_REWRITES = False
FUSION_TOP_N = 700

# --- entity bias (currently not used in code path, but kept) ---
ENTITY_BIAS = 1.2

# --- dense ranking stages ---
DENSE_STAGE1_TOP_N = 500
DENSE_STAGE2_TOP_N = 200
# --- dense: retriever hyperparams (MOVED HERE from rag/dense.py) ---
DENSE_MODEL_NAME = "intfloat/multilingual-e5-large"
DENSE_EMBEDDING_DIM = 1024
DENSE_QUERY_PREFIX = "query: "
DENSE_PASSAGE_PREFIX = "passage: "
DENSE_SEARCH_TOP_K = 10
DENSE_RERANK_TOP_K = 10
DENSE_RERANK_RETURN_EMBEDDINGS = True
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"

# --- final ---
FINAL_TOP_K = 20

# --- rewrites (on/off + quality filter) ---
USE_REWRITES = False
N_REWRITES = 2
REWRITE_MIN_COSINE = 0.75
# --- rewrites: models/devices/generation/parsing ---
REWRITE_MIN_LINE_LEN = 10
REWRITE_LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
REWRITE_EMBEDDER_MODEL_NAME = "intfloat/multilingual-e5-small"
REWRITE_LLM_DEVICE = "cpu"          # "cpu" | "cuda"
REWRITE_MAX_NEW_TOKENS = 96
REWRITE_DO_SAMPLE = False
REWRITE_TEMPERATURE = 0.0

# --- cross-encoder ---
USE_CROSS_ENCODER = True
CE_STRONG_THRESHOLD = None  # float | None
CE_TOP_N = 100
# --- cross-encoder: reranker hyperparams ---
CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CROSS_ENCODER_DEVICE = "cpu"          # "cpu" | "cuda"
CROSS_ENCODER_BATCH_SIZE = 32
CROSS_ENCODER_USE_FP16 = True         # effective only on CUDA

# --- entity fallback / expansion ---
USE_ENTITY_EXPANSION = True
ENTITY_BM25_TOP_N = 100
ENTITY_DENSE_RECALL_TOP_N = 30
ENTITY_TOP_N_PER_ENTITY = 7
BASE_TOP_X = 7
# --- entities: extractor hyperparams (MOVED HERE from rag/entities.py) ---
ENT_SPACY_MODEL = "ru_core_news_lg"
ENT_ALLOWED_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "EVENT",
    "WORK_OF_ART",
}
ENT_MIN_LEN = 3
ENT_MAX_TOKENS = 6

# --- coverage ---
USE_COVERAGE = False
COVERAGE_EPSILON = 0.005
COVERAGE_MAX_CHUNKS = 20
COVERAGE_ALPHA = 0.35
COVERAGE_POOL_MULT = 4
COVERAGE_POOL_MIN = 60

# --- HyDE ---
USE_HYDE = False
HYDE_MAX_TOKENS = 120
HYDE_DENSE_TOP_N = 200
HYDE_ONLY_IF_NO_ENTITIES = True
HYDE_MAX_QUERY_LEN = 4
# --- HyDE: generator hyperparams (MOVED HERE from rag/hyde.py) ---
HYDE_LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HYDE_LLM_DEVICE = "cpu"
HYDE_DO_SAMPLE = False
HYDE_TEMPERATURE = 0.0
HYDE_TRUST_REMOTE_CODE = True
# max tokens for generation (use your existing one as source of truth)
HYDE_MAX_NEW_TOKENS = HYDE_MAX_TOKENS

# --- Query2Doc ---
USE_QUERY2DOC = True
USE_QUERY2DOC_BM25 = True
USE_QUERY2DOC_DENSE = False
QUERY2DOC_MAX_TOKENS = 128
QUERY2DOC_BM25_TOP_N = 300
QUERY2DOC_DENSE_TOP_N = 200
QUERY2DOC_ONLY_IF_NO_ENTITIES = True
QUERY2DOC_MAX_QUERY_LEN = 6
# --- Query2Doc: generator hyperparams (MOVED HERE from rag/query2doc.py) ---
QUERY2DOC_LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QUERY2DOC_LLM_DEVICE = "cpu"
QUERY2DOC_TEMPERATURE = 0.0
QUERY2DOC_TRUST_REMOTE_CODE = True

# max tokens for generation (use your existing one as source of truth)
QUERY2DOC_MAX_NEW_TOKENS = QUERY2DOC_MAX_TOKENS

# =========================
# IMPORTS
# =========================

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector
from rag.hyde import HyDEGenerator
from rag.query2doc import Query2DocGenerator


# ================= CONFIG =================
@dataclass
class RetrieverConfig:
    # --- recall ---
    bm25_top_n: int = BM25_TOP_N
    dense_recall_top_n: int = DENSE_RECALL_TOP_N

    # --- entity-only recall limits ---
    bm25_top_n_entity: int = BM25_TOP_N_ENTITY
    dense_recall_top_n_entity: int = DENSE_RECALL_TOP_N_ENTITY

    # --- fusion (RRF) ---
    use_fusion: bool = USE_FUSION
    rrf_k: int = RRF_K
    w_bm25: float = W_BM25
    w_dense: float = W_DENSE
    fusion_use_rewrites: bool = FUSION_USE_REWRITES
    fusion_top_n: int = FUSION_TOP_N

    # --- entity bias ---
    entity_bias: float = ENTITY_BIAS

    # --- dense ranking ---
    dense_stage1_top_n: int = DENSE_STAGE1_TOP_N
    dense_stage2_top_n: int = DENSE_STAGE2_TOP_N
    # --- dense: model & behavior ---
    dense_model_name: str = DENSE_MODEL_NAME
    dense_embedding_dim: int = DENSE_EMBEDDING_DIM
    dense_query_prefix: str = DENSE_QUERY_PREFIX
    dense_passage_prefix: str = DENSE_PASSAGE_PREFIX
    dense_search_top_k: int = DENSE_SEARCH_TOP_K
    dense_rerank_top_k: int = DENSE_RERANK_TOP_K
    dense_rerank_return_embeddings: bool = DENSE_RERANK_RETURN_EMBEDDINGS


    # --- final ---
    final_top_k: int = FINAL_TOP_K

    # --- rewrites ---
    use_rewrites: bool = USE_REWRITES
    n_rewrites: int = N_REWRITES
    rewrite_min_cosine: float = REWRITE_MIN_COSINE

    # --- rewrites: full control here ---
    rewrite_min_line_len: int = REWRITE_MIN_LINE_LEN
    rewrite_llm_model_name: str = REWRITE_LLM_MODEL_NAME
    rewrite_embedder_model_name: str = REWRITE_EMBEDDER_MODEL_NAME
    rewrite_llm_device: str = REWRITE_LLM_DEVICE
    rewrite_max_new_tokens: int = REWRITE_MAX_NEW_TOKENS
    rewrite_do_sample: bool = REWRITE_DO_SAMPLE
    rewrite_temperature: float = REWRITE_TEMPERATURE

    # --- cross-encoder ---
    use_cross_encoder: bool = USE_CROSS_ENCODER
    ce_strong_threshold: Optional[float] = CE_STRONG_THRESHOLD
    ce_top_n: int = CE_TOP_N
    # --- cross-encoder: reranker hyperparams ---
    cross_encoder_model_name: str = CROSS_ENCODER_MODEL_NAME
    cross_encoder_device: str = CROSS_ENCODER_DEVICE
    cross_encoder_batch_size: int = CROSS_ENCODER_BATCH_SIZE
    cross_encoder_use_fp16: bool = CROSS_ENCODER_USE_FP16

    # --- entity fallback ---
    use_entity_expansion: bool = USE_ENTITY_EXPANSION
    entity_bm25_top_n: int = ENTITY_BM25_TOP_N
    entity_dense_recall_top_n: int = ENTITY_DENSE_RECALL_TOP_N
    entity_top_n_per_entity: int = ENTITY_TOP_N_PER_ENTITY
    base_top_x: int = BASE_TOP_X
    # --- entities ---
    ent_spacy_model: str = ENT_SPACY_MODEL
    ent_allowed_labels: Set[str] = None  # set default in __post_init__
    ent_min_len: int = ENT_MIN_LEN
    ent_max_tokens: int = ENT_MAX_TOKENS

    # --- coverage ---
    use_coverage: bool = USE_COVERAGE
    coverage_epsilon: float = COVERAGE_EPSILON
    coverage_max_chunks: int = COVERAGE_MAX_CHUNKS
    coverage_alpha: float = COVERAGE_ALPHA
    coverage_pool_mult: int = COVERAGE_POOL_MULT
    coverage_pool_min: int = COVERAGE_POOL_MIN

    # --- HyDE ---
    use_hyde: bool = USE_HYDE
    hyde_max_tokens: int = HYDE_MAX_TOKENS
    hyde_dense_top_n: int = HYDE_DENSE_TOP_N
    hyde_only_if_no_entities: bool = HYDE_ONLY_IF_NO_ENTITIES
    hyde_max_query_len: int = HYDE_MAX_QUERY_LEN
    # --- HyDE: generator hyperparams ---
    hyde_llm_model_name: str = HYDE_LLM_MODEL_NAME
    hyde_llm_device: str = HYDE_LLM_DEVICE
    hyde_max_new_tokens: int = HYDE_MAX_NEW_TOKENS
    hyde_do_sample: bool = HYDE_DO_SAMPLE
    hyde_temperature: float = HYDE_TEMPERATURE
    hyde_trust_remote_code: bool = HYDE_TRUST_REMOTE_CODE

    # --- Query2Doc ---
    use_query2doc: bool = USE_QUERY2DOC
    use_query2doc_bm25: bool = USE_QUERY2DOC_BM25
    use_query2doc_dense: bool = USE_QUERY2DOC_DENSE
    query2doc_max_tokens: int = QUERY2DOC_MAX_TOKENS
    query2doc_bm25_top_n: int = QUERY2DOC_BM25_TOP_N
    query2doc_dense_top_n: int = QUERY2DOC_DENSE_TOP_N
    query2doc_only_if_no_entities: bool = QUERY2DOC_ONLY_IF_NO_ENTITIES
    query2doc_max_query_len: int = QUERY2DOC_MAX_QUERY_LEN
    # --- Query2Doc: generator hyperparams ---
    query2doc_llm_model_name: str = QUERY2DOC_LLM_MODEL_NAME
    query2doc_llm_device: str = QUERY2DOC_LLM_DEVICE
    query2doc_max_new_tokens: int = QUERY2DOC_MAX_NEW_TOKENS
    query2doc_temperature: float = QUERY2DOC_TEMPERATURE
    query2doc_trust_remote_code: bool = QUERY2DOC_TRUST_REMOTE_CODE

    def __post_init__(self):
        if self.ent_allowed_labels is None:
            self.ent_allowed_labels = set(ENT_ALLOWED_LABELS)

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
        self.coverage_selector = coverage_selector
        self.cfg = config
        self.debug = debug

        self.dense = dense or DenseRetriever(
            chunks_path=CHUNKS_PATH,
            index_path=INDEX_DIR / "faiss.index",
            meta_path=INDEX_DIR / "faiss_meta.json",
            model_name=self.cfg.dense_model_name,
            embedding_dim=self.cfg.dense_embedding_dim,
            query_prefix=self.cfg.dense_query_prefix,
            passage_prefix=self.cfg.dense_passage_prefix,
            default_search_top_k=self.cfg.dense_search_top_k,
            default_rerank_top_k=self.cfg.dense_rerank_top_k,
            default_return_embeddings=self.cfg.dense_rerank_return_embeddings,
        )
        # если reranker не передали — создаём тут из cfg
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.cfg.cross_encoder_model_name,
            device=self.cfg.cross_encoder_device,
            batch_size=self.cfg.cross_encoder_batch_size,
            use_fp16=self.cfg.cross_encoder_use_fp16,
        )
        # если rewriter не передали — создаём тут из cfg (и все гиперы живут в retriever)
        self.rewriter = rewriter or QueryRewriter(
            llm_model_name=self.cfg.rewrite_llm_model_name,
            embedder_model_name=self.cfg.rewrite_embedder_model_name,
            llm_device=self.cfg.rewrite_llm_device,
            max_new_tokens=self.cfg.rewrite_max_new_tokens,
            do_sample=self.cfg.rewrite_do_sample,
            temperature=self.cfg.rewrite_temperature,
            min_line_len=self.cfg.rewrite_min_line_len,
            n_rewrites=self.cfg.n_rewrites,
            min_cosine=self.cfg.rewrite_min_cosine,
        )

        # >>> LOG: храним последнюю диагностику
        self.last_debug: Dict = {}

        self.hyde = (
            HyDEGenerator(
                llm_model_name=self.cfg.hyde_llm_model_name,
                llm_device=self.cfg.hyde_llm_device,
                max_new_tokens=self.cfg.hyde_max_new_tokens,
                do_sample=self.cfg.hyde_do_sample,
                temperature=self.cfg.hyde_temperature,
                trust_remote_code=self.cfg.hyde_trust_remote_code,
            )
            if self.cfg.use_hyde
            else None
        )
        self.query2doc = (
            Query2DocGenerator(
                llm_model_name=self.cfg.query2doc_llm_model_name,
                llm_device=self.cfg.query2doc_llm_device,
                max_new_tokens=self.cfg.query2doc_max_new_tokens,
                temperature=self.cfg.query2doc_temperature,
                trust_remote_code=self.cfg.query2doc_trust_remote_code,
            )
            if self.cfg.use_query2doc
            else None
        )
        # если entity_extractor не передали — создаём тут из cfg
        self.entity_extractor = entity_extractor or EntityExtractor(
            model=self.cfg.ent_spacy_model,
            allowed_labels=self.cfg.ent_allowed_labels,
            min_len=self.cfg.ent_min_len,
            max_tokens=self.cfg.ent_max_tokens,
        )
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
            rewrites = self.rewriter.rewrite(q0)

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

        # ========== Query2Doc recall ==========
        if self.cfg.use_query2doc and self.query2doc is not None:
            use_q2d = True

            if self.cfg.query2doc_only_if_no_entities and entities:
                use_q2d = False

            if len(q0.split()) > self.cfg.query2doc_max_query_len:
                use_q2d = False

            if use_q2d:
                q2d_text = self.query2doc.generate(q0)

                if self.cfg.use_query2doc_bm25:
                    bm25_query = f"{q0}\n{q2d_text}"
                    for i, r in enumerate(
                        self.bm25.search(bm25_query, self.cfg.query2doc_bm25_top_n),
                        start=1,
                    ):
                        cid = r["chunk_id"]
                        ensure(cid, r)
                        cand[cid]["bm25_rank"] = (
                            i if cand[cid]["bm25_rank"] is None
                            else min(cand[cid]["bm25_rank"], i)
                        )
                        cand[cid]["source"].add("query2doc_bm25")

                if self.cfg.use_query2doc_dense:
                    dense_query = q2d_text
                    for i, r in enumerate(
                        self.dense.search(dense_query, self.cfg.query2doc_dense_top_n),
                        start=1,
                    ):
                        cid = r["chunk_id"]
                        ensure(cid, r)
                        cand[cid]["dense_rank"] = (
                            i if cand[cid]["dense_rank"] is None
                            else min(cand[cid]["dense_rank"], i)
                        )
                        cand[cid]["source"].add("query2doc_dense")

        # ========== HyDE dense recall (fallback) ==========
        if self.cfg.use_hyde and self.hyde is not None:
            use_hyde = True

            if self.cfg.hyde_only_if_no_entities and entities:
                use_hyde = False

            if len(q0.split()) > self.cfg.hyde_max_query_len:
                use_hyde = False

            if use_hyde:
                hyde_text = self.hyde.generate(q0)
                hyde_vec = self.dense.encode_passage(hyde_text).reshape(1, -1)

                scores, idxs = self.dense.index.search(
                    hyde_vec,
                    self.cfg.hyde_dense_top_n,
                )

                for i, idx in enumerate(idxs[0], start=1):
                    if idx < 0:
                        continue

                    cid = self.dense.chunk_ids[idx]
                    ensure(cid, {
                        "chunk_id": cid,
                        "title": self.dense.titles[idx],
                        "text": self.dense.texts[idx],
                    })

                    cand[cid]["dense_rank"] = (
                        i if cand[cid]["dense_rank"] is None
                        else min(cand[cid]["dense_rank"], i)
                    )
                    cand[cid]["source"].add("hyde")

        # ========== fusion ==========
        if self.cfg.use_fusion:
            for c in cand.values():
                fs = 0.0
                if c["bm25_rank"] is not None:
                    fs += self.cfg.w_bm25 * self._rrf(self.cfg.rrf_k, c["bm25_rank"])
                if c["dense_rank"] is not None:
                    fs += self.cfg.w_dense * self._rrf(self.cfg.rrf_k, c["dense_rank"])
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

        # ========== OPTIONAL CROSS-ENCODER RERANK ==========
        if (
            self.cfg.use_cross_encoder
            and self.reranker is not None
            and final_candidates
        ):
            # берем только top-N, чтобы не убить производительность
            ce_pool = final_candidates[: self.cfg.ce_top_n]

            # считаем ce_score
            scored = self.reranker.score(q0, ce_pool)

            # опциональный hard-filter (если хочешь)
            if self.cfg.ce_strong_threshold is not None:
                scored = [
                    c for c in scored
                    if c.get("ce_score", float("-inf")) >= self.cfg.ce_strong_threshold
                ]

            # сортируем по cross-encoder
            scored = sorted(scored, key=lambda x: x["ce_score"], reverse=True)

            # добавляем хвост без изменения порядка
            used = {c["chunk_id"] for c in scored}
            tail = [c for c in final_candidates if c["chunk_id"] not in used]

            final_candidates = scored + tail

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

        return self._strip(final_candidates[: self.cfg.final_top_k])

    # --------------------------------------------------
    @staticmethod
    def _strip(chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c.pop("dense_emb", None)
        return chunks
