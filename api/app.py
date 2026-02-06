# api/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.coverage import CoverageSelector
from rag.generator import AnswerGenerator, GeneratorConfig
from rag.retriever import Retriever, RetrieverConfig
from rag.agent_executor import PlanExecutorRetriever, ExecutorConfig


# ================= PATHS =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

# ================= FLAGS =================
USE_AGENT = False  # True -> Agentic RAG, False -> Simple RAG (base retriever)

# Остальные параметры можно оставить из ENV (не обязательно)
backend = os.getenv("GEN_BACKEND", "cpu")  # "cpu" | "cuda"
device = "cuda" if backend == "cuda" else "cpu"

GEN_TOP_K = int(os.getenv("GEN_TOP_K", "5"))
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes", "YES")

# ================= BUILD RETRIEVER =================
bm25 = BM25Retriever.load(INDEX_DIR)

cfg = RetrieverConfig()

dense = DenseRetriever(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
    model_name=cfg.dense_model_name,
    embedding_dim=cfg.dense_embedding_dim,
    query_prefix=cfg.dense_query_prefix,
    passage_prefix=cfg.dense_passage_prefix,
    default_search_top_k=cfg.dense_search_top_k,
    default_rerank_top_k=cfg.dense_rerank_top_k,
    default_return_embeddings=cfg.dense_rerank_return_embeddings,
)
dense.load()

base_retriever = Retriever(
    bm25=bm25,
    dense=dense,
    reranker=CrossEncoderReranker(
        model_name=cfg.cross_encoder_model_name,
        device=device,
        batch_size=cfg.cross_encoder_batch_size,
        use_fp16=cfg.cross_encoder_use_fp16,
    ),
    rewriter=None,
    entity_extractor=None,
    coverage_selector=CoverageSelector(
        epsilon=cfg.coverage_epsilon,
        max_chunks=cfg.coverage_max_chunks,
        alpha=cfg.coverage_alpha,
    ),
    config=cfg,
    debug=DEBUG,
)

generator = AnswerGenerator(
    GeneratorConfig(
        backend=backend,
        max_new_tokens=int(os.getenv("GEN_MAX_NEW_TOKENS", "80")),
    ),
)

retriever = PlanExecutorRetriever(
    base_retriever=base_retriever,
    generator=generator,
    reranker=base_retriever.reranker,
    cfg=ExecutorConfig(
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
        default_top_k=int(os.getenv("AGENT_TOP_K", "20")),
        max_fanout=int(os.getenv("AGENT_MAX_FANOUT", "15")),
        ce_threshold=float(os.getenv("AGENT_CE_THRESHOLD", "0.30")),
        top_per_entity=int(os.getenv("AGENT_TOP_PER_ENTITY", "2")),
        max_evidence=int(os.getenv("AGENT_MAX_EVIDENCE", "6")),
    ),
    debug=DEBUG,
)


# ================= FASTAPI =================
app = FastAPI(title="RAGRPO Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    question: str
    use_agent: Optional[bool] = None  # ✅ приходит из UI


class SearchResponse(BaseModel):
    question: str
    answer: str
    candidates: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None


def _print_top_chunks(candidates: List[Dict[str, Any]], top_k: int) -> None:
    for i, c in enumerate(candidates[:top_k], start=1):
        print(f"\n--- TOP {i} ---")
        print(f"chunk_id: {c.get('chunk_id')}")
        print(f"title: {c.get('title')}")
        print(f"text: {(c.get('text') or '')[:300]}")


def _collect_agent_debug() -> Dict[str, Any]:
    dbg: Dict[str, Any] = {}

    if getattr(retriever, "last_decomp", None) is not None:
        try:
            dbg["decomposition"] = [it.model_dump() for it in retriever.last_decomp.items]
        except Exception:
            dbg["decomposition"] = "failed_to_dump"

    if getattr(retriever, "last_compiled", None) is not None:
        try:
            dbg["compiled_plan"] = retriever.last_compiled.model_dump()
        except Exception:
            dbg["compiled_plan"] = "failed_to_dump"

    if getattr(retriever, "last_dsl_lines", None) is not None:
        dbg["dsl_lines"] = list(retriever.last_dsl_lines)

    if getattr(retriever, "last_plan", None) is not None:
        try:
            dbg["parsed_plan_steps"] = [
                {"id": s.id, "op": s.op, "out": s.out, "args": s.args}
                for s in retriever.last_plan.steps
            ]
        except Exception:
            dbg["parsed_plan_steps"] = "failed_to_dump"

    if getattr(retriever, "last_state", None):
        st = retriever.last_state
        keys = {}
        for k, v in st.items():
            if isinstance(v, list):
                keys[k] = f"list[{len(v)}]"
            else:
                keys[k] = type(v).__name__
        dbg["final_state_keys"] = keys

    if getattr(retriever, "last_answer", None):
        dbg["agent_answer"] = retriever.last_answer

    return dbg


@app.post("/rag", response_model=SearchResponse)
def search(req: SearchRequest):
    question = (req.question or "").strip()

    # ✅ режим на каждый запрос
    use_agent = req.use_agent if req.use_agent is not None else USE_AGENT

    print("\n================ REQUEST =================")
    print(f"Question: {question}")
    print(f"Mode: {'AGENTIC RAG' if use_agent else 'PLAIN RAG'}")

    print("\n================ RETRIEVAL =================")

    if use_agent:
        print("[Retrieval] Using AGENT executor")
        candidates = retriever.retrieve(question)
        print(f"[Retrieval] Agent returned {len(candidates)} chunks")

        if DEBUG:
            dbg = _collect_agent_debug()
            # ... дальше твой debug print как был ...
    else:
        print("[Retrieval] Using BASE retriever (no agent)")
        candidates = base_retriever.retrieve(question)
        print(f"[Retrieval] Retrieved {len(candidates)} chunks")
        dbg = {}

    _print_top_chunks(candidates, GEN_TOP_K)

    print("\n================ GENERATION =================")

    answer: Optional[str] = None

    if use_agent:
        answer = getattr(retriever, "last_answer", None)
        if answer:
            print("[Generation] Using answer produced by AGENT")
        else:
            print("[Generation] Agent did not produce answer → fallback to generator")

    if not answer:
        top_chunks = candidates[:GEN_TOP_K]
        answer = generator.generate(question, top_chunks)

    print("\n================ ANSWER =================")
    print(answer)
    print("=========================================\n")

    debug_payload = _collect_agent_debug() if (use_agent and DEBUG) else None

    return SearchResponse(
        question=question,
        answer=answer,
        candidates=candidates,
        debug=debug_payload,
    )



@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")
