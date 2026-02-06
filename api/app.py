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



import rag, pydantic
print("[DEBUG] rag package:", rag.__file__)
print("[DEBUG] pydantic version:", pydantic.__version__)

from rag import compiled_plan_schema, json_to_dsl, compiler
print("[DEBUG] compiled_plan_schema:", compiled_plan_schema.__file__)
print("[DEBUG] json_to_dsl:", json_to_dsl.__file__)
print("[DEBUG] compiler:", compiler.__file__)

# ================= PATHS =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

# ================= ENV / FLAGS =================
backend = os.getenv("GEN_BACKEND", "cpu")  # "cpu" | "cuda"
device = "cuda" if backend == "cuda" else "cpu"

GEN_TOP_K = int(os.getenv("GEN_TOP_K", "5"))
USE_AGENT = os.getenv("USE_AGENT", "1") in ("1", "true", "True", "yes", "YES")

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
        device=device,  # динамически cpu/cuda
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


class SearchResponse(BaseModel):
    question: str
    answer: str
    candidates: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None  # чтобы удобно смотреть пайплайн (опционально)


def _print_top_chunks(candidates: List[Dict[str, Any]], top_k: int) -> None:
    for i, c in enumerate(candidates[:top_k], start=1):
        print(f"\n--- TOP {i} ---")
        print(f"chunk_id: {c.get('chunk_id')}")
        print(f"title: {c.get('title')}")
        print(f"text: {(c.get('text') or '')[:300]}")


def _collect_agent_debug() -> Dict[str, Any]:
    dbg: Dict[str, Any] = {}

    # Decomposition (L1)
    if getattr(retriever, "last_decomp", None) is not None:
        try:
            dbg["decomposition"] = [it.model_dump() for it in retriever.last_decomp.items]
        except Exception:
            dbg["decomposition"] = "failed_to_dump"

    # Compiled JSON (L2)
    if getattr(retriever, "last_compiled", None) is not None:
        try:
            dbg["compiled_plan"] = retriever.last_compiled.model_dump()
        except Exception:
            dbg["compiled_plan"] = "failed_to_dump"

    # DSL lines (translator)
    if getattr(retriever, "last_dsl_lines", None) is not None:
        dbg["dsl_lines"] = list(retriever.last_dsl_lines)

    # Parsed Plan (executor plan)
    if getattr(retriever, "last_plan", None) is not None:
        try:
            dbg["parsed_plan_steps"] = [
                {"id": s.id, "op": s.op, "out": s.out, "args": s.args}
                for s in retriever.last_plan.steps
            ]
        except Exception:
            dbg["parsed_plan_steps"] = "failed_to_dump"

    # Final state keys (what was produced)
    if getattr(retriever, "last_state", None):
        st = retriever.last_state
        keys = {}
        for k, v in st.items():
            if isinstance(v, list):
                keys[k] = f"list[{len(v)}]"
            else:
                keys[k] = type(v).__name__
        dbg["final_state_keys"] = keys

    # Final answer (agent)
    if getattr(retriever, "last_answer", None):
        dbg["agent_answer"] = retriever.last_answer

    return dbg


@app.post("/rag", response_model=SearchResponse)
def search(req: SearchRequest):
    question = (req.question or "").strip()

    print("\n================ REQUEST =================")
    print(f"Question: {question}")
    print(f"Mode: {'AGENTIC RAG' if USE_AGENT else 'PLAIN RAG'}")

    # ================= RETRIEVAL =================
    print("\n================ RETRIEVAL =================")

    if USE_AGENT:
        print("[Retrieval] Using AGENT executor")
        candidates = retriever.retrieve(question)
        print(f"[Retrieval] Agent returned {len(candidates)} chunks")

        if DEBUG:
            # debug: decomposition, compiled json, dsl, plan, state
            dbg = _collect_agent_debug()

            if "decomposition" in dbg:
                print("\n[Agent] Decomposition:")
                if isinstance(dbg["decomposition"], list):
                    for it in dbg["decomposition"]:
                        print("  -", it)
                else:
                    print("  -", dbg["decomposition"])

            if "compiled_plan" in dbg:
                print("\n[Agent] Compiled plan (JSON):")
                print(dbg["compiled_plan"])

            if "dsl_lines" in dbg:
                print("\n[Agent] DSL lines:")
                for l in dbg["dsl_lines"]:
                    print("  ", l)

            if "parsed_plan_steps" in dbg:
                print("\n[Agent] Parsed Plan:")
                if isinstance(dbg["parsed_plan_steps"], list):
                    for s in dbg["parsed_plan_steps"]:
                        print(f"  - {s['id']}: {s['op']} -> {s['out']} | args={s['args']}")
                else:
                    print("  -", dbg["parsed_plan_steps"])

            if "final_state_keys" in dbg:
                print("\n[Agent] Final state keys:")
                for k, v in dbg["final_state_keys"].items():
                    print(f"  - {k}: {v}")

    else:
        print("[Retrieval] Using BASE retriever (no agent)")
        candidates = base_retriever.retrieve(question)
        print(f"[Retrieval] Retrieved {len(candidates)} chunks")
        dbg = {}

    # ---- print TOP-K chunks ----
    _print_top_chunks(candidates, GEN_TOP_K)

    # ================= GENERATION =================
    print("\n================ GENERATION =================")

    answer: Optional[str] = None

    if USE_AGENT:
        answer = getattr(retriever, "last_answer", None)
        if answer:
            print("[Generation] Using answer produced by AGENT")
        else:
            print("[Generation] Agent did not produce answer → fallback to generator")

    if not answer:
        top_chunks = candidates[:GEN_TOP_K]
        answer = generator.generate(question, top_chunks)

    # ================= ANSWER =================
    print("\n================ ANSWER =================")
    print(answer)
    print("=========================================\n")

    # API debug payload (optional)
    debug_payload = _collect_agent_debug() if (USE_AGENT and DEBUG) else None

    return SearchResponse(
        question=question,
        answer=answer,
        candidates=candidates,
        debug=debug_payload,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")
