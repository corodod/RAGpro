# api/app.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector
from rag.generator import AnswerGenerator, GeneratorConfig
from rag.retriever import Retriever, RetrieverConfig
from rag.agent_executor import PlanExecutorRetriever, ExecutorConfig

# ================= PATHS =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

backend = os.getenv("GEN_BACKEND", "cpu")
device = "cuda" if backend == "cuda" else "cpu"

GEN_TOP_K = 5

# ---------- Build retriever ----------
bm25 = BM25Retriever.load(INDEX_DIR)

dense = DenseRetriever(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
)
dense.load()

base_retriever = Retriever(
    bm25=bm25,
    dense=dense,
    reranker=CrossEncoderReranker(device=device),
    rewriter=QueryRewriter(llm_device=device),
    entity_extractor=EntityExtractor(),
    coverage_selector=CoverageSelector(),
    config=RetrieverConfig(),
    debug=True,
)

generator = AnswerGenerator(
    GeneratorConfig(
        backend=backend,
        max_new_tokens=80,
    )
)
retriever = PlanExecutorRetriever(
    base_retriever=base_retriever,
    generator=generator,
    reranker=base_retriever.reranker,  # <-- используем твой CrossEncoderReranker
    cfg=ExecutorConfig(
        max_steps=8,
        default_top_k=20,
        max_fanout=15,
        ce_threshold=0.30,
        top_per_entity=2,
        max_evidence=6,
    ),
    debug=True,
)


# ---------- FastAPI ----------
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


@app.post("/rag", response_model=SearchResponse)
def search(req: SearchRequest):
    print("\n================ RETRIEVAL =================")
    print(f"Question: {req.question}")

    # 1️⃣ retrieve (ONE TIME)
    candidates = retriever.retrieve(req.question)

    print(f"Retrieved {len(candidates)} chunks")

    for i, c in enumerate(candidates[:GEN_TOP_K], start=1):
        print(f"\n--- TOP {i} ---")
        print(f"chunk_id: {c.get('chunk_id')}")
        print(f"title: {c.get('title')}")
        print(f"text: {(c.get('text') or '')[:300]}")

    print("\n================ GENERATION =================")

    # answer: берем плановый, если он есть; иначе fallback на обычную генерацию по top-k
    # после retrieve:
    answer = retriever.last_answer
    if not answer:
        top_chunks = candidates[:GEN_TOP_K]
        answer = retriever.llms.synthesizer.generate(req.question, top_chunks)

    print("\n================ ANSWER =================")
    print(answer)
    print("=========================================\n")

    return {
        "question": req.question,
        "answer": answer,
        "candidates": candidates,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")