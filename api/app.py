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
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.generator import AnswerGenerator, GeneratorConfig
from rag.rewrite import QueryRewriter


# ============================== CONFIG ==================================

BM25_TOP_N = 300          # агрессивный recall
DENSE_TOP_N = 100         # dense filtering
FINAL_TOP_K = 20          # Recall@20
MAX_NEW_TOKENS = 80

# rewrite config
N_REWRITES = 2
MIN_COSINE = 0.75

# cross-encoder strong answer gate
CE_STRONG_THRESHOLD = 1.25   # ← 1.2–1.5
# ========================================================================


# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

# ---------- BACKEND ----------
backend = os.getenv("GEN_BACKEND", "cpu")
print(f"[API] GEN_BACKEND={backend}")

device = "cuda" if backend == "cuda" else "cpu"


# ---------- RETRIEVERS ----------
bm25 = BM25Retriever.load(INDEX_DIR)

dense = DenseRetriever(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
)
dense.load()

reranker = CrossEncoderReranker(device=device)

hybrid = HybridRetriever(
    bm25=bm25,
    dense=dense,
    reranker=reranker,
    ce_strong_threshold = CE_STRONG_THRESHOLD,
)


# ---------- QUERY REWRITER ----------
rewriter = QueryRewriter(
    llm_device=device,
    n_rewrites=N_REWRITES,
    min_cosine=MIN_COSINE,
)


# ---------- GENERATOR ----------
gen_cfg = GeneratorConfig(
    backend=backend,
    max_new_tokens=MAX_NEW_TOKENS,
)
generator = AnswerGenerator(gen_cfg)


# ---------- FASTAPI ----------
app = FastAPI(title="RAGRPO Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- SCHEMAS ----------
class SearchRequest(BaseModel):
    question: str


class SearchResponse(BaseModel):
    question: str
    candidates: List[Dict[str, Any]]
    rewrites: List[str]


class AnswerRequest(BaseModel):
    question: str
    selected_chunk_ids: List[str]


class AnswerResponse(BaseModel):
    question: str
    selected_chunks: List[Dict[str, Any]]
    answer: str


# ---------- ROUTES ----------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    rewrites = rewriter.rewrite(req.question)

    candidates = hybrid.search(
        query=req.question,
        rewrites=rewrites,          # ✅ ВОТ ЭТОГО НЕ ХВАТАЛО
        bm25_top_n=BM25_TOP_N,
        dense_top_n=DENSE_TOP_N,
        final_top_k=FINAL_TOP_K,
    )

    # debug logs
    print("=" * 80)
    print("Q0:", req.question)
    print("Rewrites:", rewrites)
    print("Top chunk_ids:", [c["chunk_id"] for c in candidates[:5]])

    return {
        "question": req.question,
        "rewrites": rewrites,
        "candidates": candidates,
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    id_to_pos = {cid: i for i, cid in enumerate(dense.chunk_ids)}

    selected_chunks = []
    for cid in req.selected_chunk_ids:
        pos = id_to_pos.get(cid)
        if pos is None:
            continue
        selected_chunks.append(
            {
                "chunk_id": dense.chunk_ids[pos],
                "title": dense.titles[pos],
                "text": dense.texts[pos],
            }
        )

    answer_text = generator.generate(req.question, selected_chunks)

    return {
        "question": req.question,
        "selected_chunks": selected_chunks,
        "answer": answer_text,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")