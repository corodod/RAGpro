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


# ============================== CONFIG ==================================

BM25_TOP_N = 300          # для агрессивного eval: 300–500
DENSE_TOP_N = 100          # для агрессивного eval: 100
FINAL_TOP_K = 20          # для Recall@20 → ставь 20
MAX_NEW_TOKENS = 80
# ========================================================================


# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"


# ---------- RETRIEVERS ----------
bm25 = BM25Retriever.load(INDEX_DIR)

dense = DenseRetriever(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
)
dense.load()

backend = os.getenv("GEN_BACKEND", "cpu")
print(f"[API] GEN_BACKEND={backend}")

ce_device = "cuda" if backend == "cuda" else "cpu"
reranker = CrossEncoderReranker(device=ce_device)

hybrid = HybridRetriever(
    bm25=bm25,
    dense=dense,
    reranker=reranker,
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
    candidates = hybrid.search(
        query=req.question,
        bm25_top_n=BM25_TOP_N,
        dense_top_n=DENSE_TOP_N,
        final_top_k=FINAL_TOP_K,
    )
    return {
        "question": req.question,
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