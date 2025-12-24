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
from rag.generator import AnswerGenerator, GeneratorConfig


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
    embedding_dim=768,
)
dense.load()

hybrid = HybridRetriever(bm25, dense)


# ---------- GENERATOR ----------
backend = os.getenv("GEN_BACKEND", "cpu")
print(f"[API] GEN_BACKEND={backend}")

gen_cfg = GeneratorConfig(
    backend=backend,
    max_new_tokens=80,
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
    bm25_top_n: int = 100
    top_k: int = 8


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
        req.question,
        bm25_top_n=req.bm25_top_n,
        top_k=req.top_k,
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