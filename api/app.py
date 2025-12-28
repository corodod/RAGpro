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


# ================= PATHS =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

backend = os.getenv("GEN_BACKEND", "cpu")
device = "cuda" if backend == "cuda" else "cpu"

# ---------- Build retriever ----------
bm25 = BM25Retriever.load(INDEX_DIR)

dense = DenseRetriever(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
)
dense.load()

retriever = Retriever(
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
    candidates: List[Dict[str, Any]]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    candidates = retriever.retrieve(req.question)
    return {"question": req.question, "candidates": candidates}


@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")