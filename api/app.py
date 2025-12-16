# api/app.py
from __future__ import annotations
from fastapi.responses import HTMLResponse

from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever
from rag.generator import AnswerGenerator, GeneratorConfig


# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"


# ---------- load retrievers once ----------
bm25 = BM25Retriever.load(INDEX_DIR)

dense = DenseRetriever(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
    embedding_dim=768,
)
dense.load()

hybrid = HybridRetriever(bm25, dense)


# ---------- load generator once ----------
# Можно настроить модель тут
gen_cfg = GeneratorConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="auto",
    dtype="auto",
    max_new_tokens=220,
    temperature=0.2,
    top_p=0.9,
)
generator = AnswerGenerator(gen_cfg)


# ---------- api ----------
app = FastAPI(title="RAGRPO Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для локальной разработки
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    candidates = hybrid.search(req.question, bm25_top_n=req.bm25_top_n, top_k=req.top_k)
    return {"question": req.question, "candidates": candidates}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    # берём чанки по id из dense meta (там texts/titles/chunk_ids уже в памяти)
    id_to_pos = {cid: i for i, cid in enumerate(dense.chunk_ids)}

    selected_chunks = []
    for cid in req.selected_chunk_ids:
        if cid not in id_to_pos:
            continue
        pos = id_to_pos[cid]
        selected_chunks.append(
            {
                "chunk_id": dense.chunk_ids[pos],
                "title": dense.titles[pos],
                "text": dense.texts[pos],
            }
        )

    ans = generator.generate(req.question, selected_chunks)

    return {
        "question": req.question,
        "selected_chunks": selected_chunks,
        "answer": ans,
    }

# ---------- UI ----------
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")
