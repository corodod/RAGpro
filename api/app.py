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
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector


# ============================== CONFIG ==================================

BM25_TOP_N = 300
DENSE_TOP_N = 100
FINAL_TOP_K = 20
MAX_NEW_TOKENS = 80

N_REWRITES = 2
MIN_COSINE = 0.75

CE_STRONG_THRESHOLD = 11.2

DEBUG_RETRIEVAL = True

# =======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
UI_PATH = PROJECT_ROOT / "ui" / "index.html"

backend = os.getenv("GEN_BACKEND", "cpu")
device = "cuda" if backend == "cuda" else "cpu"
print(f"[API] GEN_BACKEND={backend}")

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
)

rewriter = QueryRewriter(
    llm_device=device,
    n_rewrites=N_REWRITES,
    min_cosine=MIN_COSINE,
)

entity_extractor = EntityExtractor()

coverage_selector = CoverageSelector(
    epsilon=0.01,
    max_chunks=8,
    alpha=0.5,
)

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
    rewrites: List[str]
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
    q0 = req.question

    # =====================================================
    # 0️⃣ Query rewriting
    # =====================================================
    rewrites = rewriter.rewrite(q0)

    if DEBUG_RETRIEVAL:
        print("\n" + "=" * 80)
        print("[QUERY]")
        print(q0)

        print("[REWRITES]")
        if rewrites:
            for r in rewrites:
                print("  -", r)
        else:
            print("  (none)")

    # =====================================================
    # 1️⃣ Base retrieval + CE confidence gate
    # =====================================================
    base = hybrid.search(
        query=q0,
        rewrites=rewrites,
        bm25_top_n=BM25_TOP_N,
        dense_top_n=DENSE_TOP_N,
        final_top_k=FINAL_TOP_K,
        t_strong=CE_STRONG_THRESHOLD,
    )

    if base:
        max_ce = base[0].get("ce_score", 0.0)
    else:
        max_ce = None

    if max_ce is not None and max_ce >= CE_STRONG_THRESHOLD:
        if DEBUG_RETRIEVAL:
            print("[PATH] STRONG_CE_EXIT")
            print(f"[CE] max_ce = {max_ce:.3f} ≥ {CE_STRONG_THRESHOLD}")

        return {
            "question": q0,
            "rewrites": rewrites,
            "candidates": base,
        }

    # =====================================================
    # 2️⃣ Recall expansion (LOW CONFIDENCE)
    # =====================================================
    if DEBUG_RETRIEVAL:
        print("[PATH] LOW_CONFIDENCE → ENTITY_EXPANSION")
        if max_ce is not None:
            print(f"[CE] max_ce = {max_ce:.3f} < {CE_STRONG_THRESHOLD}")
        else:
            print("[CE] missing")

    entities = entity_extractor.extract(q0)

    if DEBUG_RETRIEVAL:
        print("[ENTITIES]")
        if entities:
            for e in entities:
                print("  -", e)
        else:
            print("  (none)")

    expanded = []
    for e in entities:
        expanded.extend(
            hybrid.search(
                query=e,
                rewrites=[],
                bm25_top_n=BM25_TOP_N // 2,
                dense_top_n=DENSE_TOP_N // 2,
                final_top_k=FINAL_TOP_K,
            )
        )

    # =====================================================
    # 3️⃣ Union base + expanded
    # =====================================================
    by_id = {r["chunk_id"]: r for r in base + expanded}
    candidates = list(by_id.values())

    if not candidates:
        if DEBUG_RETRIEVAL:
            print("[RESULT] no candidates after expansion")

        return {
            "question": q0,
            "rewrites": rewrites,
            "candidates": [],
        }

    # =====================================================
    # 4️⃣ Dense rerank (q0 only)
    # =====================================================
    candidates = dense.rerank_candidates(
        q0,
        [c["chunk_id"] for c in candidates],
        top_k=len(candidates),
    )

    # =====================================================
    # 5️⃣ Coverage-aware selection
    # =====================================================
    q_emb = dense.encode_query(q0)
    selected = coverage_selector.select(
        query_emb=q_emb,
        candidates=candidates,
        emb_key="dense_emb",
    )

    if DEBUG_RETRIEVAL:
        print("[COVERAGE] selected chunks:")
        for r in selected:
            print(f"  - {r['chunk_id']}")

    return {
        "question": q0,
        "rewrites": rewrites,
        "candidates": strip_internal_fields(base),
    }
def strip_internal_fields(chunks: List[Dict]) -> List[Dict]:
    for c in chunks:
        c.pop("dense_emb", None)
    return chunks

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