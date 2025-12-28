# scripts/eval_retrieval.py
import json
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker


# ================= CONFIG =================

KS = [1, 3, 5, 10, 20]

BM25_TOP_N = 200
DENSE_TOP_N = 50
FINAL_TOP_K = max(KS)

USE_CE = True
DEVICE = "cuda"        # 🔥 GPU for dense + CE
EMBEDDING_DIM = 1024

# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"


# ================= METRICS =================

def doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]


def recall_at_k(pred_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    return 1.0 if any(d in gold_doc_ids for d in pred_doc_ids[:k]) else 0.0


def mrr_at_k(pred_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    for i, d in enumerate(pred_doc_ids[:k], start=1):
        if d in gold_doc_ids:
            return 1.0 / i
    return 0.0


# ================= MAIN ====================

def main():
    # -------------------------------------------------
    # Load retrievers
    # -------------------------------------------------
    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
        embedding_dim=EMBEDDING_DIM,
        model_name="intfloat/multilingual-e5-large",
    )
    dense.model.to(DEVICE)   # 🔥 dense encoder → GPU
    dense.load()

    reranker = (
        CrossEncoderReranker(device=DEVICE)
        if USE_CE else None
    )

    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
    )

    # -------------------------------------------------
    # Eval loop
    # -------------------------------------------------
    recalls = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line)
            q = item["question"]
            gold_docs = set(map(str, item["gold_doc_ids"]))

            # ⚠️ чистый retrieval — без rewrites
            res = hybrid.search(
                query=q,
                rewrites=[],
                bm25_top_n=BM25_TOP_N,
                dense_top_n=DENSE_TOP_N,
                final_top_k=FINAL_TOP_K,
            )

            pred_doc_ids = [
                doc_id_from_chunk_id(r["chunk_id"])
                for r in res
            ]

            for k in KS:
                recalls[k].append(recall_at_k(pred_doc_ids, gold_docs, k))
                mrrs[k].append(mrr_at_k(pred_doc_ids, gold_docs, k))

    # -------------------------------------------------
    # Report
    # -------------------------------------------------
    print("\n================ RESULTS ================")
    print(f"n_queries = {len(recalls[KS[0]])}")
    for k in KS:
        print(f"Recall@{k}: {mean(recalls[k]):.4f} | MRR@{k}: {mean(mrrs[k]):.4f}")


if __name__ == "__main__":
    main()
