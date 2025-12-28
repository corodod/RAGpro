# scripts/eval_retrieval.py
import json
from pathlib import Path
from statistics import mean
from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.retriever import Retriever, RetrieverConfig


KS = [1, 3, 5, 10, 20]
DEVICE = "cuda"  # или "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"


def doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]


def recall_at_k(pred, gold, k):
    return 1.0 if any(d in gold for d in pred[:k]) else 0.0


def mrr_at_k(pred, gold, k):
    for i, d in enumerate(pred[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def main():
    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
    )
    dense.model.to(DEVICE)  # 🔥 КЛЮЧЕВО
    dense.load()

    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        reranker = CrossEncoderReranker(device="cuda"),
    )

    retriever = Retriever(
        hybrid=hybrid,
        dense=dense,
        config=RetrieverConfig(
            use_rewrites=False,
            use_entity_expansion=False,
            use_coverage=False,
        ),
    )

    recalls = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            res = retriever.retrieve(item["question"])
            pred = [doc_id_from_chunk_id(r["chunk_id"]) for r in res]

            for k in KS:
                recalls[k].append(recall_at_k(pred, set(item["gold_doc_ids"]), k))
                mrrs[k].append(mrr_at_k(pred, set(item["gold_doc_ids"]), k))

    print("\nRESULTS")
    for k in KS:
        print(f"Recall@{k}: {mean(recalls[k]):.4f} | MRR@{k}: {mean(mrrs[k]):.4f}")


if __name__ == "__main__":
    main()
