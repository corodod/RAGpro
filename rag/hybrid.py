# rag/hybrid.py
from pathlib import Path
from typing import List, Dict

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25 = bm25
        self.dense = dense

    def search(
        self,
        query: str,
        bm25_top_n: int = 100,
        top_k: int = 10,
    ) -> List[Dict]:
        bm25_res = self.bm25.search(query, top_k=bm25_top_n)
        candidate_ids = [r["chunk_id"] for r in bm25_res]

        dense_res = self.dense.rerank_candidates(
            query, candidate_ids, top_k=top_k
        )

        bm25_scores = {r["chunk_id"]: r["bm25_score"] for r in bm25_res}
        for r in dense_res:
            r["bm25_score"] = bm25_scores.get(r["chunk_id"], 0.0)

        return dense_res


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
    CHUNKS = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"

    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
        embedding_dim=768,
    )
    dense.load()

    hybrid = HybridRetriever(bm25, dense)

    # q = "кто совершил убийство, которое спровоцировало первую мировую войну"
    q = "Серб Гаврило Принцип"
    for r in hybrid.search(q, top_k=8):
        print("=" * 80)
        print("BM25:", r["bm25_score"], "DENSE:", r["dense_score"])
        print(r["text"][:500])
