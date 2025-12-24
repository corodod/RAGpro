# rag/hybrid.py
from pathlib import Path
from typing import List, Dict, Optional

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

    def search(
        self,
        query: str,
        bm25_top_n: int = 200,
        dense_top_n: int = 50,
        top_k: int = 12,
    ) -> List[Dict]:
        # 1️⃣ BM25 — high recall
        bm25_res = self.bm25.search(query, top_k=bm25_top_n)
        candidate_ids = [r["chunk_id"] for r in bm25_res]

        # 2️⃣ Dense rerank — semantic filtering
        dense_res = self.dense.rerank_candidates(
            query,
            candidate_ids,
            top_k=dense_top_n,
        )

        # прокидываем bm25_score
        bm25_scores = {r["chunk_id"]: r["bm25_score"] for r in bm25_res}
        for r in dense_res:
            r["bm25_score"] = bm25_scores.get(r["chunk_id"], 0.0)

        # 3️⃣ Cross-encoder — final relevance
        if self.reranker is not None:
            dense_res = self.reranker.rerank(
                query,
                dense_res,
                top_k=top_k,
            )
        else:
            dense_res = dense_res[:top_k]

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

    reranker = CrossEncoderReranker(device="cpu")

    hybrid = HybridRetriever(bm25, dense, reranker=reranker)

    q = "кто совершил убийство, которое спровоцировало первую мировую войну"
    for r in hybrid.search(q, top_k=12):
        print("=" * 80)
        print(
            "BM25:", r["bm25_score"],
            "DENSE:", r["dense_score"],
            "CE:", r.get("ce_score"),
        )
        print(r["text"][:500])
