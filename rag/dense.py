# rag/dense.py
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        chunks_path: Path,
        index_path: Path,
        meta_path: Path,
        model_name: str = "intfloat/multilingual-e5-large",
        embedding_dim: int = 1024,
    ):
        self.chunks_path = chunks_path
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedding_dim = embedding_dim

        # ❗ Никакого trust_remote_code
        self.model = SentenceTransformer(model_name)
        self.index = None

        self.chunk_ids = []
        self.titles = []
        self.texts = []

    @staticmethod
    def _normalize(x: np.ndarray):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.chunk_ids = meta["chunk_ids"]
        self.titles = meta["titles"]
        self.texts = meta["texts"]

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        q = f"query: {query}"
        qv = self.model.encode([q], convert_to_numpy=True).astype("float32")
        qv = self._normalize(qv)

        scores, idxs = self.index.search(qv, top_k)

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "title": self.titles[i],
                "text": self.texts[i],
                "dense_score": float(score),
            }
            for score, i in zip(scores[0], idxs[0])
            if i >= 0
        ]

    def rerank_candidates(
        self,
        query: str,
        candidate_chunk_ids: List[str],
        top_k: int = 10,
    ) -> List[Dict]:
        id_map = {cid: i for i, cid in enumerate(self.chunk_ids)}
        positions = [id_map[cid] for cid in candidate_chunk_ids if cid in id_map]
        if not positions:
            return []

        cand_vecs = np.vstack([self.index.reconstruct(p) for p in positions])
        cand_vecs = self._normalize(cand_vecs)

        q = f"query: {query}"
        qv = self.model.encode([q], convert_to_numpy=True).astype("float32")
        qv = self._normalize(qv)

        scores = cand_vecs @ qv[0]
        best = np.argsort(-scores)[:top_k]

        return [
            {
                "chunk_id": self.chunk_ids[positions[i]],
                "title": self.titles[positions[i]],
                "text": self.texts[positions[i]],
                "dense_score": float(scores[i]),
            }
            for i in best
        ]