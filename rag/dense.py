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

        self.model = SentenceTransformer(model_name)
        self.index = None

        self.chunk_ids: List[str] = []
        self.titles: List[str] = []
        self.texts: List[str] = []

        self._id_map: Dict[str, int] = {}

    @staticmethod
    def _normalize_2d(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _normalize_1d(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.chunk_ids = meta["chunk_ids"]
        self.titles = meta["titles"]
        self.texts = meta["texts"]

        self._id_map = {cid: i for i, cid in enumerate(self.chunk_ids)}

    # -----------------------------
    # Public helpers
    # -----------------------------

    def encode_query(self, query: str) -> np.ndarray:
        """
        Returns normalized query embedding (1D float32).
        Compatible with CoverageSelector.
        """
        q = f"query: {query}"
        qv = self.model.encode([q], convert_to_numpy=True).astype("float32")
        qv = self._normalize_2d(qv)
        return qv[0]  # 1D

    def get_chunk_embeddings(self, chunk_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Returns normalized embeddings for given chunk_ids as {chunk_id: emb(1D)}.
        """
        if self.index is None:
            raise RuntimeError("DenseRetriever index is not loaded. Call dense.load().")

        positions = [self._id_map[cid] for cid in chunk_ids if cid in self._id_map]
        out: Dict[str, np.ndarray] = {}
        for cid, pos in zip([cid for cid in chunk_ids if cid in self._id_map], positions):
            v = self.index.reconstruct(pos).astype("float32")
            out[cid] = self._normalize_1d(v)
        return out

    # -----------------------------
    # Retrieval
    # -----------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("DenseRetriever index is not loaded. Call dense.load().")

        qv = self.encode_query(query).reshape(1, -1)  # 2D for FAISS

        scores, idxs = self.index.search(qv, top_k)

        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            out.append(
                {
                    "chunk_id": self.chunk_ids[i],
                    "title": self.titles[i],
                    "text": self.texts[i],
                    "dense_score": float(score),
                    # dense_emb here is optional; search() обычно не надо для coverage
                }
            )
        return out

    def rerank_candidates(
        self,
        query: str,
        candidate_chunk_ids: List[str],
        top_k: int = 10,
        return_embeddings: bool = True,
    ) -> List[Dict]:
        """
        Rerank given candidate chunk_ids by cosine similarity with query.
        Returns top_k candidates.

        If return_embeddings=True, adds:
          - dense_emb: normalized 1D embedding for coverage-aware selection
        """
        if self.index is None:
            raise RuntimeError("DenseRetriever index is not loaded. Call dense.load().")

        positions = [self._id_map[cid] for cid in candidate_chunk_ids if cid in self._id_map]
        if not positions:
            return []

        cand_vecs = np.vstack([self.index.reconstruct(p) for p in positions]).astype("float32")
        cand_vecs = self._normalize_2d(cand_vecs)

        qv = self.encode_query(query)  # 1D normalized
        scores = cand_vecs @ qv  # (n,)

        best = np.argsort(-scores)[: min(top_k, len(scores))]

        out: List[Dict] = []
        for rank_i in best:
            pos = positions[rank_i]
            cid = self.chunk_ids[pos]

            item = {
                "chunk_id": cid,
                "title": self.titles[pos],
                "text": self.texts[pos],
                "dense_score": float(scores[rank_i]),
            }

            if return_embeddings:
                # store normalized 1D embedding (needed by CoverageSelector)
                item["dense_emb"] = cand_vecs[rank_i]

            out.append(item)

        return out

    def encode_passage(self, text: str) -> np.ndarray:
        """
        Encode arbitrary document-like text (HyDE passage).
        Returns normalized 1D embedding.
        """
        p = f"passage: {text}"
        v = self.model.encode([p], convert_to_numpy=True).astype("float32")
        v = self._normalize_2d(v)
        return v[0]
