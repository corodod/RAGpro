# rag/reranker
from typing import List, Dict

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: str = "cpu",
    ):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 8,
    ) -> List[Dict]:
        """
        candidates: [{chunk_id, title, text, bm25_score, dense_score}]
        """
        pairs = [(query, c["text"]) for c in candidates]

        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)

        candidates = sorted(
            candidates,
            key=lambda x: x["ce_score"],
            reverse=True,
        )

        return candidates[:top_k]