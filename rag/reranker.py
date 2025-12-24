# rag/reranker.py
from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: str = "cpu",
    ):
        self.model = CrossEncoder(model_name, device=device)

    def score(
        self,
        query: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Adds `ce_score` to each candidate.
        Does NOT truncate or sort.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)

        return candidates
