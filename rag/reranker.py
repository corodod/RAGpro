# rag/reranker.py
from typing import List, Dict
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: str = "cpu",
        batch_size: int = 32,
        use_fp16: bool = True,
    ):
        self.device = device
        self.batch_size = batch_size

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

        # fp16 — только если CUDA
        self.use_fp16 = use_fp16 and device.startswith("cuda")

    def score(
        self,
        query: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        if not candidates:
            return candidates

        pairs = [(query, c["text"]) for c in candidates]

        with torch.no_grad():
            if self.use_fp16:
                # 🔥 новый правильный autocast API
                with torch.amp.autocast("cuda"):
                    scores = self.model.predict(
                        pairs,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                    )
            else:
                scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                )

        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)

        return candidates
