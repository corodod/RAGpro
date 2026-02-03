# rag/coverage.py
from __future__ import annotations

from typing import List, Dict
import numpy as np


class CoverageSelector:
    """
    Coverage-aware greedy selector.

    Goal:
      - maximize semantic coverage of the question
      - avoid redundant chunks
      - stop when marginal gain is small

    This module is:
      - deterministic
      - training-free
      - interpretable
    """

    def __init__(
        self,
        *,
        epsilon: float,
        max_chunks: int,
        alpha: float,
    ):
        """
        Args:
          epsilon: minimum coverage gain to continue selection
          max_chunks: safety cap
          alpha: weight for mean vs max coverage
                 coverage = alpha * mean + (1-alpha) * max
        """
        self.epsilon = epsilon
        self.max_chunks = max_chunks
        self.alpha = alpha

    # -----------------------------------------------------

    def select(
        self,
        *,
        query_emb: np.ndarray,
        candidates: List[Dict],
        emb_key: str = "dense_emb",
    ) -> List[Dict]:
        if not candidates:
            return []

        q = self._normalize(query_emb)

        # pre-normalize candidate embeddings
        cand_embs = {
            c["chunk_id"]: self._normalize(c[emb_key])
            for c in candidates
            if emb_key in c
        }

        selected: List[Dict] = []
        selected_embs: List[np.ndarray] = []

        current_coverage = 0.0

        while len(selected) < self.max_chunks:
            best_gain = 0.0
            best_chunk = None
            best_emb = None

            for c in candidates:
                cid = c["chunk_id"]
                if cid not in cand_embs:
                    continue
                if any(cid == s["chunk_id"] for s in selected):
                    continue

                emb = cand_embs[cid]

                new_coverage = self._coverage(
                    q,
                    selected_embs + [emb],
                )

                gain = new_coverage - current_coverage

                if gain > best_gain:
                    best_gain = gain
                    best_chunk = c
                    best_emb = emb

            if best_chunk is None or best_gain < self.epsilon:
                break

            selected.append(best_chunk)
            selected_embs.append(best_emb)
            current_coverage += best_gain

        return selected

    # -----------------------------------------------------

    def _coverage(
        self,
        q: np.ndarray,
        embs: List[np.ndarray],
    ) -> float:
        if not embs:
            return 0.0

        sims = [float(q @ e) for e in embs]
        max_sim = max(sims)

        mean_emb = np.mean(embs, axis=0)
        mean_emb = self._normalize(mean_emb)
        mean_sim = float(q @ mean_emb)

        return self.alpha * mean_sim + (1.0 - self.alpha) * max_sim

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)