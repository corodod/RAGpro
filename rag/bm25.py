# rag/bm25.py
import json
import re
import pickle
from pathlib import Path
from typing import List, Dict

BM25_DEFAULT_TOP_K = 10
# токенизация
BM25_TOKEN_PATTERN = r"[а-яa-z0-9]+"

class BM25Retriever:
    def __init__(self, bm25, chunk_ids, titles, texts):
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self.titles = titles
        self.texts = texts

    @staticmethod
    def tokenize(text: str):
        text = text.lower()
        return re.findall(BM25_TOKEN_PATTERN, text)

    @classmethod
    def load(cls, index_dir: Path):
        bm25_path = index_dir / "bm25.pkl"
        meta_path = index_dir / "bm25_meta.json"

        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        texts = []
        with open(meta["chunks_path"], encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])

        return cls(
            bm25=bm25,
            chunk_ids=meta["chunk_ids"],
            titles=meta["titles"],
            texts=texts,
        )

    def search(self, query: str, top_k: int = BM25_DEFAULT_TOP_K) -> List[Dict]:
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "title": self.titles[i],
                "text": self.texts[i],
                "bm25_score": float(scores[i]),
            }
            for i in ranked
        ]
