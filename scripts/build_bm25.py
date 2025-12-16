# scripts/build_bm25.py
import json
import re
import pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

BM25_PATH = INDEX_DIR / "bm25.pkl"
BM25_META_PATH = INDEX_DIR / "bm25_meta.json"


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[а-яa-z0-9]+", text)


def main():
    documents = []
    chunk_ids = []
    titles = []

    print("Loading chunks...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            documents.append(tokenize(item["text"]))
            chunk_ids.append(item["chunk_id"])
            titles.append(item.get("title", ""))

    print("Building BM25...")
    bm25 = BM25Okapi(documents)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    meta = {
        "chunk_ids": chunk_ids,
        "titles": titles,
        "chunks_path": str(CHUNKS_PATH),
    }
    with open(BM25_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"BM25 saved to {BM25_PATH}")


if __name__ == "__main__":
    main()
