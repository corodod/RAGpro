# scripts/build_dense.py
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


# ---------- sanity check ----------
print("CUDA available:", torch.cuda.is_available())


# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
INDEX_DIR.mkdir(exist_ok=True, parents=True)


# ---------- config ----------
MODEL_NAME = "intfloat/multilingual-e5-large"
EMB_DIM = 1024
BATCH_SIZE = 128   # для RTX 3050 оптимально (если OOM → 64)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main():
    # ---------- load model ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    # ускорение на GPU
    if device == "cuda":
        model = model.half()

    # ---------- load chunks ----------
    chunk_ids, titles, texts = [], [], []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunk_ids.append(item["chunk_id"])
            titles.append(item.get("title", ""))
            texts.append(f"passage: {item['text']}")

    # ---------- FAISS index ----------
    index = faiss.IndexFlatIP(EMB_DIM)
    all_vecs = []

    # ---------- encode ----------
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i : i + BATCH_SIZE]

        vecs = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        vecs = normalize(vecs)
        all_vecs.append(vecs)

    all_vecs = np.vstack(all_vecs)
    index.add(all_vecs)

    # ---------- save ----------
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "faiss_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunk_ids": chunk_ids,
                "titles": titles,
                "texts": texts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
