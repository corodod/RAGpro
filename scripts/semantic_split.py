# scripts/semantic_split.py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from razdel import sentenize
from sentence_transformers import SentenceTransformer

# === Пути ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wiki_corpus.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "wiki_chunks.jsonl"

# === Модель ===
model = SentenceTransformer("intfloat/multilingual-e5-small")

# === Параметры чанкинга ===
MAX_CHARS = 800
MIN_CHARS = 200
SIM_THRESHOLD = 0.8

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

chunk_count = 0

with open(RAW_PATH, encoding="utf-8") as f_in, \
     open(OUT_PATH, "w", encoding="utf-8") as f_out:

    for line in tqdm(f_in, desc="Semantic splitting"):
        doc = json.loads(line)

        sentences = [s.text.strip() for s in sentenize(doc["text"]) if s.text.strip()]
        if len(sentences) < 2:
            continue

        sent_embs = model.encode(sentences, normalize_embeddings=True)

        current_chunk = sentences[0]
        current_emb = sent_embs[0]

        for sent, emb in zip(sentences[1:], sent_embs[1:]):
            sim = cosine(current_emb, emb)

            if sim > SIM_THRESHOLD and len(current_chunk) + len(sent) < MAX_CHARS:
                current_chunk += " " + sent
                current_emb = (current_emb + emb) / 2
            else:
                if len(current_chunk) >= MIN_CHARS:
                    record = {
                        "chunk_id": f"{doc['doc_id']}_{chunk_count}",
                        "doc_id": doc["doc_id"],
                        "title": doc.get("title", ""),
                        "text": current_chunk
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_count += 1

                current_chunk = sent
                current_emb = emb

        # последний чанк
        if len(current_chunk) >= MIN_CHARS:
            record = {
                "chunk_id": f"{doc['doc_id']}_{chunk_count}",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "text": current_chunk
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunk_count += 1

print(f"Saved chunks to {OUT_PATH}")
print(f"Total chunks: {chunk_count}")
