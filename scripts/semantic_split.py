import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from razdel import sentenize
from sentence_transformers import SentenceTransformer


# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wiki_expanded.jsonl"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "wiki_chunks.jsonl"


# ================== MODEL ==================
model = SentenceTransformer("intfloat/multilingual-e5-small")


# ================== CHUNK PARAMS ==================
MAX_CHARS = 800
MIN_CHARS = 200
SIM_THRESHOLD = 0.80


def cosine(a, b):
    return float(np.dot(a, b))


chunk_count = 0

with open(RAW_PATH, encoding="utf-8") as f_in, \
     open(OUT_PATH, "w", encoding="utf-8") as f_out:

    for line in tqdm(f_in, desc="Semantic splitting"):
        doc = json.loads(line)

        sentences = [
            s.text.strip()
            for s in sentenize(doc["text"])
            if len(s.text.strip()) > 20
        ]

        if len(sentences) < 2:
            continue

        sent_embs = model.encode(
            [f"passage: {s}" for s in sentences],
            normalize_embeddings=True,
        )

        current_chunk = sentences[0]
        current_emb = sent_embs[0]

        for sent, emb in zip(sentences[1:], sent_embs[1:]):
            sim = cosine(current_emb, emb)

            if sim >= SIM_THRESHOLD and len(current_chunk) + len(sent) < MAX_CHARS:
                current_chunk += " " + sent
                current_emb = (current_emb + emb) / 2
                current_emb /= np.linalg.norm(current_emb) + 1e-12
            else:
                if len(current_chunk) >= MIN_CHARS:
                    record = {
                        "chunk_id": f"{doc['doc_id']}_{chunk_count}",
                        "doc_id": doc["doc_id"],
                        "title": doc.get("title", ""),
                        "section": "body",
                        "text": current_chunk,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_count += 1

                current_chunk = sent
                current_emb = emb

        # ---- last chunk ----
        if len(current_chunk) >= MIN_CHARS:
            record = {
                "chunk_id": f"{doc['doc_id']}_{chunk_count}",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "section": "body",
                "text": current_chunk,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunk_count += 1

print(f"[OK] Saved chunks to {OUT_PATH}")
print(f"[OK] Total chunks: {chunk_count}")
