# scripts/semantic_split.py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from razdel import sentenize
from sentence_transformers import SentenceTransformer


# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wiki_corpus.jsonl"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "wiki_chunks.jsonl"


# ================== MODEL ==================
model = SentenceTransformer("intfloat/multilingual-e5-small")


# ================== CHUNK PARAMS ==================
MAX_CHARS = 500
MIN_CHARS = 120
SIM_THRESHOLD = 0.65


def cosine(a, b):
    return float(np.dot(a, b))


chunk_count = 0

# 🔑 храним последний чанк предыдущего документа
prev_last_chunk: str | None = None


with open(RAW_PATH, encoding="utf-8") as f_in, \
     open(OUT_PATH, "w", encoding="utf-8") as f_out:

    for line in tqdm(f_in, desc="Semantic splitting"):
        doc = json.loads(line)
        doc_id = str(doc["doc_id"])
        title = doc.get("title", "")
        text = (doc.get("text") or "").strip()

        if not text:
            continue

        sentences = [
            s.text.strip()
            for s in sentenize(text)
            if len(s.text.strip()) > 20
        ]

        doc_chunks = []

        # ==================================================
        # CASE 1: нормальный документ → semantic chunking
        # ==================================================
        if len(sentences) >= 2:
            sent_embs = model.encode(
                [f"passage: {s}" for s in sentences],
                normalize_embeddings=True,
            )

            current_chunk = sentences[0]
            chunk_anchor_emb = sent_embs[0]
            chunk_mean_emb = sent_embs[0]

            for sent, emb in zip(sentences[1:], sent_embs[1:]):
                sim = cosine(chunk_anchor_emb, emb)

                if sim >= SIM_THRESHOLD and len(current_chunk) + len(sent) < MAX_CHARS:
                    current_chunk += " " + sent
                    chunk_mean_emb = chunk_mean_emb + emb
                    chunk_mean_emb /= np.linalg.norm(chunk_mean_emb) + 1e-12
                else:
                    if len(current_chunk) >= MIN_CHARS:
                        doc_chunks.append(current_chunk)

                    current_chunk = sent
                    chunk_anchor_emb = emb
                    chunk_mean_emb = emb

            if len(current_chunk) >= MIN_CHARS:
                doc_chunks.append(current_chunk)

        # ==================================================
        # CASE 2: короткий / плохой документ
        # ==================================================
        if not doc_chunks:
            if prev_last_chunk:
                merged = (
                    text.strip()
                    + "\n\n[КОНТЕКСТ]\n"
                    + prev_last_chunk.strip()
                )
                doc_chunks = [merged]
            else:
                # крайний случай: вообще без контекста
                doc_chunks = [text.strip()]

        # ==================================================
        # SAVE CHUNKS
        # ==================================================
        for chunk_text in doc_chunks:
            record = {
                "chunk_id": f"{doc_id}_{chunk_count}",
                "doc_id": doc_id,
                "title": title,
                "section": "body",
                "text": chunk_text,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunk_count += 1

        # 🔒 обновляем последний чанк документа
        prev_last_chunk = doc_chunks[-1]


print(f"[OK] Saved chunks to {OUT_PATH}")
print(f"[OK] Total chunks: {chunk_count}")
