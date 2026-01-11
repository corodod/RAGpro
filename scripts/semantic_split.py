# scripts/semantic_split.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from razdel import sentenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer


# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wiki_corpus.jsonl"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "wiki_chunks.jsonl"


# ================== MODELS ==================
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

NER_MODEL = "ekaterinatao/nerel-bio-rubert-base"

ner = pipeline(
    "ner",
    model=NER_MODEL,
    aggregation_strategy="simple",
    device=0, # поставь 0 если есть GPU
)

ner_tokenizer = AutoTokenizer.from_pretrained(
    NER_MODEL,
    use_fast=True,
)


# ================== PARAMS ==================
# semantic chunking
MAX_CHARS = 500
MIN_CHARS = 120
SIM_THRESHOLD = 0.58

# short-doc merge
DOC_EMB_MAX_CHARS = 1200
MERGED_MAX_CHARS = 1200

# entities
ENTITY_MAX = 12
ENTITY_PREFIX_HEADER = "[СУЩНОСТИ]"
ENTITY_JOINER = "; "
NER_MAX_TOKENS = 480  # безопасно <512

# только то, что реально нужно
ALLOWED_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "LOC",
    "EVENT",
}
ENTITY_LABEL_MAP = {
    # люди
    "PERSON": "PERSON",
    "PER": "PERSON",
    # организации
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    # география
    "LOC": "LOC",
    "GPE": "LOC",
    "CITY": "LOC",
    "COUNTRY": "LOC",
    # события
    "EVENT": "EVENT",
}


# ================== UTILS ==================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def embed_doc(text: str) -> np.ndarray:
    text = (text or "").strip()
    if len(text) > DOC_EMB_MAX_CHARS:
        text = text[:DOC_EMB_MAX_CHARS]
    return embedder.encode(
        [f"passage: {text}"],
        normalize_embeddings=True,
    )[0]


def looks_like_proper_name(text: str) -> bool:
    return any(ch.isupper() for ch in text)


# ================== NER ==================
def extract_doc_entities(text: str) -> List[str]:
    if not text:
        return []

    sentences = [s.text.strip() for s in sentenize(text) if s.text.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(ner_tokenizer.encode(sent, add_special_tokens=False))

        if sent_len > NER_MAX_TOKENS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.append(sent)
            continue

        if current_len + sent_len <= NER_MAX_TOKENS:
            current.append(sent)
            current_len += sent_len
        else:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len

    if current:
        chunks.append(" ".join(current))

    collected: List[str] = []

    for chunk in chunks:
        ents = ner(chunk)

        for e in ents:
            raw_label = e.get("entity_group")
            label = ENTITY_LABEL_MAP.get(raw_label)

            if not label:
                continue

            if label in ALLOWED_ENTITY_TYPES:
                word = e.get("word", "").strip()
                word = word.strip(" .,—–-")

                if len(word) >= 3 and looks_like_proper_name(word):
                    collected.append(word)

    # dedup с сохранением порядка
    seen = set()
    uniq: List[str] = []
    for e in collected:
        if e not in seen:
            uniq.append(e)
            seen.add(e)

    return uniq[:ENTITY_MAX]


def build_entity_prefix(entities: List[str]) -> str:
    if not entities:
        return ""
    return f"{ENTITY_PREFIX_HEADER}\n{ENTITY_JOINER.join(entities)}\n\n"


def with_entities(entities: List[str], text: str) -> str:
    prefix = build_entity_prefix(entities)
    return prefix + text.strip() if prefix else text.strip()


# ================== SEMANTIC CHUNKING ==================
def semantic_chunk_doc(text: str) -> List[str]:
    sentences = [s.text.strip() for s in sentenize(text) if len(s.text.strip()) > 20]
    if len(sentences) < 2:
        return []

    sent_embs = embedder.encode(
        [f"passage: {s}" for s in sentences],
        normalize_embeddings=True,
    )

    chunks: List[str] = []
    current = sentences[0]
    anchor = sent_embs[0]

    for sent, emb in zip(sentences[1:], sent_embs[1:]):
        if cosine(anchor, emb) >= SIM_THRESHOLD and len(current) + len(sent) < MAX_CHARS:
            current += " " + sent
        else:
            if len(current) >= MIN_CHARS:
                chunks.append(current)
            current = sent
            anchor = emb

    if len(current) >= MIN_CHARS:
        chunks.append(current)

    return chunks


# ================== SHORT DOC MERGE ==================
def build_short_chunk(
    short_text: str,
    prev_chunk: Optional[str],
    next_chunk: Optional[str],
    use_prev: bool,
    use_next: bool,
) -> str:
    parts: List[str] = []

    if use_prev and prev_chunk:
        parts.append("[КОНТЕКСТ_ПРЕД]\n" + prev_chunk.strip())

    parts.append(short_text.strip())

    if use_next and next_chunk:
        parts.append("[КОНТЕКСТ_СЛЕД]\n" + next_chunk.strip())

    merged = "\n\n".join(parts)
    return merged[:MERGED_MAX_CHARS].rstrip()


# ================== MAIN ==================
def main() -> None:
    chunk_count = 0

    prev_normal_emb: Optional[np.ndarray] = None
    prev_last_chunk: Optional[str] = None
    prev_entities: List[str] = []

    pending_shorts: List[Dict[str, Any]] = []

    with open(RAW_PATH, encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Semantic splitting + NER"):
            doc = json.loads(line)

            doc_id = str(doc.get("doc_id") or doc.get("_id") or doc.get("id"))
            title = doc.get("title", "") or ""
            text = (doc.get("text") or "").strip()
            if not text:
                continue

            # ---------- NORMAL DOC ----------
            doc_chunks = semantic_chunk_doc(text)

            if doc_chunks:
                entities = extract_doc_entities(text)

                first_chunk = with_entities(entities, doc_chunks[0])
                last_chunk = with_entities(entities, doc_chunks[-1])
                doc_emb = embed_doc(text)

                # обработка short-доков перед normal
                for sd in pending_shorts:
                    short_emb = sd["emb"]

                    sim_prev = cosine(short_emb, prev_normal_emb) if prev_normal_emb is not None else -1
                    sim_next = cosine(short_emb, doc_emb)

                    use_prev = sim_prev >= sim_next
                    use_next = not use_prev

                    merged = build_short_chunk(
                        short_text=sd["text"],
                        prev_chunk=prev_last_chunk,
                        next_chunk=first_chunk,
                        use_prev=use_prev,
                        use_next=use_next,
                    )

                    f_out.write(json.dumps({
                        "chunk_id": f"{sd['doc_id']}_{chunk_count}",
                        "doc_id": sd["doc_id"],
                        "title": sd["title"],
                        "section": "body",
                        "text": merged,
                        "entities": prev_entities,
                    }, ensure_ascii=False) + "\n")
                    chunk_count += 1

                pending_shorts.clear()

                # normal chunks
                for ch in doc_chunks:
                    f_out.write(json.dumps({
                        "chunk_id": f"{doc_id}_{chunk_count}",
                        "doc_id": doc_id,
                        "title": title,
                        "section": "body",
                        "text": with_entities(entities, ch),
                        "entities": entities,
                    }, ensure_ascii=False) + "\n")
                    chunk_count += 1

                prev_normal_emb = doc_emb
                prev_last_chunk = last_chunk
                prev_entities = entities

            # ---------- SHORT DOC ----------
            else:
                pending_shorts.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "emb": embed_doc(text),
                })

    print(f"[OK] Saved chunks to {OUT_PATH}")
    print(f"[OK] Total chunks: {chunk_count}")


if __name__ == "__main__":
    print(extract_doc_entities("ЦСКА — российский футбольный клуб из города Москва. Валерий Газзаев был тренером ЦСКА."))
    print(ner("ЦСКА — российский футбольный клуб из города Москва. Валерий Газзаев был тренером ЦСКА."))
    main()
