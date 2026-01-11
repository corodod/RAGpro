# scripts/semantic_split.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from razdel import sentenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer


# ================== FLAGS ==================
# 1) извлекать сущности из ДЛИННЫХ документов (short никогда не NER-им)
EXTRACT_ENTITIES = True

# 2) присоединять/не присоединять CONTEXT CHUNK к short-докам
ATTACH_CONTEXT_CHUNK_TO_SHORT = True

# 3) присоединять/не присоединять ENTITIES от соседнего long-дока к short-докам
ATTACH_ENTITIES_TO_SHORT = True


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
    device=0,  # поставь -1 если CPU
)

ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=True)


# ================== PARAMS ==================
# semantic chunking
MAX_CHARS = 500
MIN_CHARS = 120
SIM_THRESHOLD = 0.58

# short-doc merge
DOC_EMB_MAX_CHARS = 1200
MERGED_MAX_CHARS = 1200

# entities
ENTITY_MAX = 10
ENTITY_PREFIX_HEADER = "[СУЩНОСТИ]"
ENTITY_JOINER = "; "
NER_MAX_TOKENS = 480  # безопасно <512

ENTITY_SCORE_THRESHOLD = 0.37
MAX_ENTITY_TOKENS = 5

ALLOWED_ENTITY_TYPES = {"PERSON", "ORG", "LOC", "EVENT"}

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


# ================== REGEX ==================
EDGE_PUNCT_RE = re.compile(r'^[\s"«».,;:!?—–-]+|[\s"«».,;:!?—–-]+$')
PREP_PREFIX_RE = re.compile(r"^(из|в|на|по|при)\s+", re.IGNORECASE)
ABBR_RE = re.compile(r"^[A-ZА-ЯЁ]{2,10}$")


# ================== UTILS ==================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def embed_doc(text: str) -> np.ndarray:
    text = (text or "").strip()[:DOC_EMB_MAX_CHARS]
    return embedder.encode([f"passage: {text}"], normalize_embeddings=True)[0]


def normalize_entity(text: str) -> str:
    t = (text or "").strip()
    t = EDGE_PUNCT_RE.sub("", t)
    t = PREP_PREFIX_RE.sub("", t)
    return " ".join(t.split())


def is_abbreviation(text: str) -> bool:
    return bool(ABBR_RE.fullmatch((text or "").strip()))


def norm_key_first3(text: str) -> str:
    t = normalize_entity(text).lower()
    t = re.sub(r"[^\w]", "", t)
    return t[:3]


def dedup_by_first3_keep_form(entities: List[str], limit: int) -> List[str]:
    seen_keys = set()
    uniq: List[str] = []
    for ent in entities:
        k = norm_key_first3(ent)
        if not k:
            continue
        if k not in seen_keys:
            uniq.append(ent)  # сохраняем реальную форму
            seen_keys.add(k)
        if len(uniq) >= limit:
            break
    return uniq


# ================== NER (ТОЛЬКО ДЛИННЫЕ ДОКИ) ==================
def extract_doc_entities(text: str) -> List[str]:
    """
    Извлекаем сущности из одного ДЛИННОГО документа.
    Short-доки сюда не должны попадать (логика main() это обеспечивает).
    """
    if not EXTRACT_ENTITIES or not text:
        return []

    sentences = [s.text.strip() for s in sentenize(text) if s.text.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in sentences:
        l = len(ner_tokenizer.encode(s, add_special_tokens=False))

        if l > NER_MAX_TOKENS:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            toks = ner_tokenizer.encode(s, add_special_tokens=False)[:NER_MAX_TOKENS]
            chunks.append(ner_tokenizer.decode(toks, skip_special_tokens=True))
            continue

        if cur_len + l > NER_MAX_TOKENS:
            if cur:
                chunks.append(" ".join(cur))
            cur, cur_len = [], 0

        cur.append(s)
        cur_len += l

    if cur:
        chunks.append(" ".join(cur))

    raw_entities: List[str] = []

    for chunk in chunks:
        for e in ner(chunk):
            label = ENTITY_LABEL_MAP.get(e.get("entity_group"))
            if label not in ALLOWED_ENTITY_TYPES:
                continue

            word = normalize_entity(e.get("word", ""))
            if not word:
                continue

            # ❌ режем слишком длинные сущности
            # считаем по словам, а не по токенам модели
            if len(word.split()) > MAX_ENTITY_TOKENS:
                continue

            score = float(e.get("score", 0.0))

            # аббревиатуры пропускаем без score
            if is_abbreviation(word):
                raw_entities.append(word)
                continue

            # обычные сущности: score + длина
            if score >= ENTITY_SCORE_THRESHOLD and len(word) >= 2:
                raw_entities.append(word)

    return dedup_by_first3_keep_form(raw_entities, ENTITY_MAX)


def build_entity_prefix(entities: List[str]) -> str:
    if not entities:
        return ""
    return f"{ENTITY_PREFIX_HEADER}\n{ENTITY_JOINER.join(entities)}\n\n"


def with_entities(entities: List[str], text: str) -> str:
    return build_entity_prefix(entities) + text.strip() if entities else text.strip()


# ================== SEMANTIC CHUNKING ==================
def semantic_chunk_doc(text: str) -> List[str]:
    sents = [s.text.strip() for s in sentenize(text) if len(s.text.strip()) > 20]
    if len(sents) < 2:
        return []

    embs = embedder.encode([f"passage: {s}" for s in sents], normalize_embeddings=True)

    chunks: List[str] = []
    cur = sents[0]
    anchor = embs[0]

    for s, e in zip(sents[1:], embs[1:]):
        if cosine(anchor, e) >= SIM_THRESHOLD and len(cur) + len(s) < MAX_CHARS:
            cur += " " + s
        else:
            if len(cur) >= MIN_CHARS:
                chunks.append(cur)
            cur, anchor = s, e

    if len(cur) >= MIN_CHARS:
        chunks.append(cur)

    return chunks


# ================== SHORT DOC MERGE ==================
def build_short_chunk(
    short_text: str,
    prev_chunk: Optional[str],
    next_chunk: Optional[str],
    use_prev: bool,
    use_next: bool,
) -> str:
    if not ATTACH_CONTEXT_CHUNK_TO_SHORT:
        return short_text.strip()

    parts: List[str] = []
    if use_prev and prev_chunk:
        parts.append("[КОНТЕКСТ_ПРЕД]\n" + prev_chunk.strip())

    parts.append(short_text.strip())

    if use_next and next_chunk:
        parts.append("[КОНТЕКСТ_СЛЕД]\n" + next_chunk.strip())

    return "\n\n".join(parts)[:MERGED_MAX_CHARS].rstrip()


# ================== MAIN ==================
def main() -> None:
    chunk_count = 0

    prev_emb: Optional[np.ndarray] = None
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

            doc_chunks = semantic_chunk_doc(text)

            # ---------- LONG DOC ----------
            if doc_chunks:
                # ✅ сущности извлекаем только здесь (из long)
                entities = extract_doc_entities(text)
                doc_emb = embed_doc(text)

                first_chunk = with_entities(entities, doc_chunks[0])
                last_chunk = with_entities(entities, doc_chunks[-1])

                # обработка short-доков, которые встретились перед этим long
                for sd in pending_shorts:
                    sim_prev = cosine(sd["emb"], prev_emb) if prev_emb is not None else -1
                    sim_next = cosine(sd["emb"], doc_emb)

                    use_prev = sim_prev >= sim_next
                    use_next = not use_prev

                    merged_text = build_short_chunk(
                        short_text=sd["text"],
                        prev_chunk=prev_last_chunk,
                        next_chunk=first_chunk,
                        use_prev=use_prev,
                        use_next=use_next,
                    )

                    # ✅ short никогда не NER-им
                    merged_entities: List[str] = []
                    if ATTACH_ENTITIES_TO_SHORT:
                        neighbor = prev_entities if use_prev else entities
                        merged_entities = neighbor[:ENTITY_MAX]

                    f_out.write(json.dumps({
                        "chunk_id": f'{sd["doc_id"]}_{chunk_count}',
                        "doc_id": sd["doc_id"],
                        "title": sd.get("title", ""),
                        "section": "body",
                        "text": merged_text,
                        "entities": merged_entities,
                    }, ensure_ascii=False) + "\n")
                    chunk_count += 1

                pending_shorts.clear()

                # пишем чанки long-дока
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

                prev_emb = doc_emb
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
    # тест: тут НЕ извлекаем сущности, потому что тест — это просто строка.
    # Если хочешь проверить NER — прогоняй через extract_doc_entities вручную.
    # Но помни: по твоей логике это "short-like" и в пайплайне NER по нему не будет.
    print(extract_doc_entities("ЦСКА — российский футбольный клуб из города Москва. Валерий Газзаев был тренером ЦСКА."))
    print(ner("ЦСКА — российский футбольный клуб из города Москва. Валерий Газзаев был тренером ЦСКА."))
    main()
