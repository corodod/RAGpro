# scripts/semantic_split.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
SIM_THRESHOLD = 0.65  # внутри-документная семантическая связность (anchor)

# ================== SHORT-DOC BRIDGING PARAMS ==================
# Если doc не дал ни одного чанка (doc_chunks == []), считаем его "short/bad"
# и "приклеиваем" контекст от соседнего NORMAL документа.

DOC_EMB_MAX_CHARS = 1200        # сколько символов брать для doc-emb (ускорение)
MERGED_MAX_CHARS = 1200         # максимум символов у склеенного short-чанка

# пороги для решения "приклеивать к prev или next"
BRIDGE_SIM_MIN = 0.45           # если обе близости ниже — всё равно выбираем лучшую, но считаем "слабой"
BRIDGE_TIE_MARGIN = 0.03        # если sim_prev и sim_next почти равны — можно приклеить оба
BRIDGE_ATTACH_BOTH_MIN = 0.55   # минимальная близость, чтобы приклеивать ОБА контекста (если tie)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings уже нормализованы, dot == cosine
    return float(np.dot(a, b))


def embed_doc(text: str) -> np.ndarray:
    text = (text or "").strip()
    if len(text) > DOC_EMB_MAX_CHARS:
        text = text[:DOC_EMB_MAX_CHARS]
    # e5 ожидает префикс passage:
    emb = model.encode([f"passage: {text}"], normalize_embeddings=True)[0]
    return emb


def semantic_chunk_doc(text: str) -> List[str]:
    """
    Возвращает список чанков для нормального документа.
    Если чанков не получилось — вернёт [].
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.text.strip() for s in sentenize(text) if len(s.text.strip()) > 20]
    if len(sentences) < 2:
        return []

    sent_embs = model.encode(
        [f"passage: {s}" for s in sentences],
        normalize_embeddings=True,
    )

    chunks: List[str] = []
    current_chunk = sentences[0]
    chunk_anchor_emb = sent_embs[0]  # фиксированный якорь

    for sent, emb in zip(sentences[1:], sent_embs[1:]):
        sim = cosine(chunk_anchor_emb, emb)

        if sim >= SIM_THRESHOLD and (len(current_chunk) + 1 + len(sent) < MAX_CHARS):
            current_chunk += " " + sent
        else:
            if len(current_chunk) >= MIN_CHARS:
                chunks.append(current_chunk)

            current_chunk = sent
            chunk_anchor_emb = emb

    if len(current_chunk) >= MIN_CHARS:
        chunks.append(current_chunk)

    return chunks


def build_short_chunk(
    short_text: str,
    short_id: str,
    prev_last_chunk: Optional[str],
    next_first_chunk: Optional[str],
    sim_prev: Optional[float],
    sim_next: Optional[float],
) -> str:
    """
    Склеиваем short документ с контекстом.
    По умолчанию кладём short текст как основной, а контекст — как [КОНТЕКСТ_*].
    """
    short_text = (short_text or "").strip()

    parts: List[str] = []
    used_prev = False
    used_next = False

    # decide attach
    if prev_last_chunk is None and next_first_chunk is None:
        parts = [short_text]
    elif prev_last_chunk is None:
        used_next = True
    elif next_first_chunk is None:
        used_prev = True
    else:
        assert sim_prev is not None and sim_next is not None

        # tie -> оба, если достаточно высокий уровень
        if abs(sim_prev - sim_next) <= BRIDGE_TIE_MARGIN and max(sim_prev, sim_next) >= BRIDGE_ATTACH_BOTH_MIN:
            used_prev = True
            used_next = True
        else:
            used_prev = sim_prev >= sim_next
            used_next = not used_prev

    # assemble (контекст обрамляем маркерами, чтобы было видно)
    if used_prev and prev_last_chunk:
        parts.append("[КОНТЕКСТ_ПРЕД]\n" + prev_last_chunk.strip())

    parts.append(short_text)

    if used_next and next_first_chunk:
        parts.append("[КОНТЕКСТ_СЛЕД]\n" + next_first_chunk.strip())

    merged = "\n\n".join([p for p in parts if p.strip()])

    # ограничим длину, чтобы не раздувать индекс
    if len(merged) > MERGED_MAX_CHARS:
        merged = merged[:MERGED_MAX_CHARS].rstrip()

    return merged


def main() -> None:
    chunk_count = 0

    # состояние последнего NORMAL документа
    prev_normal_id: Optional[str] = None
    prev_normal_emb: Optional[np.ndarray] = None
    prev_last_chunk: Optional[str] = None

    # буфер short документов между prev_normal и next_normal
    pending_shorts: List[Dict[str, Any]] = []

    # статистика
    n_docs = 0
    n_normal = 0
    n_short = 0
    n_short_attached_prev = 0
    n_short_attached_next = 0
    n_short_attached_both = 0
    n_short_no_context = 0

    def flush_pending_shorts(
        next_normal_id: Optional[str],
        next_normal_emb: Optional[np.ndarray],
        next_first_chunk: Optional[str],
        f_out,
    ) -> None:
        """
        Разруливаем все short-доки, накопленные до next_normal.
        Если next_normal_* == None, значит конец корпуса: приклеиваем только к prev.
        """
        nonlocal chunk_count
        nonlocal n_short_attached_prev, n_short_attached_next, n_short_attached_both, n_short_no_context

        if not pending_shorts:
            return

        for sd in pending_shorts:
            short_id = sd["doc_id"]
            title = sd.get("title", "")
            text = sd.get("text", "")
            short_emb = sd.get("emb")

            # similarity
            sim_prev = None
            sim_next = None

            if prev_normal_emb is not None and short_emb is not None:
                sim_prev = cosine(short_emb, prev_normal_emb)
            if next_normal_emb is not None and short_emb is not None:
                sim_next = cosine(short_emb, next_normal_emb)

            # если обе есть — выбираем
            used_prev = False
            used_next = False

            if prev_last_chunk is None and next_first_chunk is None:
                used_prev = used_next = False
            elif prev_last_chunk is None:
                used_next = True
            elif next_first_chunk is None:
                used_prev = True
            else:
                # обе стороны доступны
                assert sim_prev is not None and sim_next is not None

                # если обе слабые — всё равно берём лучшую сторону
                if max(sim_prev, sim_next) < BRIDGE_SIM_MIN:
                    used_prev = sim_prev >= sim_next
                    used_next = not used_prev
                else:
                    # tie -> оба (если достаточно высокая близость)
                    if abs(sim_prev - sim_next) <= BRIDGE_TIE_MARGIN and max(sim_prev, sim_next) >= BRIDGE_ATTACH_BOTH_MIN:
                        used_prev = True
                        used_next = True
                    else:
                        used_prev = sim_prev >= sim_next
                        used_next = not used_prev

            # статистика (до сборки, чтобы считать фактическое использование)
            if used_prev and used_next:
                n_short_attached_both += 1
            elif used_prev:
                n_short_attached_prev += 1
            elif used_next:
                n_short_attached_next += 1
            else:
                n_short_no_context += 1

            merged_text = build_short_chunk(
                short_text=text,
                short_id=short_id,
                prev_last_chunk=prev_last_chunk if used_prev else None,
                next_first_chunk=next_first_chunk if used_next else None,
                sim_prev=sim_prev,
                sim_next=sim_next,
            )

            record = {
                "chunk_id": f"{short_id}_{chunk_count}",
                "doc_id": short_id,
                "title": title,
                "section": "body",
                "text": merged_text,
                # опционально (не ломает пайплайн)
                "bridge": {
                    "prev_doc_id": prev_normal_id,
                    "next_doc_id": next_normal_id,
                    "sim_prev": sim_prev,
                    "sim_next": sim_next,
                    "used_prev": used_prev,
                    "used_next": used_next,
                },
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunk_count += 1

        pending_shorts.clear()

    with open(RAW_PATH, encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Semantic splitting (doc-aware)"):
            n_docs += 1
            doc = json.loads(line)

            doc_id = str(doc.get("doc_id") or doc.get("_id") or doc.get("id"))
            title = doc.get("title", "") or ""
            text = (doc.get("text") or "").strip()

            if not text:
                continue

            # 1) пробуем сделать normal-chunks
            doc_chunks = semantic_chunk_doc(text)

            # 2) short/bad doc -> откладываем до встречи next normal
            if not doc_chunks:
                n_short += 1
                pending_shorts.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "emb": embed_doc(text),
                })
                continue

            # 3) normal doc: сначала получаем first/last + emb
            n_normal += 1
            next_first_chunk = doc_chunks[0]
            next_last_chunk = doc_chunks[-1]
            next_emb = embed_doc(text)

            # 4) как только встретили normal — можем разрулить pending_shorts перед ним
            flush_pending_shorts(
                next_normal_id=doc_id,
                next_normal_emb=next_emb,
                next_first_chunk=next_first_chunk,
                f_out=f_out,
            )

            # 5) пишем чанки текущего normal документа
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

            # 6) обновляем prev_normal состояние
            prev_normal_id = doc_id
            prev_normal_emb = next_emb
            prev_last_chunk = next_last_chunk

        # 7) конец корпуса: разруливаем хвост short-доков (только prev доступен)
        flush_pending_shorts(
            next_normal_id=None,
            next_normal_emb=None,
            next_first_chunk=None,
            f_out=f_out,
        )

    print("\n================ SUMMARY ================")
    print(f"[OK] Saved chunks to: {OUT_PATH}")
    print(f"[OK] Total chunks: {chunk_count}")
    print(f"[OK] Docs processed: {n_docs}")
    print(f"[OK] Normal docs: {n_normal}")
    print(f"[OK] Short docs:  {n_short}")
    print("\n[Short bridging stats]")
    print(f"attached prev: {n_short_attached_prev}")
    print(f"attached next: {n_short_attached_next}")
    print(f"attached both: {n_short_attached_both}")
    print(f"no context:    {n_short_no_context}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
