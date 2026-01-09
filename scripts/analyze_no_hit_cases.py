import json
from pathlib import Path
from collections import defaultdict

# ================== PATHS ==================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NO_HIT_PATH = PROJECT_ROOT / "eval_reports" / "no_hit.jsonl"
WIKI_PATH = PROJECT_ROOT / "data" / "raw" / "wiki_corpus.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"

OUT_PATH = PROJECT_ROOT / "eval_reports" / "no_hit_analysis.json"

# ================== LOAD NO_HIT ==================

no_hit_items = []

with open(NO_HIT_PATH, encoding="utf-8") as f:
    for line in f:
        no_hit_items.append(json.loads(line))

print(f"[ИНФО] Загружено no_hit вопросов: {len(no_hit_items)}")

# ================== COLLECT GOLD DOC IDS ==================

needed_doc_ids = set()

for item in no_hit_items:
    for doc_id in item.get("gold_doc_ids", []):
        needed_doc_ids.add(str(doc_id))

print(f"[ИНФО] Уникальных gold_doc_ids: {len(needed_doc_ids)}")

# ================== LOAD WIKI DOCS ==================

wiki_docs = {}

with open(WIKI_PATH, encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = str(doc.get("doc_id") or doc.get("id"))
        if doc_id in needed_doc_ids:
            wiki_docs[doc_id] = doc

print(f"[ИНФО] Найдено документов в wiki: {len(wiki_docs)}")

# ================== LOAD CHUNKS ==================

doc_chunks = defaultdict(list)

with open(CHUNKS_PATH, encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        doc_id = str(chunk.get("doc_id"))
        if doc_id in needed_doc_ids:
            doc_chunks[doc_id].append({
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
            })

print(f"[ИНФО] Загружены чанки для gold-документов")

# ================== BUILD ANALYSIS ==================

analysis = []

for item in no_hit_items:
    entry = {
        "question": item.get("question"),
        "gold_doc_ids": item.get("gold_doc_ids"),
        "miss_stage": item.get("miss_stage"),
        "entities": item.get("entities"),
        "entity_hit": item.get("entity_hit"),
        "documents": [],
    }

    for doc_id in item.get("gold_doc_ids", []):
        doc_id = str(doc_id)
        doc = wiki_docs.get(doc_id)

        if not doc:
            entry["documents"].append({
                "doc_id": doc_id,
                "error": "Документ не найден в wiki_corpus",
            })
            continue

        entry["documents"].append({
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "chunks": doc_chunks.get(doc_id, []),
        })

    analysis.append(entry)

# ================== SAVE ==================

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"[OK] Анализ сохранён: {OUT_PATH}")
