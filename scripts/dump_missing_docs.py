import json
from pathlib import Path

# ================= ПУТИ =================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NO_HIT_PATH = PROJECT_ROOT / "eval_reports" / "no_hit.jsonl"

# источник документов
WIKI_PATH = PROJECT_ROOT / "data" / "raw" / "wiki_corpus.jsonl"

OUT_PATH = PROJECT_ROOT / "eval_reports" / "missing_docs.json"

# ================= ЗАГРУЗКА NO_HIT =================

missing_doc_ids = set()

with open(NO_HIT_PATH, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("missing_from_chunks"):
            for doc_id in item.get("gold_doc_ids", []):
                missing_doc_ids.add(str(doc_id))

print(f"[ИНФО] Найдено gold-документов, отсутствующих в чанках: {len(missing_doc_ids)}")

# ================= ПОИСК В WIKI =================

documents = []
not_found = set(missing_doc_ids)

with open(WIKI_PATH, encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = str(doc.get("id") or doc.get("doc_id"))

        if doc_id in missing_doc_ids:
            documents.append(doc)
            not_found.discard(doc_id)

# ================= СОХРАНЕНИЕ =================

result = {
    "stats": {
        "missing_doc_ids_total": len(missing_doc_ids),
        "documents_found": len(documents),
        "documents_not_found": len(not_found),
    },
    "documents": documents,
    "not_found_doc_ids": sorted(not_found),
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# ================= ЛОГ =================

print(f"[ИНФО] Документов найдено и сохранено: {len(documents)}")
print(f"[ИНФО] Документов не найдено в wiki-источнике: {len(not_found)}")

if not_found:
    print("[ПРЕДУПРЕЖДЕНИЕ] Примеры отсутствующих doc_id:")
    print(list(not_found)[:10])
else:
    print("[OK] Все отсутствующие документы найдены в wiki-источнике")

print(f"[OK] Итоговый файл сохранён: {OUT_PATH}")
