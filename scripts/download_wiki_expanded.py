# scripts/download_wiki_expanded.py
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm


# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

OUT_PATH = DATA_RAW / "wiki_expanded.jsonl"


# ================== CONFIG ==================
MAX_PARAGRAPHS = 5        # сколько абзацев брать после intro
MAX_CHARS_TOTAL = 6000    # жёсткий лимит на статью


# ================== LOAD RuBQ CORPUS IDS ==================
print("[1] Loading RuBQ corpus doc_ids...")
rubq_corpus = load_dataset("kaengreg/rubq", "corpus", split="train")
rubq_doc_ids = set(str(x["_id"]) for x in rubq_corpus)

print(f"[OK] RuBQ doc_ids: {len(rubq_doc_ids)}")


# ================== LOAD WIKIPEDIA ==================
print("[2] Loading Russian Wikipedia dump...")
wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.ru",
    split="train",
    streaming=True,
)



# ================== BUILD EXPANDED CORPUS ==================
written = 0

with open(OUT_PATH, "w", encoding="utf-8") as f_out:
    for page in tqdm(wiki, desc="Processing wiki pages"):
        doc_id = str(page.get("id"))
        if doc_id not in rubq_doc_ids:
            continue

        title = page.get("title", "")
        text = page.get("text", "").strip()
        if not text:
            continue

        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
        if not paragraphs:
            continue

        # ---- собираем intro + первые абзацы ----
        collected = []
        total_chars = 0

        for p in paragraphs[: MAX_PARAGRAPHS + 1]:
            if total_chars + len(p) > MAX_CHARS_TOTAL:
                break
            collected.append(p)
            total_chars += len(p)

        if not collected:
            continue

        record = {
            "doc_id": doc_id,
            "title": title,
            "text": "\n\n".join(collected),
        }

        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

print(f"[OK] Saved expanded corpus: {OUT_PATH}")
print(f"[OK] Documents written: {written}")
