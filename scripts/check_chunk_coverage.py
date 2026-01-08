import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"


def load_chunk_doc_ids():
    doc_ids = set()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if cid:
                doc_ids.add(cid.split("_", 1)[0])
    return doc_ids


def main():
    chunk_doc_ids = load_chunk_doc_ids()
    print(f"[OK] Docs in chunks: {len(chunk_doc_ids)}")

    missing = Counter()
    total = 0

    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gold_ids = set(map(str, item["gold_doc_ids"]))
            total += len(gold_ids)

            for gid in gold_ids:
                if gid not in chunk_doc_ids:
                    missing[gid] += 1

    print("\n================ MISSING DOCS ================")
    print(f"Total gold doc refs: {total}")
    print(f"Missing unique docs: {len(missing)}")

    if missing:
        print("\nTop missing docs (by frequency):")
        for doc_id, cnt in missing.most_common(20):
            print(f"doc_id={doc_id}  count={cnt}")


if __name__ == "__main__":
    main()
