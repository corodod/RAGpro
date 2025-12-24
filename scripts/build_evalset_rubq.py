# scripts/build_evalset_rubq.py
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "rubq_eval.jsonl"

# Источники:
QUERIES_DS = "kaengreg/rubq"
QRELS_DS = "kaengreg/rubq-qrels"


def safe_get_question(row: dict) -> str:
    # разные датасеты называют поле по-разному
    return (
        row.get("question")
        or row.get("query")
        or row.get("text")
        or row.get("Question")
        or ""
    )


def safe_get_answers(row: dict):
    # answers может отсутствовать — тогда вернем []
    ans = row.get("answers") or row.get("answer") or row.get("Answer")
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(x) for x in ans]
    return [str(ans)]


def main():
    # 1) грузим qrels (query-id -> список corpus-id)
    qrels = load_dataset(QRELS_DS, "qrels", split="test")
    gold: dict[str, list[str]] = defaultdict(list)

    for r in tqdm(qrels, desc="Loading qrels"):
        qid = str(r["query-id"])
        doc_id = str(r["corpus-id"])
        score = int(r.get("score", 1))
        if score > 0:
            gold[qid].append(doc_id)

    # 2) грузим queries
    queries = load_dataset(QUERIES_DS, "queries", split="train")

    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for r in tqdm(queries, desc="Building evalset"):
            qid = str(r.get("_id") or r.get("id") or r.get("query-id") or r.get("qid") or "")
            if not qid:
                continue

            question = safe_get_question(r).strip()
            if not question:
                continue

            gold_doc_ids = gold.get(qid, [])
            if not gold_doc_ids:
                # нет разметки — пропускаем
                continue

            record = {
                "qid": qid,
                "question": question,
                "answers": safe_get_answers(r),
                "gold_doc_ids": gold_doc_ids,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] saved: {OUT_PATH}")
    print(f"[OK] n_written: {n_written}")


if __name__ == "__main__":
    main()