# scripts/eval_recall.py
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever


def build_qrels_map():
    qrels = load_dataset("kngrg/rubq-qrels")["train"]
    # ожидаем поля: query-id, corpus-id (названия могут отличаться)
    # посмотрим безопасно:
    cols = qrels.column_names
    qid_key = "query-id" if "query-id" in cols else cols[0]
    did_key = "corpus-id" if "corpus-id" in cols else cols[1]

    m = {}
    for row in qrels:
        qid = str(row[qid_key])
        did = str(row[did_key])
        m.setdefault(qid, set()).add(did)
    return m


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"

    # Queries
    queries_ds = load_dataset("kaengreg/rubq", "queries")["train"]
    # обычно там есть _id и text
    q_cols = queries_ds.column_names
    qid_key = "_id" if "_id" in q_cols else q_cols[0]
    qtext_key = "text" if "text" in q_cols else q_cols[1]

    qrels_map = build_qrels_map()

    # BM25
    bm25 = BM25Retriever(CHUNKS_PATH)

    # Dense (FAISS)
    INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
    INDEX_PATH = INDEX_DIR / "faiss.index"
    META_PATH = INDEX_DIR / "faiss_meta.json"
    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        model_name="intfloat/multilingual-e5-small",
        index_path=INDEX_PATH,
        meta_path=META_PATH,
    )
    if INDEX_PATH.exists() and META_PATH.exists():
        dense.load()
    else:
        dense.build_index(batch_size=64, save=True)

    hybrid = HybridRetriever(bm25, dense)

    K = 10
    total = 0
    hit_bm25 = 0
    hit_hybrid = 0

    for row in tqdm(queries_ds, desc="Evaluating"):
        qid = str(row[qid_key])
        q = row[qtext_key]

        gold = qrels_map.get(qid)
        if not gold:
            continue

        total += 1

        bm25_res = bm25.search(q, top_k=K)
        bm25_ids = {r["chunk_id"] for r in bm25_res}
        if bm25_ids & gold:
            hit_bm25 += 1

        hybrid_res = hybrid.search(q, bm25_top_n=50, top_k=K)
        hybrid_ids = {r["chunk_id"] for r in hybrid_res}
        if hybrid_ids & gold:
            hit_hybrid += 1

    print(f"Total evaluated queries: {total}")
    print(f"BM25 recall@{K}:   {hit_bm25 / total:.4f}")
    print(f"Hybrid recall@{K}: {hit_hybrid / total:.4f}")


if __name__ == "__main__":
    main()
