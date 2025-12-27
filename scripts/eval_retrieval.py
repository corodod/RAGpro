# scripts/eval_retrieval.py
import json
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever

# если хочешь тестировать CE:
from rag.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"


KS = [1, 3, 5, 10, 20]


def doc_id_from_chunk_id(chunk_id: str) -> str:
    # твой chunk_id: "{doc_id}_{chunk_count}"
    return chunk_id.split("_", 1)[0]


def recall_at_k(pred_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    topk = pred_doc_ids[:k]
    return 1.0 if any(d in gold_doc_ids for d in topk) else 0.0


def mrr_at_k(pred_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    topk = pred_doc_ids[:k]
    for i, d in enumerate(topk, start=1):
        if d in gold_doc_ids:
            return 1.0 / i
    return 0.0


def main():
    # --- retrievers ---
    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
        embedding_dim=1024,  # у тебя в build_dense EMB_DIM=1024
    )
    dense.load()

    # включай/выключай CE одним флагом
    USE_CE = True
    reranker = CrossEncoderReranker(device="cpu") if USE_CE else None

    hybrid = HybridRetriever(bm25=bm25, dense=dense, reranker=reranker)

    # --- eval loop ---
    recalls = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    n = 0
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line)
            q = item["question"]
            gold_docs = set(map(str, item["gold_doc_ids"]))

            # главное: сделай k >= max(KS), чтобы было из чего считать
            res = hybrid.search(
                query=q,
                rewrites=[],  # 👈 важно
                bm25_top_n=200,
                dense_top_n=50,
                final_top_k=max(KS),  # 👈 вместо top_k
            )

            pred_doc_ids = [doc_id_from_chunk_id(r["chunk_id"]) for r in res]

            for k in KS:
                recalls[k].append(recall_at_k(pred_doc_ids, gold_docs, k))
                mrrs[k].append(mrr_at_k(pred_doc_ids, gold_docs, k))

            n += 1

    print("\n================ RESULTS ================")
    print(f"n_queries = {n}")
    for k in KS:
        print(f"Recall@{k}: {mean(recalls[k]):.4f} | MRR@{k}: {mean(mrrs[k]):.4f}")


if __name__ == "__main__":
    main()
'''
Этот скрипт отвечает ТОЛЬКО на один вопрос:

Нашёл ли retrieval нужный документ?

Он:

Берёт вопрос

Запускает твой HybridRetriever

Получает список chunk_id

Превращает их в doc_id

Сравнивает с gold_doc_ids

Считает метрики:

- Recall@k    “Попал ли хотя бы один правильный документ в топ-k?”
- MRR@k       “Насколько высоко был первый правильный документ?”
'''

