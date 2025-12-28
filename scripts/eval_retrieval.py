# scripts/eval_retrieval.py
import json
from pathlib import Path
from statistics import mean
from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.retriever import Retriever, RetrieverConfig
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector


# ===================== CONFIG =====================

KS = [1, 3, 5, 10, 20]
DEVICE = "cuda"          # "cuda" или "cpu"

# retrieval policy (одна точка правды)
RETRIEVER_CONFIG = RetrieverConfig(
    bm25_top_n=300,
    dense_top_n=100,
    final_top_k=max(KS),

    use_rewrites=False,
    use_cross_encoder=True,
    ce_strong_threshold=2.5,

    use_entity_expansion=True,
    entity_bm25_top_n=150,
    entity_dense_top_n=50,

    use_coverage=True,
)

# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"


# ===================== METRICS =====================

def doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]


def recall_at_k(pred_doc_ids, gold_doc_ids, k):
    return 1.0 if any(d in gold_doc_ids for d in pred_doc_ids[:k]) else 0.0


def mrr_at_k(pred_doc_ids, gold_doc_ids, k):
    for i, d in enumerate(pred_doc_ids[:k], start=1):
        if d in gold_doc_ids:
            return 1.0 / i
    return 0.0


# ===================== MAIN =====================

def main():
    # ---------- Load retrievers ----------
    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
    )
    dense.model.to(DEVICE)     # 🔥 Dense encoder → GPU/CPU
    dense.load()

    reranker = (
        CrossEncoderReranker(device=DEVICE)
        if RETRIEVER_CONFIG.use_cross_encoder
        else None
    )

    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
    )

    # ---------- Optional modules ----------
    rewriter = (
        QueryRewriter(
            llm_device=DEVICE,
            n_rewrites=2,
            min_cosine=0.75,
        )
        if RETRIEVER_CONFIG.use_rewrites
        else None
    )

    entity_extractor = (
        EntityExtractor()
        if RETRIEVER_CONFIG.use_entity_expansion
        else None
    )

    coverage_selector = (
        CoverageSelector()
        if RETRIEVER_CONFIG.use_coverage
        else None
    )

    # ---------- Unified Retriever ----------
    retriever = Retriever(
        hybrid=hybrid,
        dense=dense,
        rewriter=rewriter,
        entity_extractor=entity_extractor,
        coverage_selector=coverage_selector,
        config=RETRIEVER_CONFIG,
        debug=False,
    )

    # ---------- Eval loop ----------
    recalls = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line)

            res = retriever.retrieve(item["question"])
            pred_doc_ids = [
                doc_id_from_chunk_id(r["chunk_id"])
                for r in res
            ]

            gold = set(map(str, item["gold_doc_ids"]))

            for k in KS:
                recalls[k].append(recall_at_k(pred_doc_ids, gold, k))
                mrrs[k].append(mrr_at_k(pred_doc_ids, gold, k))

    # ---------- Report ----------
    print("\n================ RESULTS ================")
    print(f"n_queries = {len(recalls[KS[0]])}")
    for k in KS:
        print(
            f"Recall@{k}: {mean(recalls[k]):.4f} | "
            f"MRR@{k}: {mean(mrrs[k]):.4f}"
        )


if __name__ == "__main__":
    main()
