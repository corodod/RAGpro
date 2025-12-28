# scripts/eval_retrieval.py
import json
import random
from pathlib import Path
from statistics import mean
from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.retriever import Retriever, RetrieverConfig
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector


# ==================================================
# RETRIEVER POLICY (single source of truth)
# ==================================================

RETRIEVER_CONFIG = RetrieverConfig()

# ==================================================
# EVAL CONFIG (experiment protocol)
# ==================================================
KS = [1, 3, 5, 10, 20]
DEVICE = "cuda"          # "cuda" or "cpu"

MAX_QUERIES: int | None = None      # e.g. 1000
QUERY_FRACTION: float | None = 0.2  # e.g. 0.2 (20%)

SHUFFLE: bool = False
RANDOM_SEED: int = 42

# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"


# ==================================================
# METRICS
# ==================================================

def doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]


def recall_at_k(pred_doc_ids, gold_doc_ids, k):
    return 1.0 if any(d in gold_doc_ids for d in pred_doc_ids[:k]) else 0.0


def mrr_at_k(pred_doc_ids, gold_doc_ids, k):
    for i, d in enumerate(pred_doc_ids[:k], start=1):
        if d in gold_doc_ids:
            return 1.0 / i
    return 0.0

def print_config():
    print("\n================ CONFIG =================")

    print("\n[RetrieverConfig]")
    for k, v in vars(RETRIEVER_CONFIG).items():
        print(f"{k:25s}: {v}")

    print("\n[EvalConfig]")
    print(f"{'DEVICE':25s}: {DEVICE}")
    print(f"{'MAX_QUERIES':25s}: {MAX_QUERIES}")
    print(f"{'QUERY_FRACTION':25s}: {QUERY_FRACTION}")
    print(f"{'SHUFFLE':25s}: {SHUFFLE}")
    print(f"{'RANDOM_SEED':25s}: {RANDOM_SEED}")

    print("\n[Metrics]")
    print(f"{'KS':25s}: {KS}")

    print("=" * 40)

# ==================================================
# MAIN
# ==================================================

def main():
    # ---------- Sanity checks ----------
    if MAX_QUERIES is not None and QUERY_FRACTION is not None:
        raise ValueError(
            "Use only one of MAX_QUERIES or QUERY_FRACTION, not both."
        )

    # ---------- Load dataset ----------
    with open(EVAL_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if SHUFFLE:
        random.seed(RANDOM_SEED)
        random.shuffle(items)

    if QUERY_FRACTION is not None:
        n = int(len(items) * QUERY_FRACTION)
        items = items[:n]
    elif MAX_QUERIES is not None:
        items = items[:MAX_QUERIES]

    # ---------- Load retrievers ----------
    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
    )
    dense.model.to(DEVICE)
    dense.load()

    reranker = (
        CrossEncoderReranker(device=DEVICE)
        if RETRIEVER_CONFIG.use_cross_encoder
        else None
    )

    rewriter = (
        QueryRewriter(llm_device=DEVICE)
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

    retriever = Retriever(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        rewriter=rewriter,
        entity_extractor=entity_extractor,
        coverage_selector=coverage_selector,
        config=RETRIEVER_CONFIG,
        debug=False,
    )

    # ---------- Eval loop ----------
    recalls = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    for item in tqdm(items, desc="Evaluating"):
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
    print_config()

    print("\n================ RESULTS ================")
    print(f"n_queries = {len(items)}")
    for k in KS:
        print(
            f"Recall@{k}: {mean(recalls[k]):.4f} | "
            f"MRR@{k}: {mean(mrrs[k]):.4f}"
        )


if __name__ == "__main__":
    main()
