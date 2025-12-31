# scripts/eval_retrieval.py
import json
import random
from pathlib import Path
from statistics import mean
from collections import Counter
from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.retriever import Retriever, RetrieverConfig
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector


# ==================================================
# RETRIEVER POLICY
# ==================================================

RETRIEVER_CONFIG = RetrieverConfig()

# ==================================================
# EVAL CONFIG
# ==================================================

KS = [1, 3, 5, 10, 20]
DEVICE = "cuda"

MAX_QUERIES: int | None = None
QUERY_FRACTION: float | None = 0.2

SHUFFLE: bool = True
RANDOM_SEED: int = 42

# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "rubq_eval.jsonl"

REPORT_DIR = PROJECT_ROOT / "eval_reports"
REPORT_DIR.mkdir(exist_ok=True)

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


def question_len_bucket(q: str) -> str:
    n = len(q.split())
    if n <= 4:
        return "short"
    if n <= 8:
        return "medium"
    return "long"


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
    # ---------- sanity ----------
    if MAX_QUERIES is not None and QUERY_FRACTION is not None:
        raise ValueError("Use only one of MAX_QUERIES or QUERY_FRACTION")

    # ---------- load dataset ----------
    with open(EVAL_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if SHUFFLE:
        random.seed(RANDOM_SEED)
        random.shuffle(items)

    if QUERY_FRACTION is not None:
        items = items[: int(len(items) * QUERY_FRACTION)]
    elif MAX_QUERIES is not None:
        items = items[:MAX_QUERIES]

    # ---------- load retriever ----------
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

    # ---------- eval stats ----------
    recalls = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    failures = {
        "no_hit": [],
        "miss_at_1": [],
        "low_rank": [],
    }

    rank_hist = Counter()
    bucket_stats = {"short": [], "medium": [], "long": []}

    # ---------- eval loop ----------
    for item in tqdm(items, desc="Evaluating"):
        question = item["question"]
        qid = item.get("id")
        gold = set(map(str, item["gold_doc_ids"]))

        res = retriever.retrieve(question)
        pred_doc_ids = [doc_id_from_chunk_id(r["chunk_id"]) for r in res]
        # >>> LOG №1: стадия потери
        recall_docs = set(
            doc_id_from_chunk_id(cid)
            for cid in retriever.last_debug.get("candidate_ids", [])
        )

        miss_stage = None
        if not (recall_docs & gold):
            miss_stage = "recall"
        elif not (set(pred_doc_ids[:20]) & gold):
            miss_stage = "rerank"

        # metrics
        for k in KS:
            recalls[k].append(recall_at_k(pred_doc_ids, gold, k))
            mrrs[k].append(mrr_at_k(pred_doc_ids, gold, k))

        # rank analysis
        rank = None
        for i, d in enumerate(pred_doc_ids, start=1):
            if d in gold:
                rank = i
                break

        bucket = question_len_bucket(question)
        bucket_stats[bucket].append(rank is not None)

        if rank is None:
            failures["no_hit"].append({
                "id": qid,
                "question": question,
                "gold_doc_ids": list(gold),
                "miss_stage": miss_stage,
                "entities": retriever.last_debug.get("entities"),
                "entity_hit": retriever.last_debug.get("entity_hit"),
            })
        else:
            rank_hist[rank] += 1
            if rank > 1:
                failures["miss_at_1"].append(qid)
            if rank > 10:
                failures["low_rank"].append({
                    "id": qid,
                    "question": question,
                    "rank": rank,
                })

    # ---------- report ----------
    print_config()

    print("\n================ RESULTS ================")
    print(f"n_queries = {len(items)}")
    for k in KS:
        print(
            f"Recall@{k}: {mean(recalls[k]):.4f} | "
            f"MRR@{k}: {mean(mrrs[k]):.4f}"
        )

    print("\n================ FAILURE STATS ================")
    print(f"NO_HIT:     {len(failures['no_hit'])}")
    print(f"MISS@1:     {len(failures['miss_at_1'])}")
    print(f"LOW_RANK>10:{len(failures['low_rank'])}")

    print("\n================ RANK HISTOGRAM ================")
    for r in sorted(rank_hist):
        if r <= 20:
            print(f"rank {r:2d}: {rank_hist[r]}")

    print("\n================ RECALL BY QUESTION LENGTH ================")
    for b, xs in bucket_stats.items():
        if xs:
            print(f"{b:6s}: {sum(xs)/len(xs):.3f}")

    # ---------- save reports ----------
    def dump(name, data):
        path = REPORT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} → {path}")

    dump("no_hit", failures["no_hit"])
    dump("low_rank", failures["low_rank"])


if __name__ == "__main__":
    main()
