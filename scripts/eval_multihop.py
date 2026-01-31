# scripts/eval_multihop.py
import json
import random
from pathlib import Path
from statistics import mean
from collections import Counter
from tqdm import tqdm

from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.reranker import CrossEncoderReranker
from rag.retriever import RetrieverConfig, Retriever
from rag.rewrite import QueryRewriter
from rag.entities import EntityExtractor
from rag.coverage import CoverageSelector
from rag.multihop import MultiHopRetriever
from rag.generator import AnswerGenerator, GeneratorConfig

# ==================================================
# CONFIG
# ==================================================
USE_MULTIHOP = True
MAX_HOPS = 3
DEVICE = "cuda"
KS = [1, 3, 5, 10, 20]
QUERY_FRACTION: float | None = None
MAX_QUERIES: int | None = None
SHUFFLE: bool = True
RANDOM_SEED: int = 42

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "multihop_questions.jsonl"

REPORT_DIR = PROJECT_ROOT / "eval_reports"
REPORT_DIR.mkdir(exist_ok=True)

# ==================================================
# METRICS
# ==================================================
def doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]

# хотя бы один золотой документ найден
def recall_at_k_loose(pred_doc_ids, gold_doc_ids, k):
    return 1.0 if any(d in gold_doc_ids for d in pred_doc_ids[:k]) else 0.0

# все золотые документы найдены
def recall_at_k_strict(pred_doc_ids, gold_doc_ids, k):
    return 1.0 if all(d in pred_doc_ids[:k] for d in gold_doc_ids) else 0.0

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

def dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


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

    print(f"[INFO] Evaluating {len(items)} questions")

    # ---------- load retrievers ----------
    bm25 = BM25Retriever.load(INDEX_DIR)

    dense = DenseRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_DIR / "faiss.index",
        meta_path=INDEX_DIR / "faiss_meta.json",
    )
    dense.model.to(DEVICE)
    dense.load()

    reranker = CrossEncoderReranker(device=DEVICE)
    rewriter = QueryRewriter(llm_device=DEVICE)
    entity_extractor = EntityExtractor()
    coverage_selector = CoverageSelector()
    generator = AnswerGenerator(
        GeneratorConfig(backend="cuda" if DEVICE == "cuda" else "cpu")
    )

    base_retriever = Retriever(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        rewriter=rewriter,
        entity_extractor=entity_extractor,
        coverage_selector=coverage_selector,
        config=RetrieverConfig(),
        debug=False,
    )

    if USE_MULTIHOP:
        retriever = MultiHopRetriever(
            base_retriever=base_retriever,
            generator=generator,
            max_hops=MAX_HOPS,
            debug=True,
            eval_retrieval=True,
        )
    else:
        retriever = base_retriever

    # ---------- eval stats ----------
    recalls_loose = {k: [] for k in KS}
    recalls_strict = {k: [] for k in KS}
    mrrs = {k: [] for k in KS}

    rank_hist = Counter()
    bucket_stats = {"short": [], "medium": [], "long": []}
    failures = {"no_hit": [], "low_rank": []}

    # ---------- eval loop ----------
    for item in tqdm(items, desc="Evaluating"):
        question = item["question"]
        qid = item.get("qid")
        gold = set(map(str, item["gold_doc_ids"]))

        res = retriever.retrieve(question)
        pred_doc_ids = [doc_id_from_chunk_id(r["chunk_id"]) for r in res]
        pred_doc_ids = dedup_keep_order(pred_doc_ids)

        for k in KS:
            recalls_loose[k].append(recall_at_k_loose(pred_doc_ids, gold, k))
            recalls_strict[k].append(recall_at_k_strict(pred_doc_ids, gold, k))
            mrrs[k].append(mrr_at_k(pred_doc_ids, gold, k))

        # rank analysis (first gold)
        rank = None
        for i, d in enumerate(pred_doc_ids, start=1):
            if d in gold:
                rank = i
                break

        bucket = question_len_bucket(question)
        bucket_stats[bucket].append(rank is not None)

        if rank is None:
            failures["no_hit"].append({
                "qid": qid,
                "question": question,
                "gold_doc_ids": list(gold),
            })
        else:
            rank_hist[rank] += 1
            if rank > 10:
                failures["low_rank"].append({
                    "qid": qid,
                    "question": question,
                    "rank": rank,
                })

    # ---------- report ----------
    print(f"\nEvaluated {len(items)} questions\n")

    for k in KS:
        print(
            f"@{k:2d} | "
            f"Recall(loose): {mean(recalls_loose[k]):.4f} | "
            f"Recall(strict): {mean(recalls_strict[k]):.4f} | "
            f"MRR: {mean(mrrs[k]):.4f}"
        )

    print("\nRank histogram (top 20):")
    for r in sorted(rank_hist):
        if r <= 20:
            print(f"rank {r:2d}: {rank_hist[r]}")

    print("\nRecall by question length:")
    for b, xs in bucket_stats.items():
        if xs:
            print(f"{b:6s}: {sum(xs)/len(xs):.3f}")

    # ---------- save failures ----------
    def dump(name, data):
        path = REPORT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} → {path}")

    dump("no_hit_multihop", failures["no_hit"])
    dump("low_rank_multihop", failures["low_rank"])


if __name__ == "__main__":
    main()
