# scripts/build_dense.py
from pathlib import Path
from rag.dense import DenseRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS = PROJECT_ROOT / "data" / "processed" / "wiki_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

dr = DenseRetriever(
    chunks_path=CHUNKS,
    index_path=INDEX_DIR / "faiss.index",
    meta_path=INDEX_DIR / "faiss_meta.json",
    embedding_dim=768,
)

dr.build_index(batch_size=32, save=True)
