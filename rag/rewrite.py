# # rag/rewrite.py
# from __future__ import annotations
#
# import re
# import sqlite3
# import time
# from pathlib import Path
# from typing import Iterable, List, Optional
#
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#
#
# # ============================================================
# # Helpers
# # ============================================================
#
# _RU_STOPWORDS = {
#     "и", "а", "но", "или", "либо", "что", "это", "как", "какой", "какая", "какие",
#     "кто", "кого", "кому", "чем", "чего", "где", "когда", "почему", "зачем",
#     "в", "во", "на", "по", "для", "из", "у", "о", "об", "от", "до", "при",
#     "же", "бы", "не", "ни", "да", "нет", "там", "тут", "этот", "эта", "эти",
#     "тот", "та", "те", "его", "ее", "их", "он", "она", "они", "мы", "вы", "я",
# }
#
# def normalize_text(s: str) -> str:
#     s = (s or "").strip().lower()
#     s = s.replace("ё", "е")
#     s = re.sub(r"\s+", " ", s)
#     return s
#
# def tokenize_ru(s: str) -> List[str]:
#     return re.findall(r"[а-яa-z0-9]+", normalize_text(s))
#
# def dedup_keep_order(items: Iterable[str]) -> List[str]:
#     seen = set()
#     out = []
#     for x in items:
#         x = x.strip()
#         if not x or x in seen:
#             continue
#         seen.add(x)
#         out.append(x)
#     return out
#
#
# # ============================================================
# # SQLite cache
# # ============================================================
#
# class SQLiteRewriteCache:
#     def __init__(self, path: Path):
#         self.path = path
#         self.path.parent.mkdir(parents=True, exist_ok=True)
#         self._init()
#
#     def _init(self):
#         with sqlite3.connect(self.path) as con:
#             con.execute(
#                 """
#                 CREATE TABLE IF NOT EXISTS rewrites (
#                     query TEXT PRIMARY KEY,
#                     rewrites TEXT NOT NULL,
#                     created_at INTEGER NOT NULL
#                 )
#                 """
#             )
#             con.commit()
#
#     def get(self, query: str) -> Optional[List[str]]:
#         q = normalize_text(query)
#         with sqlite3.connect(self.path) as con:
#             row = con.execute(
#                 "SELECT rewrites FROM rewrites WHERE query = ?", (q,)
#             ).fetchone()
#         if not row:
#             return None
#         return row[0].split("\n")
#
#     def set(self, query: str, rewrites: List[str]):
#         q = normalize_text(query)
#         payload = "\n".join(rewrites)
#         with sqlite3.connect(self.path) as con:
#             con.execute(
#                 "INSERT OR REPLACE INTO rewrites VALUES (?,?,?)",
#                 (q, payload, int(time.time())),
#             )
#             con.commit()
#
#
# # ============================================================
# # RuWordNet adapter
# # ============================================================
#
# class RuWordNetAdapter:
#     def __init__(self):
#         self.enabled = False
#         self.wn = None
#         try:
#             import ruwordnet
#             self.wn = ruwordnet.RuWordNet()
#             self.enabled = True
#             print("[Rewrite] RuWordNet enabled")
#         except Exception:
#             print("[Rewrite] RuWordNet NOT available (ok)")
#
#     def synonyms(self, lemma: str, limit: int = 3) -> List[str]:
#         if not self.enabled:
#             return []
#         out = []
#         for synset in self.wn.get_synsets(lemma):
#             for w in synset.get_words():
#                 out.append(w)
#                 if len(out) >= limit:
#                     break
#         return dedup_keep_order(out)
#
#
# # ============================================================
# # QueryRewriter (Qwen 2.5)
# # ============================================================
#
#
#
# class QueryRewriter:
#     """
#         Production-grade query rewriting for retrieval.
#         Uses:
#           - RuWordNet (optional)
#           - Qwen2.5-1.5B-Instruct for paraphrasing
#           - plain-text output (NO JSON)
#           - SQLite cache
#     """
#
#     def __init__(
#         self,
#         *,
#         llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
#         llm_device: str = "cpu",
#         cache_path: Path,
#         n_rewrites: int = 2,
#         max_terms: int = 20,
#         min_cosine: float = 0.7,
#     ):
#         self.n_rewrites = n_rewrites
#         self.max_terms = max_terms
#         self.min_cosine = min_cosine
#
#         self.cache = SQLiteRewriteCache(cache_path)
#         self.ruwn = RuWordNetAdapter()
#
#         self.embedder = SentenceTransformer("intfloat/multilingual-e5-small")
#
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             llm_model_name,
#             trust_remote_code=True,
#         )
#         self.model = AutoModelForCausalLM.from_pretrained(
#             llm_model_name,
#             trust_remote_code=True,
#         )
#
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device=-1 if llm_device == "cpu" else 0,
#         )
#
#     # --------------------------------------------------------
#
#     def _extract_terms(self, query: str) -> List[str]:
#         tokens = tokenize_ru(query)
#         tokens = [t for t in tokens if t not in _RU_STOPWORDS and len(t) >= 3]
#         return tokens[: self.max_terms]
#
#     def _expand_terms(self, terms: List[str]) -> List[str]:
#         expanded = []
#         for t in terms[:8]:
#             expanded.extend(self.ruwn.synonyms(t))
#         return dedup_keep_order(expanded)
#
#     def _build_prompt(self, query: str, terms: List[str]) -> str:
#         return f"""
# Ты — модуль переформулировки ПОИСКОВЫХ запросов для Википедии.
#
# Исходный вопрос:
# {query}
#
# Ключевые слова и синонимы:
# {", ".join(terms)}
#
# Задача:
# Переформулируй вопрос ДВУМЯ способами, чтобы его было легче найти в Википедии.
#
# Правила:
# - не отвечай на вопрос
# - не добавляй факты
# - используй нейтральную энциклопедическую лексику
# - выведи РОВНО ДВЕ строки
# - без нумерации
# - без кавычек
# - без комментариев
# """.strip()
#
#     def _parse_output(self, text: str) -> List[str]:
#         lines = []
#         for l in text.splitlines():
#             l = normalize_text(l)
#             l = re.sub(r"^[\-\*\d\.\)\s]+", "", l)
#             if len(l) < 10:
#                 continue
#             if l.startswith(("первый вариант", "второй вариант", "вариант")):
#                 continue
#             lines.append(l)
#         return dedup_keep_order(lines)
#
#     def _filter_by_similarity(self, query: str, rewrites: List[str]) -> List[str]:
#         if not rewrites:
#             return []
#
#         texts = [f"query: {query}"] + [f"query: {r}" for r in rewrites]
#         embs = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")
#         embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
#
#         qv = embs[0]
#         out = []
#         for r, rv in zip(rewrites, embs[1:]):
#             if float(qv @ rv) >= self.min_cosine:
#                 out.append(r)
#         return out
#
#     # --------------------------------------------------------
#
#     def rewrite(self, query: str) -> List[str]:
#         query_n = normalize_text(query)
#         if not query_n:
#             return []
#
#         cached = self.cache.get(query_n)
#         if cached is not None:
#             return cached
#
#         base_terms = self._extract_terms(query_n)
#         expanded_terms = self._expand_terms(base_terms)
#         all_terms = dedup_keep_order(base_terms + expanded_terms)
#
#         prompt = self._build_prompt(query_n, all_terms)
#
#         out = self.pipe(
#             prompt,
#             max_new_tokens=96,
#             do_sample=False,
#             temperature=0.0,
#             return_full_text=False,
#             pad_token_id=self.tokenizer.eos_token_id,
#         )[0]["generated_text"]
#
#         rewrites = self._parse_output(out)
#         rewrites = [r for r in rewrites if r != query_n]
#         rewrites = self._filter_by_similarity(query_n, rewrites)
#         rewrites = rewrites[: self.n_rewrites]
#
#         self.cache.set(query_n, rewrites)
#         return rewrites