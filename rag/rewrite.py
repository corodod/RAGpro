# rag/rewrite.py
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# HYPERPARAMETERS
# =========================

# (оставляем только "технические" значения по умолчанию,
#  но всё важное будет задаваться из retriever через конструктор)

REWRITE_MIN_LINE_LEN = 10
REWRITE_MAX_NEW_TOKENS = 96
REWRITE_DO_SAMPLE = False
REWRITE_TEMPERATURE = 0.0


# ============================================================
# Helpers
# ============================================================

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ============================================================
# QueryRewriter (LLM-only + cosine filter)
# ============================================================

class QueryRewriter:
    """
    Stateless query rewriting module.

    Everything is configured from outside (RetrieverConfig).
    """

    def __init__(
        self,
        *,
        llm_model_name: str,
        embedder_model_name: str,
        llm_device: str = "cpu",  # "cpu" | "cuda"
        max_new_tokens: int = REWRITE_MAX_NEW_TOKENS,
        do_sample: bool = REWRITE_DO_SAMPLE,
        temperature: float = REWRITE_TEMPERATURE,
        min_line_len: int = REWRITE_MIN_LINE_LEN,
        n_rewrites: int = 2,
        min_cosine: float = 0.75,
    ):
        self.n_rewrites = n_rewrites
        self.min_cosine = min_cosine

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.min_line_len = min_line_len

        # embedder for cosine filter
        self.embedder = SentenceTransformer(embedder_model_name)

        # LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if llm_device == "cpu" else 0,
        )

    # --------------------------------------------------------

    def _build_prompt(self, query: str, *, n_rewrites: int) -> str:
        return f"""
Ты — модуль переформулировки ПОИСКОВЫХ запросов для Википедии.

Исходный вопрос:
{query}

Задача:
Переформулируй вопрос {n_rewrites} способами, чтобы его было легче найти в Википедии.

Правила:
- не отвечай на вопрос
- не добавляй факты
- не расширяй смысл
- используй нейтральную энциклопедическую лексику
- выведи РОВНО {n_rewrites} строк
- без нумерации
- без кавычек
- без комментариев
""".strip()

    def _parse_output(self, text: str) -> List[str]:
        lines = []
        for l in text.splitlines():
            l = normalize_text(l)
            l = re.sub(r"^[\-\*\d\.\)\s]+", "", l)
            if len(l) < self.min_line_len:
                continue
            if l.startswith(("первый вариант", "второй вариант", "вариант")):
                continue
            lines.append(l)
        return dedup_keep_order(lines)

    def _filter_by_similarity(self, query: str, rewrites: List[str]) -> List[str]:
        if not rewrites:
            return []

        texts = [f"query: {query}"] + [f"query: {r}" for r in rewrites]
        embs = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        qv = embs[0]
        out = []
        for r, rv in zip(rewrites, embs[1:]):
            if float(qv @ rv) >= self.min_cosine:
                out.append(r)
        return out

    # --------------------------------------------------------

    def rewrite(self, query: str) -> List[str]:
        query_n = normalize_text(query)
        if not query_n:
            return []

        prompt = self._build_prompt(query_n, n_rewrites=self.n_rewrites)

        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        rewrites = self._parse_output(out)
        rewrites = [r for r in rewrites if r != query_n]
        rewrites = self._filter_by_similarity(query_n, rewrites)
        return rewrites[: self.n_rewrites]