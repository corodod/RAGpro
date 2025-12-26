# rag/rewrite.py
from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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

    Responsibilities:
      - generate paraphrases via LLM
      - filter semantic drift via cosine similarity
      - return clean rewrite list

    No cache. No lexicons. No heuristics.
    """

    def __init__(
        self,
        *,
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_device: str = "cpu",
        n_rewrites: int = 2,
        min_cosine: float = 0.75,
    ):
        self.n_rewrites = n_rewrites
        self.min_cosine = min_cosine

        # embedder for cosine filter
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-small")

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

    def _build_prompt(self, query: str) -> str:
        return f"""
Ты — модуль переформулировки ПОИСКОВЫХ запросов для Википедии.

Исходный вопрос:
{query}

Задача:
Переформулируй вопрос ДВУМЯ способами, чтобы его было легче найти в Википедии.

Правила:
- не отвечай на вопрос
- не добавляй факты
- не расширяй смысл
- используй нейтральную энциклопедическую лексику
- выведи РОВНО ДВЕ строки
- без нумерации
- без кавычек
- без комментариев
""".strip()

    def _parse_output(self, text: str) -> List[str]:
        lines = []
        for l in text.splitlines():
            l = normalize_text(l)
            l = re.sub(r"^[\-\*\d\.\)\s]+", "", l)
            if len(l) < 10:
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

        prompt = self._build_prompt(query_n)

        out = self.pipe(
            prompt,
            max_new_tokens=96,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        rewrites = self._parse_output(out)
        rewrites = [r for r in rewrites if r != query_n]
        rewrites = self._filter_by_similarity(query_n, rewrites)
        return rewrites[: self.n_rewrites]

