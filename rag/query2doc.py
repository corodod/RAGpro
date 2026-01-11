# rag/query2doc.py
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Query2DocGenerator:
    """
    Query2Doc: Pseudo-document generator for query expansion.

    Generates a single encyclopedic-style passage that expands
    the original query with relevant background terms and context.

    Intended usage:
      - BM25(query + pseudo_doc)
      - Dense(query [SEP] pseudo_doc)

    Important:
      - NOT a hypothetical document like HyDE
      - Acts as query expansion, not query replacement
    """

    def __init__(
        self,
        *,
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_device: str = "cpu",
        max_new_tokens: int = 128,
    ):
        self.max_new_tokens = max_new_tokens

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

    # --------------------------------------------------

    def _build_prompt(self, query: str) -> str:
        """
        Prompt is intentionally DIFFERENT from HyDE.

        Goal:
          - enrich query with typical terminology
          - reduce lexical gap
          - stay encyclopedic and neutral
        """
        return f"""
Ты — модуль расширения ПОИСКОВОГО запроса для энциклопедического поиска.

Исходный запрос:
{query}

Задача:
Напиши ОДИН краткий энциклопедический абзац, который:
- раскрывает контекст запроса
- использует типичные термины, связанные с темой
- помогает найти релевантные статьи

Правила:
- не отвечай напрямую на вопрос
- не добавляй новые факты, если они не являются общеизвестными
- допускаются обобщённые формулировки
- не используй личные местоимения
- нейтральный, энциклопедический стиль
- без списков и без заголовков
- один связный абзац текста

Текст:
""".strip()

    # --------------------------------------------------

    def generate(self, query: str) -> str:
        """
        Generate pseudo-document for query expansion.
        """
        prompt = self._build_prompt(query)

        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        return out.strip()
