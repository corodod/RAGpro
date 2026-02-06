# rag/query2doc.py
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Query2DocGenerator:
    """
    Query2Doc: Pseudo-document generator for query expansion.
    """

    def __init__(
        self,
        *,
        llm_model_name: str,
        llm_device: str,
        max_new_tokens: int,
        temperature: float,
        trust_remote_code: bool,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            trust_remote_code=trust_remote_code,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if llm_device == "cpu" else 0,
        )

    def _build_prompt(self, query: str) -> str:
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

    def generate(self, query: str) -> str:
        prompt = self._build_prompt(query)

        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=self.temperature,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        return out.strip()
