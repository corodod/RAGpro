# rag/hyde.py
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class HyDEGenerator:
    """
    HyDE: Hypothetical Document Generator
    Used ONLY to improve dense retrieval.
    """

    def __init__(
        self,
        *,
        llm_model_name: str,
        llm_device: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        trust_remote_code: bool,
    ):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
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
Ты — энциклопедический текстовый генератор.

Задача:
Напиши краткий энциклопедический абзац, который МОГ БЫ содержаться
в статье Википедии и отвечать на вопрос.

Правила:
- не используй личные местоимения
- не добавляй рассуждений
- пиши нейтрально и информативно
- допускаются неточные или обобщённые формулировки
- НЕ упоминай, что это гипотетический текст

Вопрос:
{query}

Текст:
""".strip()

    def generate(self, query: str) -> str:
        prompt = self._build_prompt(query)

        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        return out.strip()
