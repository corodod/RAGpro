# rag/hyde.py
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# =========================
# HYPERPARAMETERS
# =========================

HYDE_LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HYDE_LLM_DEVICE = "cpu"

HYDE_MAX_NEW_TOKENS = 120
HYDE_DO_SAMPLE = False
HYDE_TEMPERATURE = 0.0
HYDE_TRUST_REMOTE_CODE = True

class HyDEGenerator:
    """
    HyDE: Hypothetical Document Generator

    Generates a single document-like passage that represents
    a hypothetical relevant document for the query.

    Used ONLY to improve dense retrieval.
    """

    def __init__(
        self,
        *,
        llm_model_name: str = HYDE_LLM_MODEL_NAME,
        llm_device: str = HYDE_LLM_DEVICE,
        max_new_tokens: int = HYDE_MAX_NEW_TOKENS,
    ):
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=HYDE_TRUST_REMOTE_CODE,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            trust_remote_code=HYDE_TRUST_REMOTE_CODE,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if llm_device == "cpu" else 0,
        )

    # --------------------------------------------------

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

    # --------------------------------------------------

    def generate(self, query: str) -> str:
        prompt = self._build_prompt(query)

        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=HYDE_DO_SAMPLE,
            temperature=HYDE_TEMPERATURE,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        return out.strip()
