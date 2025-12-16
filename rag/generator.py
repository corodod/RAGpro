# rag/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GeneratorConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # можно поменять на другую
    device: str = "auto"  # "cuda" / "cpu" / "auto"
    dtype: str = "auto"   # "float16" / "bfloat16" / "auto"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9


class AnswerGenerator:
    """
    Простой генератор ответа:
    - строит prompt из вопроса + выбранных чанков
    - вызывает локальную HF CausalLM (Instruct)
    """

    def __init__(self, cfg: Optional[GeneratorConfig] = None):
        self.cfg = cfg or GeneratorConfig()

        self.device = self._resolve_device(self.cfg.device)
        self.dtype = self._resolve_dtype(self.cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cuda":
            self.model.to("cuda")

        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str):
        if dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def build_context(chunks: List[Dict], max_chars: int = 5000) -> str:
        """
        Склеиваем чанки в контекст. Обрезаем по символам, чтобы не раздувать prompt.
        """
        parts = []
        total = 0
        for i, c in enumerate(chunks, start=1):
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            block = f"[{i}] {title}\n{text}".strip()

            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block) + 2

        return "\n\n".join(parts)

    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        context = self.build_context(chunks)

        # Простой “RAG-подобный” prompt: строго по контексту
        prompt = f"""Ты — ассистент, который отвечает на вопросы строго по предоставленному контексту.
Если в контексте нет ответа, скажи: "В контексте нет достаточной информации."

Контекст:
{context}

Вопрос:
{question}

Ответ:"""
        return prompt

    @torch.inference_mode()
    def generate(self, question: str, chunks: List[Dict]) -> str:
        prompt = self.build_prompt(question, chunks)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Вернём только то, что после "Ответ:"
        marker = "Ответ:"
        if marker in text:
            return text.split(marker, 1)[-1].strip()

        # fallback
        return text.strip()
