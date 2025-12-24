# rag/generator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GeneratorConfig:
    backend: str  # "cuda" | "mps" | "cpu"
    max_new_tokens: int = 80
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.12


class AnswerGenerator:
    def __init__(self, cfg: Optional[GeneratorConfig] = None):
        self.cfg = cfg or GeneratorConfig(backend="cpu")

        # ---------- backend & model selection ----------
        if self.cfg.backend == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            self.dtype = torch.float16

        else:
            # Mac (MPS) или обычный CPU
            if torch.backends.mps.is_available() and self.cfg.backend == "mps":
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self.dtype = torch.float32

        print(
            f"[Generator] backend={self.cfg.backend} "
            f"model={self.model_name} device={self.device}"
        )

        # ---------- tokenizer ----------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )

        # ---------- model ----------
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ---------- context ----------
    @staticmethod
    def build_context(chunks: List[Dict], max_chars: int = 4000) -> str:
        parts, total = [], 0
        for i, c in enumerate(chunks, start=1):
            text = (c.get("text") or "").strip()
            block = f"[{i}] {text}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n".join(parts)

    # ---------- generation ----------
    def generate(self, question: str, chunks: List[Dict]) -> str:
        context = self.build_context(chunks)

        system = (
            "Ты — система извлечения фактов.\n"
            "Отвечай ТОЛЬКО фактами из контекста.\n"
            "НЕ добавляй объяснений.\n"
            "Если ответа нет — напиши ровно:\n"
            "В контексте нет достаточной информации."
        )

        user = (
            f"Контекст:\n{context}\n\n"
            f"Вопрос:\n{question}\n\n"
            "Ответ (1–2 предложения):"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.cfg.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = output[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return answer.strip().split("\n\n")[0]
