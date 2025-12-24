# rag/generator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


# ================= CONFIG =================
@dataclass
class GeneratorConfig:
    backend: str  # "cuda" | "cpu"
    max_new_tokens: int = 120
    temperature: float = 0.7
    repetition_penalty: float = 1.12


# ================= GENERATOR =================
class AnswerGenerator:
    def __init__(self, cfg: Optional[GeneratorConfig] = None):
        self.cfg = cfg or GeneratorConfig(backend="cpu")

        # -------- CUDA → QWEN --------
        if self.cfg.backend == "cuda" and torch.cuda.is_available():
            self.scenario = "qwen"
            self.device = torch.device("cuda")
            self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            self.dtype = torch.float16

        # -------- CPU / MAC → TINYLLAMA --------
        else:
            self.scenario = "tinyllama"
            self.device = torch.device("cpu")
            self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.dtype = torch.float32

            # 🔴 КРИТИЧНО ДЛЯ MAC (segfault fix)
            torch.set_num_threads(1)

        print(
            f"[Generator] scenario={self.scenario} "
            f"model={self.model_name} device={self.device}"
        )

        # -------- LOAD TOKENIZER --------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # -------- LOAD MODEL --------
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        # -------- PIPELINE ТОЛЬКО ДЛЯ TINYLLAMA --------
        if self.scenario == "tinyllama":
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU only
            )
        else:
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ================= CONTEXT =================
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

    # ================= PUBLIC API =================
    def generate(self, question: str, chunks: List[Dict]) -> str:
        if self.scenario == "tinyllama":
            return self._generate_tinyllama(question, chunks)
        else:
            return self._generate_qwen(question, chunks)

    # ================= TINYLLAMA =================
    def _generate_tinyllama(self, question: str, chunks: List[Dict]) -> str:
        context = self.build_context(chunks)

        prompt = (
            "Answer strictly using the facts from the context below.\n"
            "If the answer is not present, say:\n"
            "Not enough information in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        ).strip()

        out = self.pipe(
            prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        return out.strip()

    # ================= QWEN =================
    def _generate_qwen(self, question: str, chunks: List[Dict]) -> str:
        context = self.build_context(chunks)
        # "НЕ добавляй объяснений.\n"
        system = (
            "Ты — система извлечения фактов.\n"
            "Отвечай ТОЛЬКО фактами из контекста.\n"
            "Если ответа нет — напиши ровно:\n"
            "В контексте нет достаточной информации."
        )

        user = (
            f"Контекст:\n{context}\n\n"
            f"Вопрос:\n{question}\n\n"
            "Ответ (2–3 предложения):"
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
