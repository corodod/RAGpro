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

# =========================
# HYPERPARAMETERS
# =========================

# backends / scenarios

# model names
GEN_MODEL_MAX = "Qwen/Qwen2.5-3B-Instruct"
GEN_MODEL_MIN = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# default generation params
GEN_MAX_NEW_TOKENS = 120
GEN_TEMPERATURE = 0.7
GEN_REPETITION_PENALTY = 1.12

# tokenizer/model input limits
GEN_MAX_LENGTH = 4096  # tokenizer truncation max_length

# context building
GEN_CONTEXT_MAX_CHARS_DEFAULT = 4000

# tinyllama pipeline defaults
GEN_PIPELINE_TASK = "text-generation"
GEN_PIPELINE_DEVICE_CPU = -1

# Mac/CPU stability
GEN_CPU_NUM_THREADS = 1  # segfault fix on some mac setups
# ================= CONFIG =================
@dataclass
class GeneratorConfig:
    backend: str  # "cuda" | "cpu"
    max_new_tokens: int = GEN_MAX_NEW_TOKENS
    temperature: float = GEN_TEMPERATURE
    repetition_penalty: float = GEN_REPETITION_PENALTY


# ================= GENERATOR =================
class AnswerGenerator:
    def __init__(self, cfg: Optional[GeneratorConfig] = None, model_name: Optional[str] = None):
        self.cfg = cfg or GeneratorConfig(backend="cpu")

        # -------- CUDA → QWEN --------
        if self.cfg.backend == "cuda" and torch.cuda.is_available():
            self.scenario = "qwen"
            self.device = torch.device("cuda")
            self.model_name = model_name or "GEN_MODEL_MAX"
            self.dtype = torch.float16

        # -------- CPU / MAC → TINYLLAMA --------
        else:
            self.scenario = "tinyllama"
            self.device = torch.device("cpu")
            self.model_name = model_name or "GEN_MODEL_MIN"
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
                GEN_PIPELINE_TASK,
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
    def build_context(chunks: List[Dict], max_chars: int = GEN_CONTEXT_MAX_CHARS_DEFAULT) -> str:
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
            max_length=GEN_MAX_LENGTH,
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

    def generate_chat(self, system: str, user: str, *, max_new_tokens: int | None = None) -> str:
        max_new = max_new_tokens or self.cfg.max_new_tokens

        if self.scenario == "tinyllama":
            # Tinyllama не chat-template → просто склеим
            prompt = f"{system}\n\n{user}\n\nОтвет:"
            out = self.pipe(
                prompt,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=GEN_TINYLLAMA_TEMP_FOR_CHAT,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"]
            return out.strip()

        # Qwen chat-template
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
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
            max_length=GEN_MAX_LENGTH,
        ).to(self.device)

        with torch.inference_mode():
            do_sample = bool(self.cfg.temperature and self.cfg.temperature > GEN_TINYLLAMA_TEMP_FOR_CHAT)

            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=self.cfg.temperature if do_sample else None,
                repetition_penalty=self.cfg.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = output[0][inputs["input_ids"].shape[-1]:]
        txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return txt.strip()
