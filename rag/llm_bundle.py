# rag/llm_bundle.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from rag.generator import AnswerGenerator


@dataclass
class LLMBundle:
    planner: AnswerGenerator
    extractor: AnswerGenerator
    synthesizer: AnswerGenerator

    @staticmethod
    def from_single(gen: AnswerGenerator) -> "LLMBundle":
     # Один и тот же генератор на все роли (одна модель в VRAM)
        return LLMBundle(planner=gen, extractor=gen, synthesizer=gen)