# rag/llm_bundle.py
from __future__ import annotations
from dataclasses import dataclass

from rag.generator import AnswerGenerator


@dataclass
class LLMBundle:
    # new roles
    decomposer: AnswerGenerator
    compiler: AnswerGenerator

    # existing roles
    planner: AnswerGenerator
    extractor: AnswerGenerator
    synthesizer: AnswerGenerator

    @staticmethod
    def from_single(gen: AnswerGenerator) -> "LLMBundle":
        # Один и тот же генератор на все роли (одна модель в VRAM)
        return LLMBundle(
            decomposer=gen,
            compiler=gen,
            planner=gen,      # legacy (можно убрать позже)
            extractor=gen,
            synthesizer=gen,
        )