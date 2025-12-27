# rag/entities.py
from __future__ import annotations

from typing import List, Set
import spacy
import re


class EntityExtractor:
    """
    Robust entity extractor for retrieval expansion.

    Strategy:
    1) spaCy NER (high precision)
    2) Fallback: capitalized noun phrases (high recall)

    This is retrieval-oriented, not pure NER.
    """

    def __init__(
        self,
        model: str = "ru_core_news_lg",
        allowed_labels: Set[str] | None = None,
        min_len: int = 3,
    ):
        self.nlp = spacy.load(model)

        self.allowed_labels = allowed_labels or {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "EVENT",
            "WORK_OF_ART",
        }

        self.min_len = min_len

    def extract(self, text: str) -> List[str]:
        doc = self.nlp(text)

        entities: List[str] = []
        seen = set()

        # --------------------------------------------------
        # 1️⃣ Primary: spaCy NER
        # --------------------------------------------------
        for ent in doc.ents:
            if ent.label_ not in self.allowed_labels:
                continue

            val = ent.text.strip()
            if len(val) < self.min_len:
                continue

            key = val.lower()
            if key in seen:
                continue

            seen.add(key)
            entities.append(val)

        if entities:
            return entities

        # --------------------------------------------------
        # 2️⃣ Fallback: capitalized spans (RU-friendly)
        # --------------------------------------------------
        text_norm = text.strip()

        # ищем последовательности слов с заглавных букв
        candidates = re.findall(
            r"(?:[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)+)",
            text_norm,
        )

        for c in candidates:
            c = c.strip()
            if len(c) < self.min_len:
                continue

            key = c.lower()
            if key in seen:
                continue

            seen.add(key)
            entities.append(c)

        return entities