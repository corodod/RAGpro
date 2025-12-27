# rag/entities.py
from __future__ import annotations

from typing import List, Set
import spacy


class EntityExtractor:
    """
    High-quality entity extractor based on spaCy NER.

    Extracts:
      - named entities (PERSON, ORG, GPE, EVENT, etc.)
      - returns clean surface forms
      - suitable for recall-expansion queries

    This is NOT a toy extractor.
    """

    def __init__(
        self,
        model: str = "ru_core_news_lg",
        allowed_labels: Set[str] | None = None,
        min_len: int = 3,
    ):
        self.nlp = spacy.load(model)

        # по умолчанию — всё полезное для retrieval
        self.allowed_labels = allowed_labels or {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "EVENT",
            "WORK_OF_ART",
            "DATE",
        }

        self.min_len = min_len

    def extract(self, text: str) -> List[str]:
        """
        Extract entities from question.

        Returns:
          List[str]: unique entity strings, order preserved
        """
        doc = self.nlp(text)

        seen = set()
        entities: List[str] = []

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

        return entities