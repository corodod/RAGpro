# rag/entities.py
from __future__ import annotations

from typing import List, Set
import spacy
import re


class EntityExtractor:
    """
    Retrieval-oriented entity / concept extractor for RU Wiki QA.

    Goal:
    extract referential, self-sufficient entities
    (not roles, not temporal fillers, not generic object types)
    """

    def __init__(
        self,
        model: str = "ru_core_news_lg",
        allowed_labels: Set[str] | None = None,
        min_len: int = 3,
        max_tokens: int = 6,
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
        self.max_tokens = max_tokens
        self.abbrev_re = re.compile(r"\b[А-ЯЁ]{2,6}\b")

    # --------------------------------------------------

    def _is_retrieval_entity(self, span: List[spacy.tokens.Token]) -> bool:
        """
        Structural filter: decide if span is retrieval-useful.
        """

        # too short
        if len(span) == 0 or len(span) > self.max_tokens:
            return False

        # head token
        head = span[-1]

        # dependency-based filtering
        if head.dep_ in {"obl", "obj"}:
            return False

        # single noun → only if PROPN
        if len(span) == 1:
            return head.pos_ == "PROPN"

        # verb-headed constructions → no
        if any(t.pos_ == "VERB" for t in span):
            return False

        # capitalization / properness signal
        if any(t.is_title for t in span):
            return True

        if all(t.pos_ == "PROPN" for t in span):
            return True

        # adjective + noun concepts are allowed
        if span[0].pos_ == "ADJ" and span[-1].pos_ == "NOUN":
            return True

        return False

    # --------------------------------------------------

    def extract(self, text: str) -> List[str]:
        doc = self.nlp(text)

        entities: List[str] = []
        seen = set()

        def add(val: str):
            val = val.strip()
            if len(val) < self.min_len:
                return
            key = val.lower()
            if key in seen:
                return
            seen.add(key)
            entities.append(val)

        # ==================================================
        # 1️⃣ spaCy NER (always trusted)
        # ==================================================
        for ent in doc.ents:
            if ent.label_ in self.allowed_labels:
                add(ent.text)

        # ==================================================
        # 2️⃣ POS-based noun phrases (filtered!)
        # Pattern: ADJ* + NOUN+
        # ==================================================
        tokens = list(doc)
        i = 0
        while i < len(tokens):
            start = i

            while i < len(tokens) and tokens[i].pos_ == "ADJ":
                i += 1

            if i < len(tokens) and tokens[i].pos_ == "NOUN":
                while i < len(tokens) and tokens[i].pos_ == "NOUN":
                    i += 1

                span = tokens[start:i]

                if self._is_retrieval_entity(span):
                    text_span = " ".join(t.text for t in span)
                    add(text_span)

                    lemma_span = " ".join(t.lemma_ for t in span)
                    add(lemma_span)
            else:
                i = start + 1

        # ==================================================
        # 3️⃣ Capitalized multi-word spans
        # ==================================================
        caps = re.findall(
            r"(?:[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)+)",
            text,
        )
        for c in caps:
            add(c)

        # ==================================================
        # 4️⃣ Abbreviations
        # ==================================================
        for abbr in self.abbrev_re.findall(text):
            add(abbr)

        return entities