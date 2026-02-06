# rag/entities.py
from __future__ import annotations

from typing import List, Set
import re
import spacy


class EntityExtractor:
    """
    Retrieval-oriented entity / concept extractor for RU Wiki QA.
    """

    def __init__(
        self,
        *,
        model: str,
        allowed_labels: Set[str],
        min_len: int,
        max_tokens: int,
    ):
        self.nlp = spacy.load(model)
        self.allowed_labels = set(allowed_labels)
        self.min_len = min_len
        self.max_tokens = max_tokens
        self.abbrev_re = re.compile(r"\b[А-ЯЁ]{2,6}\b")

    def _is_retrieval_entity(self, span: List[spacy.tokens.Token]) -> bool:
        # too short / too long
        if len(span) == 0 or len(span) > self.max_tokens:
            return False

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

        # 1) spaCy NER
        for ent in doc.ents:
            if ent.label_ in self.allowed_labels:
                add(ent.text)

        # 2) POS-based noun phrases: ADJ* + NOUN+
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

        # 3) Capitalized multi-word spans
        caps = re.findall(r"(?:[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)+)", text)
        for c in caps:
            add(c)

        # 4) Abbreviations
        for abbr in self.abbrev_re.findall(text):
            add(abbr)

        return entities