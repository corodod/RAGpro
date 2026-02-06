# rag/decomp_schema.py
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class DecompItem(BaseModel):
    id: str = Field(..., description="Question id, e.g. Q1")
    text: str = Field(..., description="Sub-question text, may contain {slot}")
    deps: List[str] = Field(default_factory=list, description="Dependencies: e.g. ['Q1']")
    slot: Optional[str] = Field(default=None, description="Slot name produced by this question")

    @model_validator(mode="after")
    def _clean(self) -> "DecompItem":
        self.id = self.id.strip()
        self.text = (self.text or "").strip()
        self.deps = [d.strip() for d in (self.deps or []) if d and d.strip()]
        self.slot = self.slot.strip() if isinstance(self.slot, str) and self.slot.strip() else None
        return self


class DecompGraph(BaseModel):
    items: List[DecompItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_nonempty(self) -> "DecompGraph":
        if not self.items:
            raise ValueError("DecompGraph.items must not be empty")
        return self
