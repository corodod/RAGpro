# rag/plan_schema.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


OpName = Literal[
    "retrieve",
    "extract_entities",
    "map_retrieve",
    "filter_ce",
    "extract_answer",
    "compose_query",
    "union_hits",
    "intersect_hits",
    "synthesize",
]


class Step(BaseModel):
    id: str = Field(..., description="Unique step id")
    op: OpName
    args: Dict[str, Any] = Field(default_factory=dict)
    out: str = Field(..., description="State key to store output under")


class Plan(BaseModel):
    version: str = "v1"
    steps: List[Step]

    @model_validator(mode="after")
    def _validate_unique_ids_and_outs(self) -> "Plan":
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate step.id in plan")

        outs = [s.out for s in self.steps]
        if len(set(outs)) != len(outs):
            raise ValueError("Duplicate step.out in plan")

        # ensure synthesize exists and is last (simplifies executor)
        if not self.steps or self.steps[-1].op != "synthesize":
            raise ValueError("Last step must be synthesize")

        return self
