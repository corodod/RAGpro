# rag/compiled_plan_schema.py
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
import re

_QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)

class CompiledNode(BaseModel):
    id: str = Field(..., description="Q1..Qn")
    question: str = Field(..., description="Plain question or template with {x}")
    deps: List[str] = Field(default_factory=list)
    produces_slot: Optional[str] = None  # e.g. "x"
    consumes_slot: Optional[str] = None  # only "x" supported by COMPOSE_QUERY
    out_hits: str = Field(..., description="state key for hits, e.g. hits0/h1/h2")
    out_slot: Optional[str] = None       # state key for extracted slot dict, e.g. x

    @model_validator(mode="after")
    def _clean(self) -> "CompiledNode":
        self.id = self.id.strip()
        self.question = (self.question or "").strip()
        self.deps = [d.strip() for d in (self.deps or []) if d and d.strip()]
        self.produces_slot = self.produces_slot.strip() if isinstance(self.produces_slot, str) and self.produces_slot.strip() else None
        self.consumes_slot = self.consumes_slot.strip() if isinstance(self.consumes_slot, str) and self.consumes_slot.strip() else None
        self.out_hits = self.out_hits.strip()
        self.out_slot = self.out_slot.strip() if isinstance(self.out_slot, str) and self.out_slot.strip() else None
        return self


class CompiledFinal(BaseModel):
    op: Literal["union", "intersect"] = "union"
    merge: List[str] = Field(default_factory=list, description="list of hits state keys to merge")
    out: str = "final_hits"


class CompiledPlan(BaseModel):
    original_question: str
    nodes: List[CompiledNode]
    final: Optional[CompiledFinal] = None
    synth_from: str = "hits0"
    max_evidence: int = 6

    @model_validator(mode="after")
    def _validate(self) -> "CompiledPlan":
        print("[VALIDATOR] CompiledPlan._validate called, synth_from=", repr(self.synth_from))
        if not self.nodes:
            raise ValueError("CompiledPlan.nodes empty")
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node.id")
        outs = [n.out_hits for n in self.nodes]
        if len(set(outs)) != len(outs):
            raise ValueError("duplicate out_hits")

        # ✅ NEW: synth_from must reference state key, not node id
        sf = (self.synth_from or "").strip()
        if _QID_RE.match(sf):
            raise ValueError("synth_from must be a state key (hits*/final_hits), not a node id like Q2")

        allowed = set(outs) | {"final_hits", "hits0"}
        if sf and sf not in allowed:
            raise ValueError(f"synth_from='{sf}' must be one of {sorted(allowed)}")

        return self
