# rag/line_planner.py
from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from rag.plan_schema import Plan, Step


def _parse_line(line: str) -> Dict[str, Any]:
    parts = shlex.split(line.strip())
    if not parts:
        return {}
    op = parts[0].upper()
    args: Dict[str, Any] = {}
    out = None

    for token in parts[1:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if k == "out":
            out = v
        else:
            args[k] = v

    if not out:
        raise ValueError(f"Missing out= in line: {line}")

    return {"op": op, "out": out, "args": args}


def _to_step(idx: int, d: Dict[str, Any]) -> Step:
    op_map = {
        "RETRIEVE": "retrieve",
        "EXTRACT_ENTITIES": "extract_entities",
        "MAP_RETRIEVE": "map_retrieve",
        "FILTER_CE": "filter_ce",
        "SYNTHESIZE": "synthesize",
        "EXTRACT_ANSWER": "extract_answer",
        "COMPOSE_QUERY": "compose_query",
        "UNION_HITS": "union_hits",
        "INTERSECT_HITS": "intersect_hits",
    }
    op = op_map.get(d["op"])
    if not op:
        raise ValueError(f"Unknown op: {d['op']}")

    return Step(
        id=f"s{idx}",
        op=op,  # type: ignore
        args=d["args"],
        out=d["out"],
    )


class LinePlanParser:
    """
    Parse line-DSL text into Plan (no LLM here).
    """

    def __init__(self, *, debug: bool = False):
        self.debug = debug

    def parse_to_plan(self, question: str, txt: str) -> Plan:
        lines = [l.strip() for l in (txt or "").splitlines() if l.strip()]
        lines = [l for l in lines if not l.startswith("```")]

        parsed: List[Dict[str, Any]] = []
        bad = 0
        for l in lines:
            try:
                parsed.append(_parse_line(l))
            except Exception:
                bad += 1
                continue

        if not parsed:
            # hard fallback
            parsed = [
                {"op": "RETRIEVE", "out": "hits0", "args": {"query": question, "top_k": "20"}},
                {"op": "SYNTHESIZE", "out": "answer", "args": {"question": question, "from": "hits0", "max_evidence": "6"}},
            ]

        steps = [_to_step(i + 1, d) for i, d in enumerate(parsed)]

        if steps[-1].op != "synthesize":
            steps.append(
                Step(
                    id=f"s{len(steps)+1}",
                    op="synthesize",
                    args={"question": question, "from": steps[-1].out, "max_evidence": "6"},
                    out="answer",
                )
            )

        plan = Plan(steps=steps)

        if self.debug:
            print("[LinePlanParser] bad_lines=", bad)
            print("[LinePlanParser] lines:\n" + "\n".join(lines[:40]))
            print("[LinePlanParser] plan:", plan.model_dump())

        return plan
