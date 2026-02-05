# rag/json_to_dsl.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from rag.compiled_plan_schema import CompiledPlan, CompiledNode


_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _toposort_nodes(nodes: List[CompiledNode]) -> List[CompiledNode]:
    by_id: Dict[str, CompiledNode] = {n.id: n for n in nodes}
    indeg: Dict[str, int] = {n.id: 0 for n in nodes}
    adj: Dict[str, List[str]] = {n.id: [] for n in nodes}

    for n in nodes:
        for d in n.deps:
            if d in by_id:
                adj[d].append(n.id)
                indeg[n.id] += 1

    q = [nid for nid, deg in indeg.items() if deg == 0]
    out: List[str] = []

    while q:
        cur = q.pop(0)
        out.append(cur)
        for nxt in adj.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(out) != len(nodes):
        # cycle or missing deps; fallback in given order
        return nodes

    return [by_id[i] for i in out]


@dataclass
class JsonToDslConfig:
    top_k: int = 20
    max_evidence: int = 6


class JsonToDslTranslator:
    def __init__(self, *, cfg: Optional[JsonToDslConfig] = None, debug: bool = False):
        self.cfg = cfg or JsonToDslConfig()
        self.debug = debug

    def to_lines(self, compiled: CompiledPlan) -> List[str]:
        lines: List[str] = []

        nodes = _toposort_nodes(compiled.nodes)

        # Ensure convention: if some node uses out_hits=hits0, keep it as is.
        # (We trust L2, but still apply minimal safety.)
        for n in nodes:
            q = n.question

            # If question contains {something} -> must be handled via COMPOSE_QUERY with {x}
            used_slots = _SLOT_RE.findall(q or "")
            if used_slots and n.consumes_slot != "x":
                # harden: if placeholder exists, enforce consumes_slot="x"
                n.consumes_slot = "x"

            # 1) Retrieve for this node
            if n.consumes_slot == "x":
                # Compose query from out_slot (state key)
                x_from = n.out_slot or "x"
                # Ensure template uses {x}, not {slot}
                template = q
                # Replace any {something} with {x}
                template = _SLOT_RE.sub("{x}", template)

                qkey = f"q_{n.out_hits}"
                lines.append(f'COMPOSE_QUERY out={qkey} template="{template}" x_from={x_from}')
                lines.append(f"RETRIEVE out={n.out_hits} query_from={qkey} top_k={self.cfg.top_k}")
            else:
                lines.append(f'RETRIEVE out={n.out_hits} query="{q}" top_k={self.cfg.top_k}')

            # 2) Slot extraction if needed
            if n.produces_slot:
                slot_key = n.out_slot or n.produces_slot
                # question for extractor: short answer in the producer question
                lines.append(
                    f'EXTRACT_ANSWER out={slot_key} from={n.out_hits} question="{q}"'
                )

        # 3) Merge
        synth_from = compiled.synth_from or "hits0"
        if compiled.final and compiled.final.merge:
            merge_keys = compiled.final.merge
            out_key = compiled.final.out or "final_hits"
            if compiled.final.op == "intersect":
                lines.append(f"INTERSECT_HITS out={out_key} from={','.join(merge_keys)}")
            else:
                lines.append(f"UNION_HITS out={out_key} from={','.join(merge_keys)}")
            synth_from = out_key

        # 4) Final synthesize
        max_ev = compiled.max_evidence or self.cfg.max_evidence
        lines.append(
            f'SYNTHESIZE out=answer question="{compiled.original_question}" from={synth_from} max_evidence={max_ev}'
        )
        return lines
