# rag/json_to_dsl.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from rag.compiled_plan_schema import CompiledPlan, CompiledNode


_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)  # ✅ add

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

        # ✅ NEW: id -> out_hits mapping (to fix synth_from=Qn)
        id_to_hits: Dict[str, str] = {n.id: n.out_hits for n in nodes if n.id and n.out_hits}

        for n in nodes:
            q = n.question

            used_slots = _SLOT_RE.findall(q or "")
            if used_slots and n.consumes_slot != "x":
                n.consumes_slot = "x"

            # 1) Retrieve for this node
            if n.consumes_slot == "x":
                # 1) Determine which slot is being consumed (prefer placeholder name)
                used_slots = _SLOT_RE.findall(q or "")

                x_from: Optional[str] = None

                # Case A: question explicitly references {slot} (best signal)
                if len(used_slots) == 1:
                    x_from = used_slots[0]

                # Case B: no/ambiguous placeholder -> infer from deps (producer with produces_slot)
                if not x_from:
                    for dep_id in (n.deps or []):
                        dep = next((m for m in nodes if m.id == dep_id), None)
                        if dep and dep.produces_slot:
                            # producer writes slot into state under dep.out_slot or dep.produces_slot
                            x_from = dep.out_slot or dep.produces_slot
                            break

                # Final fallback (won't pass guard unless state has "x", but keeps behavior explicit)
                x_from = x_from or "x"

                template = _SLOT_RE.sub("{x}", q)
                # если placeholder-ов нет вообще, то нечего подставлять -> добавляем {x}
                if "{x}" not in template:
                    # мягко: убираем "данного/этого ..." (опционально), а потом добавляем x
                    # можешь оставить только добавление "{x}" если не хочешь лезть в текст
                    template = (template.rstrip(" ?.!") + " {x}").strip()
                qkey = f"q_{n.out_hits}"

                lines.append(f'COMPOSE_QUERY out={qkey} template="{template}" x_from={x_from}')
                lines.append(f"RETRIEVE out={n.out_hits} query_from={qkey} top_k={self.cfg.top_k}")
            else:
                lines.append(f'RETRIEVE out={n.out_hits} query="{q}" top_k={self.cfg.top_k}')

            # 2) Slot extraction if needed
            if n.produces_slot:
                slot_key = n.out_slot or n.produces_slot
                lines.append(f'EXTRACT_ANSWER out={slot_key} from={n.out_hits} question="{q}"')

        # ---------------- MERGE / SYNTH_FROM ----------------

        # Start with compiler-provided synth_from
        synth_from = (compiled.synth_from or "hits0").strip()

        # ✅ FIX A: if synth_from mistakenly refers to node id (Q2), map it to hits key
        if _QID_RE.match(synth_from) and synth_from in id_to_hits:
            synth_from = id_to_hits[synth_from]

        # ✅ FIX B: if final is not provided but we have multiple hit-lists,
        # union everything into final_hits so synthesize has best evidence pool.
        if not compiled.final:
            hit_keys = [n.out_hits for n in nodes if n.out_hits]
            if len(hit_keys) >= 2:
                lines.append(f"UNION_HITS out=final_hits from={','.join(hit_keys)}")
                synth_from = "final_hits"
            elif len(hit_keys) == 1:
                synth_from = hit_keys[0]
            else:
                synth_from = "hits0"

        # If final exists, respect it
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
