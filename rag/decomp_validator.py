# rag/decomp_validator.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from rag.decomp_schema import DecompGraph, DecompItem

_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_slots(text: str) -> List[str]:
    if not text:
        return []
    return _SLOT_RE.findall(text)


def validate_decomp_graph(
    graph: DecompGraph,
    *,
    max_questions: int = 5,
    max_deps: int = 2,
) -> Tuple[bool, Optional[str]]:
    items = graph.items or []
    if not items:
        return False, "empty items"

    if len(items) > max_questions:
        return False, f"too many questions: {len(items)} > {max_questions}"

    by_id: Dict[str, DecompItem] = {}
    for it in items:
        if not it.id:
            return False, "empty id"
        if it.id in by_id:
            return False, f"duplicate id: {it.id}"
        by_id[it.id] = it

    # deps exist + max_deps
    for it in items:
        if len(it.deps) > max_deps:
            return False, f"{it.id}: too many deps: {len(it.deps)} > {max_deps}"
        for d in it.deps:
            if d not in by_id:
                return False, f"{it.id}: dep refers to missing id: {d}"

    # DAG check (DFS)
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(n: str) -> bool:
        if n in visited:
            return True
        if n in visiting:
            return False
        visiting.add(n)
        for d in by_id[n].deps:
            if not dfs(d):
                return False
        visiting.remove(n)
        visited.add(n)
        return True

    for it in items:
        if not dfs(it.id):
            return False, "cycle detected in deps"

    # slots defined
    slot_to_q: Dict[str, str] = {}
    for it in items:
        if it.slot:
            if it.slot in slot_to_q:
                return False, f"slot '{it.slot}' defined multiple times ({slot_to_q[it.slot]} and {it.id})"
            slot_to_q[it.slot] = it.id

    # placeholder slots usage: if {x} used -> slot x must exist
    used_slots: Set[str] = set()
    for it in items:
        for s in _extract_slots(it.text):
            used_slots.add(s)
            if s not in slot_to_q:
                return False, f"{it.id}: uses slot {{{s}}} but no question defines slot={s}"

    # optional: slot must be used somewhere
    for slot, qid in slot_to_q.items():
        if slot not in used_slots:
            # not fatal, but helps quality; treat as warning -> not failing
            pass

    # strong check: if a question uses {slot}, ensure it depends (directly or indirectly) on the slot-producer
    # Build ancestor closure for each node.
    ancestors: Dict[str, Set[str]] = {it.id: set() for it in items}

    def build_anc(qid: str) -> Set[str]:
        if ancestors[qid]:
            return ancestors[qid]
        res: Set[str] = set()
        for d in by_id[qid].deps:
            res.add(d)
            res |= build_anc(d)
        ancestors[qid] = res
        return res

    for it in items:
        anc = build_anc(it.id)
        for s in _extract_slots(it.text):
            producer = slot_to_q.get(s)
            if producer and producer != it.id and producer not in anc:
                return False, f"{it.id}: uses {{{s}}} but does not depend on producer {producer}"

            # if producer and producer != it.id and producer not in anc and it.deps:
            #     # If it has deps but doesn't include producer, it's suspicious.
            #     return False, f"{it.id}: uses {{{s}}} but does not depend on producer {producer}"

    return True, None
