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

        # id -> out_hits (если synth_from вдруг пришёл как "Q2")
        id_to_hits: Dict[str, str] = {n.id: n.out_hits for n in nodes if n.id and n.out_hits}

        # удобный доступ по id
        by_id: Dict[str, CompiledNode] = {n.id: n for n in nodes if n.id}

        for n in nodes:
            q = (n.question or "").strip()

            # --- Нормализация consumes_slot ---
            # Варианты:
            #   consumes_slot = None
            #   consumes_slot = "x"
            #   consumes_slot = "home_club"   (LLM так иногда делает)  -> это значит x_from=home_club
            forced_x_from: Optional[str] = None
            if n.consumes_slot:
                cs = n.consumes_slot.strip()
                if cs and cs != "x":
                    forced_x_from = cs
                    n.consumes_slot = "x"
                elif cs == "x":
                    n.consumes_slot = "x"
                else:
                    n.consumes_slot = None

            used_slots = _SLOT_RE.findall(q)

            # Если в тексте есть {slot} — это точно slot-aware consumer
            if used_slots and n.consumes_slot != "x":
                n.consumes_slot = "x"

            # --- Решаем, нужен ли COMPOSE_QUERY ---
            # Нужно если:
            #   - consumes_slot == "x" (явно сказано компилятором/нормализацией)
            # И при этом можем вычислить x_from хоть как-то (forced / placeholder / deps).
            if n.consumes_slot == "x":
                # 1) x_from приоритеты
                x_from: Optional[str] = None

                # A) если LLM дал конкретный слот в consumes_slot (home_club), мы его сохранили в forced_x_from
                if forced_x_from:
                    x_from = forced_x_from

                # B) если в тексте есть ровно один {slot}
                if not x_from and len(used_slots) == 1:
                    x_from = used_slots[0]

                # C) иначе пытаемся вывести из deps: ищем dep-узел который produces_slot/out_slot
                if not x_from:
                    for dep_id in (n.deps or []):
                        dep = by_id.get(dep_id)
                        if dep and (dep.out_slot or dep.produces_slot):
                            x_from = dep.out_slot or dep.produces_slot
                            break

                # D) последний fallback
                x_from = x_from or "x"

                # 2) template: заменяем {любая_переменная} -> {x}
                template = _SLOT_RE.sub("{x}", q)

                # 3) КРИТИЧНО: если в тексте вообще не было {slot}, то после sub там не появится {x}.
                # Тогда добавляем {x} в конец (твой "спасательный" кусок).
                if "{x}" not in template:
                    template = (template.rstrip(" ?.!") + " {x}").strip()

                qkey = f"q_{n.out_hits}"
                lines.append(f'COMPOSE_QUERY out={qkey} template="{template}" x_from={x_from}')
                lines.append(f"RETRIEVE out={n.out_hits} query_from={qkey} top_k={self.cfg.top_k}")
            else:
                # обычный retrieve
                lines.append(f'RETRIEVE out={n.out_hits} query="{q}" top_k={self.cfg.top_k}')

            # Slot extraction if needed
            if n.produces_slot:
                slot_key = n.out_slot or n.produces_slot
                lines.append(f'EXTRACT_ANSWER out={slot_key} from={n.out_hits} question="{q}"')

        # ---------------- MERGE / SYNTH_FROM ----------------

        synth_from = (compiled.synth_from or "hits0").strip()

        # FIX A: synth_from мог прийти как node id (Q2)
        if _QID_RE.match(synth_from) and synth_from in id_to_hits:
            synth_from = id_to_hits[synth_from]

        # FIX B: если final отсутствует — объединяем все hits
        if not compiled.final:
            hit_keys = [n.out_hits for n in nodes if n.out_hits]
            if len(hit_keys) >= 2:
                lines.append(f"UNION_HITS out=final_hits from={','.join(hit_keys)}")
                synth_from = "final_hits"
            elif len(hit_keys) == 1:
                synth_from = hit_keys[0]
            else:
                synth_from = "hits0"

        # Respect compiled.final if present
        if compiled.final and compiled.final.merge:
            merge_keys = compiled.final.merge
            out_key = compiled.final.out or "final_hits"
            if compiled.final.op == "intersect":
                lines.append(f"INTERSECT_HITS out={out_key} from={','.join(merge_keys)}")
            else:
                lines.append(f"UNION_HITS out={out_key} from={','.join(merge_keys)}")
            synth_from = out_key

        max_ev = compiled.max_evidence or self.cfg.max_evidence
        lines.append(
            f'SYNTHESIZE out=answer question="{compiled.original_question}" from={synth_from} max_evidence={max_ev}'
        )
        return lines
