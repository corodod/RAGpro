# rag/line_planner.py
from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from rag.plan_schema import Plan, Step

LINE_SYSTEM = """
Ты — планировщик для Agentic RAG.
Верни ТОЛЬКО план в виде команд, по одной команде на строку.
Никакого текста вокруг. Никаких объяснений. Только команды.

Доступные операции:
- RETRIEVE out=<key> query="<text>" top_k=<int>
- RETRIEVE out=<key> query_from=<state_key> top_k=<int>
- EXTRACT_ENTITIES out=<key> from=<hits_key> max_entities=<int>
- MAP_RETRIEVE out=<key> from=<rows_key> template="<text with {entity}>" top_k=<int> max_fanout=<int>
- FILTER_CE out=<key> rows_from=<rows_key> hits_from=<hits_by_entity_key> query_template="<text with {entity}>" threshold=<float> top_per_entity=<int>
- EXTRACT_ANSWER out=<key> from=<hits_key> question="<subquestion>"
- COMPOSE_QUERY out=<key> template="<text with {x}>" x_from=<x_key>
- UNION_HITS out=<key> from=<hits_key1,hits_key2,...>
- INTERSECT_HITS out=<key> from=<hits_key1,hits_key2,...>
- SYNTHESIZE out=answer question="<original>" from=<hits_or_evidence_key> max_evidence=<int>

Правила:
- Всегда заканчивай SYNTHESIZE.
- Ключи state — это значения out предыдущих команд.
- Если нужен последовательный multihop A→X→B:
  RETRIEVE (subq1) -> EXTRACT_ANSWER (x) -> COMPOSE_QUERY -> RETRIEVE (subq2) -> SYNTHESIZE
""".strip()

REPAIR_SYSTEM = """
Ты вывел план в неправильном формате.
Нужно: ТОЛЬКО команды, по одной на строку.
Без объяснений, без маркдауна, без ```.

Проверь:
- каждая строка начинается с операции (RETRIEVE/EXTRACT_ENTITIES/...)
- в каждой строке есть out=<key>
- последняя команда обязана быть SYNTHESIZE
""".strip()


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


class LinePlanner:
    def __init__(self, llm, *, max_new_tokens: int = 240, debug: bool = False):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.debug = debug

    def _gen(self, system: str, user: str) -> str:
        return self.llm.generate_chat(
            system=system,
            user=user,
            max_new_tokens=self.max_new_tokens,
        ).strip()

    def _parse_to_plan(self, question: str, txt: str) -> Optional[Plan]:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
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
            return None

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

        # Валидация Plan может кинуть исключение — это ок, перехватим выше.
        plan = Plan(steps=steps)

        if self.debug:
            print("[LinePlanner] Parsed lines:\n" + "\n".join(lines[:30]))
            print(f"[LinePlanner] bad_lines={bad} ok_lines={len(parsed)}")
            print("[LinePlanner] Plan:", plan.model_dump())

        return plan

    def plan(self, question: str) -> Plan:
        base_user = f"""
Сгенерируй план команд для вопроса:

{question}

Подсказка по паттернам:

Паттерн A (простой):
RETRIEVE ... -> SYNTHESIZE ...

Паттерн Seq (A→X→B):
RETRIEVE subq1 -> EXTRACT_ANSWER -> COMPOSE_QUERY -> RETRIEVE -> SYNTHESIZE

Паттерн Parallel:
RETRIEVE a -> RETRIEVE b -> UNION_HITS/INTERSECT_HITS -> SYNTHESIZE
""".strip()

        last_txt = None
        last_err = None

        # ✅ FIX 4: retry / repair loop
        for attempt in range(1, 4):  # 1 + 2 repair tries
            if attempt == 1:
                txt = self._gen(system=LINE_SYSTEM, user=base_user)
            else:
                repair_user = f"""
Вот твой предыдущий вывод, он не распарсился/не прошёл валидацию:

{last_txt}

Ошибка/проблема:
{last_err}

Сгенерируй ИСПРАВЛЕННЫЙ план. Напоминаю: только команды, по одной на строку.
""".strip()
                txt = self._gen(system=REPAIR_SYSTEM, user=repair_user)

            try:
                plan = self._parse_to_plan(question, txt)
                if plan is None:
                    raise ValueError("No parsable lines produced")
                return plan
            except Exception as e:
                last_txt = txt
                last_err = repr(e)
                if self.debug:
                    print(f"[LinePlanner] attempt={attempt} failed: {e}")

        # если всё совсем плохо — честный fallback
        parsed = [
            {"op": "RETRIEVE", "out": "hits0", "args": {"query": question, "top_k": "20"}},
            {"op": "SYNTHESIZE", "out": "answer", "args": {"question": question, "from": "hits0", "max_evidence": "6"}},
        ]
        steps = [_to_step(i + 1, d) for i, d in enumerate(parsed)]
        return Plan(steps=steps)