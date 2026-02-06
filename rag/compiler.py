# rag/compiler.py
# LLM#2
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from rag.decomp_schema import DecompGraph
from rag.compiled_plan_schema import CompiledPlan, CompiledNode
from rag.generator import AnswerGenerator


COMPILER_SYSTEM = """
Ты — Compiler для Agentic RAG.
Тебе на вход дают:
- исходный вопрос
- валидный список под-вопросов (DAG) в формате Qn + deps + slot

Твоя задача: вернуть СТРОГО JSON без текста вокруг.
JSON должен однозначно описывать:
- какие под-вопросы выполнять
- какие дают slot (и каким state key)
- какие используют slot через COMPOSE_QUERY (поддерживает только {x})
- как объединять результаты (union/intersect) и откуда делать synthesize

Ограничения:
- COMPOSE_QUERY поддерживает только {x}. Нельзя другие плейсхолдеры.
- Если под-вопрос содержит {slot}, то компиляция должна:
  1) извлечь slot на producer-узле через EXTRACT_ANSWER
  2) подставлять в consumer-узле через COMPOSE_QUERY с {x}
- Если слотов несколько, делай несколько последовательных шагов (но старайся <= 8 шагов).

Верни JSON, который соответствует схеме:
{
  "original_question": "...",
  "nodes": [
    {
      "id": "Q1",
      "question": "...",
      "deps": [],
      "produces_slot": "x",
      "consumes_slot": null,
      "out_hits": "hits0",
      "out_slot": "x"
    }
  ],
  "final": {"op":"union","merge":["hits0","h1"],"out":"final_hits"},
  "synth_from": "final_hits",
  "max_evidence": 6
}
""".strip()

COMPILER_REPAIR_SYSTEM = """
Ты вывел неправильный JSON.
Нужно: вернуть ТОЛЬКО валидный JSON-объект, без текста вокруг, без ```.

Проверь:
- это JSON object
- есть original_question, nodes (array), synth_from
""".strip()


@dataclass
class CompilerConfig:
    max_new_tokens: int = 260


class Compiler:
    def __init__(self, *, llm: AnswerGenerator, cfg: Optional[CompilerConfig] = None, debug: bool = False):
        self.llm = llm
        self.cfg = cfg or CompilerConfig()
        self.debug = debug
        self.last_raw: Optional[str] = None

    def _loads_json(self, txt: str) -> Optional[dict]:
        if not txt:
            return None
        s = txt.strip()
        # try raw
        try:
            return json.loads(s)
        except Exception:
            pass
        # try substring
        if "{" in s and "}" in s:
            j = s[s.find("{"): s.rfind("}") + 1]
            try:
                return json.loads(j)
            except Exception:
                return None
        return None

    def compile(self, question: str, graph: DecompGraph) -> CompiledPlan:
        user = {
            "original_question": question,
            "decomposition": [it.model_dump() for it in graph.items],
        }
        user_txt = json.dumps(user, ensure_ascii=False, indent=2)

        last_txt = None
        last_err = None

        for attempt in range(1, 4):
            if attempt == 1:
                txt = self.llm.generate_chat(
                    system=COMPILER_SYSTEM,
                    user=user_txt,
                    max_new_tokens=self.cfg.max_new_tokens,
                ).strip()
            else:
                repair_user = f"""
Предыдущий вывод был плохой:

{last_txt}

Ошибка:
{last_err}

Исправь и верни валидный JSON.
""".strip()
                txt = self.llm.generate_chat(
                    system=COMPILER_REPAIR_SYSTEM,
                    user=repair_user,
                    max_new_tokens=self.cfg.max_new_tokens,
                ).strip()

            self.last_raw = txt
            raw = self._loads_json(txt)
            if isinstance(raw, dict):
                try:
                    compiled = CompiledPlan(**raw)
                    print("[COMPILER] compiled.synth_from =", repr(compiled.synth_from))
                    compiled = self._ensure_all_nodes_present(compiled, graph)
                    return compiled
                    # return CompiledPlan(**raw)
                except Exception as e:
                    last_txt = txt
                    last_err = repr(e)
                    continue

            last_txt = txt
            last_err = "invalid json"

        # fallback: minimal compiled plan (single retrieve)
        return CompiledPlan(
            original_question=question,
            nodes=[
                {
                    "id": "Q1",
                    "question": question,
                    "deps": [],
                    "produces_slot": None,
                    "consumes_slot": None,
                    "out_hits": "hits0",
                    "out_slot": None,
                }
            ],
            final=None,
            synth_from="hits0",
            max_evidence=6,
        )

    def _ensure_all_nodes_present(self, compiled: CompiledPlan, graph: DecompGraph) -> CompiledPlan:
        want_ids = [it.id for it in (graph.items or []) if it.id]
        have_ids = {n.id for n in (compiled.nodes or [])}

        missing = [qid for qid in want_ids if qid not in have_ids]
        if not missing:
            return compiled

        used_outs = {n.out_hits for n in compiled.nodes}
        next_idx = 0

        for qid in missing:
            it = next((x for x in graph.items if x.id == qid), None)
            if not it:
                continue

            while True:
                out_hits = f"hits_extra{next_idx}"
                next_idx += 1
                if out_hits not in used_outs:
                    used_outs.add(out_hits)
                    break

            consumes = "x" if ("{" in (it.text or "") and "}" in (it.text or "")) else None

            compiled.nodes.append(
                CompiledNode(
                    id=qid,
                    question=it.text,
                    deps=it.deps,
                    produces_slot=it.slot,
                    consumes_slot=consumes,
                    out_hits=out_hits,
                    out_slot=it.slot,
                )
            )

        return compiled
