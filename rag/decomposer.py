# rag/decomposer.py
# LLM#1
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from rag.decomp_schema import DecompGraph, DecompItem
from rag.generator import AnswerGenerator

DECOMP_SYSTEM = """
Ты — Decomposer для Agentic RAG.
Твоя задача: разложить пользовательский вопрос на небольшой граф под-вопросов (DAG), чтобы по нему можно было найти ответ в поисковом корпусе.

Верни ТОЛЬКО строки (по одной на строку), без объяснений и без маркдауна.

Формат строки:
Q<n>: <текст под-вопроса> | dep=Qk[,Qm...] | slot=<name>

Ограничения:
- Кол-во под-вопросов: 1..5
- deps на вопрос: 0..2
- dep отсутствует у корневых вопросов.
- slot=<name> указывай ТОЛЬКО если ответ этого под-вопроса НУЖЕН дальше как переменная.
- Если под-вопрос использует переменную, вставь плейсхолдер {slot} в текст.
- Если slot указан, он ДОЛЖЕН использоваться хотя бы в одном другом под-вопросе через {slot}.

Главная стратегия (самое важное):
1) Не добавляй “фоновые” вопросы (про причины, лидеров стран, общий контекст, историю, отношения между странами),
   если исходный вопрос просит конкретную сущность/факт: “кто/что/где/когда/какой/сколько”.
   Сосредоточься на том, что прямо нужно для ответа.

2) Если исходный вопрос содержит ссылку/относительную конструкцию или “скрытую сущность” (например: “который/которая/которое”, “тот, что…”, “этот/данный”, 
   “убийство/событие/договор/закон/фильм/книга/компания/город, которое/который …”), то делай 2 шага:
   - Шаг A (resolve): определить, ЧТО ЭТО за объект/событие (идентифицировать X) и сохранить в slot.
   - Шаг B (use): спросить нужный атрибут/участника/действие про {slot}; dep должен включать producer.
   Пример паттерна (НЕ копируй буквально, используй смысл):
   “Какое <X> ... ? -> slot=x” затем “Кто/что/где ... {x}? -> dep=producer”.

3) Делай минимум шагов:
   - Если можно ответить одним поисковым вопросом — верни 1 под-вопрос.
   - Делай 2 шага только когда действительно нужно сначала идентифицировать X, а потом спросить про него.

Правила deps:
- Если в тексте под-вопроса есть {slot}, то этот под-вопрос обязан зависеть (dep=...) от вопроса, который определяет slot.
- deps должны отражать порядок: сначала producer, потом consumer.

Требования к тексту под-вопросов:
- Пиши под-вопросы так, чтобы их можно было отправить в поиск (коротко, конкретно, без лишней философии).
- Не перефразируй слишком абстрактно; сохраняй ключевые слова исходного вопроса.

4) Вопросы-пересечения ("кто из ... и ...", "кто одновременно ... и ...", "кто также ..."):
   - НЕ дроби на много шагов.
   - Сделай 2 под-вопроса:
     Q1: найти список/участников по условию A
     Q2: найти список/участников по условию B
   - Не используй slot (slot нужен только для конструкций "который/тот, что..." где надо сначала идентифицировать X).

Сгенерируй под-вопросы.
""".strip()


DECOMP_REPAIR_SYSTEM = """
Ты вывел неправильный формат.
Нужно: ТОЛЬКО строки вида:
Q<n>: ... | dep=Qk[,Qm...] | slot=<name>

Без маркдауна, без ``` и без текста вокруг.
""".strip()


_LINE_RE = re.compile(r"^(Q\d+)\s*:\s*(.+)$", re.IGNORECASE)


def _parse_decomp_lines(txt: str) -> Optional[DecompGraph]:
    lines = [l.strip() for l in (txt or "").splitlines() if l.strip()]
    lines = [l for l in lines if not l.startswith("```")]

    items: List[DecompItem] = []
    for line in lines:
        m = _LINE_RE.match(line)
        if not m:
            continue
        qid = m.group(1).upper()
        rest = m.group(2).strip()

        # Split by "|"
        parts = [p.strip() for p in rest.split("|")]
        text = parts[0].strip() if parts else ""
        deps: List[str] = []
        slot: Optional[str] = None

        for p in parts[1:]:
            if p.lower().startswith("dep="):
                dep_val = p.split("=", 1)[1].strip()
                deps = [d.strip().upper() for d in dep_val.split(",") if d.strip()]
            elif p.lower().startswith("slot="):
                slot_val = p.split("=", 1)[1].strip()
                slot = slot_val if slot_val else None

        items.append(DecompItem(id=qid, text=text, deps=deps, slot=slot))

    if not items:
        return None
    return DecompGraph(items=items)


@dataclass
class DecomposerConfig:
    max_questions: int = 5
    max_deps: int = 2
    max_new_tokens: int = 220


class Decomposer:
    def __init__(self, *, llm: AnswerGenerator, cfg: Optional[DecomposerConfig] = None, debug: bool = False):
        self.llm = llm
        self.cfg = cfg or DecomposerConfig()
        self.debug = debug

        self.last_raw: Optional[str] = None

    def decompose(self, question: str) -> DecompGraph:
        user = f"""
Исходный вопрос:
{question}

Сгенерируй под-вопросы.
""".strip()

        last_txt = None
        last_err = None

        for attempt in range(1, 4):
            if attempt == 1:
                txt = self.llm.generate_chat(
                    system=DECOMP_SYSTEM,
                    user=user,
                    max_new_tokens=self.cfg.max_new_tokens,
                ).strip()
            else:
                repair_user = f"""
Предыдущий вывод был плохой:

{last_txt}

Проблема:
{last_err}

Исправь формат.
""".strip()
                txt = self.llm.generate_chat(
                    system=DECOMP_REPAIR_SYSTEM,
                    user=repair_user,
                    max_new_tokens=self.cfg.max_new_tokens,
                ).strip()

            self.last_raw = txt
            g = _parse_decomp_lines(txt)
            if g is not None:
                return g

            last_txt = txt
            last_err = "no parsable lines"

        # fallback: single question
        return DecompGraph(items=[DecompItem(id="Q1", text=question, deps=[], slot=None)])
