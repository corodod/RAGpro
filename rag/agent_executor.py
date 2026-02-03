# rag/agent_executor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag.retriever import Retriever
from rag.reranker import CrossEncoderReranker
from rag.generator import AnswerGenerator

from rag.plan_schema import Plan
from rag.line_planner import LinePlanner
from rag.llm_bundle import LLMBundle

# --- agent executor ---
AGENT_ENABLED = True

AGENT_MAX_STEPS = 8
AGENT_DEFAULT_TOP_K = 20
AGENT_MAX_FANOUT = 15
AGENT_CACHE_ENABLED = True

AGENT_CE_THRESHOLD = 0.30
AGENT_TOP_PER_ENTITY = 2

AGENT_MAX_EVIDENCE = 6

# clamp guardrails
AGENT_RETRIEVE_TOPK_MIN = 1
AGENT_RETRIEVE_TOPK_MAX = 50

AGENT_MAP_TOPK_MIN = 1
AGENT_MAP_TOPK_MAX = 30
AGENT_MAP_FANOUT_MIN = 1
AGENT_MAP_FANOUT_MAX = 20

AGENT_ENTITIES_MIN = 1
AGENT_ENTITIES_MAX = 30
AGENT_ENTITIES_DEFAULT = 20

AGENT_CE_TOP_PER_ENTITY_MIN = 1
AGENT_CE_TOP_PER_ENTITY_MAX = 3

AGENT_EVIDENCE_MIN = 1
AGENT_EVIDENCE_MAX = 10

# generation/context sizes
AGENT_EXTRACT_ANSWER_HITS_TOP = 6
AGENT_EXTRACT_ANSWER_CTX_CHARS = 2600
AGENT_EXTRACT_ANSWER_MAX_NEW_TOKENS = 120

AGENT_SYNTH_HITS_CTX_CHARS = 3200
AGENT_SYNTH_HITS_MAX_NEW_TOKENS = 180

AGENT_EVIDENCE_TEXT_MAX_CHARS = 800
AGENT_SYNTH_EVIDENCE_MAX_NEW_TOKENS = 160

@dataclass
class ExecutorConfig:
    agent_enabled: bool = AGENT_ENABLED

    max_steps: int = AGENT_MAX_STEPS
    default_top_k: int = AGENT_DEFAULT_TOP_K
    max_fanout: int = AGENT_MAX_FANOUT
    cache_enabled: bool = AGENT_CACHE_ENABLED

    ce_threshold: float = AGENT_CE_THRESHOLD
    top_per_entity: int = AGENT_TOP_PER_ENTITY

    max_evidence: int = AGENT_MAX_EVIDENCE

    # guardrails
    retrieve_topk_min: int = AGENT_RETRIEVE_TOPK_MIN
    retrieve_topk_max: int = AGENT_RETRIEVE_TOPK_MAX

    map_topk_min: int = AGENT_MAP_TOPK_MIN
    map_topk_max: int = AGENT_MAP_TOPK_MAX
    map_fanout_min: int = AGENT_MAP_FANOUT_MIN
    map_fanout_max: int = AGENT_MAP_FANOUT_MAX

    entities_min: int = AGENT_ENTITIES_MIN
    entities_max: int = AGENT_ENTITIES_MAX
    entities_default: int = AGENT_ENTITIES_DEFAULT

    ce_top_per_entity_min: int = AGENT_CE_TOP_PER_ENTITY_MIN
    ce_top_per_entity_max: int = AGENT_CE_TOP_PER_ENTITY_MAX

    evidence_min: int = AGENT_EVIDENCE_MIN
    evidence_max: int = AGENT_EVIDENCE_MAX

    # generation/context
    extract_answer_hits_top: int = AGENT_EXTRACT_ANSWER_HITS_TOP
    extract_answer_ctx_chars: int = AGENT_EXTRACT_ANSWER_CTX_CHARS
    extract_answer_max_new_tokens: int = AGENT_EXTRACT_ANSWER_MAX_NEW_TOKENS

    synth_hits_ctx_chars: int = AGENT_SYNTH_HITS_CTX_CHARS
    synth_hits_max_new_tokens: int = AGENT_SYNTH_HITS_MAX_NEW_TOKENS

    evidence_text_max_chars: int = AGENT_EVIDENCE_TEXT_MAX_CHARS
    synth_evidence_max_new_tokens: int = AGENT_SYNTH_EVIDENCE_MAX_NEW_TOKENS


def _dedup_hits_keep_order(hits: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for h in hits:
        cid = h.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(h)
    return out


def _extract_entity_candidates_from_hits(hits: List[Dict], *, limit: int = 20) -> List[str]:
    """
    Use precomputed 'entities' field from chunks if present.
    Fallback: try to parse [СУЩНОСТИ] prefix, but prefer field.
    """
    out: List[str] = []
    seen = set()

    for h in hits:
        ents = h.get("entities")
        if isinstance(ents, list):
            for e in ents:
                if not isinstance(e, str):
                    continue
                key = e.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(e.strip())
                if len(out) >= limit:
                    return out
    return out


class PlanExecutorRetriever:
    """
    Drop-in replacement for MultiHopRetriever:
      - retrieve(question) -> List[chunk dicts]
    Additionally stores last_answer and last_plan for debugging.
    """

    def __init__(
        self,
        *,
        base_retriever: Retriever,
        generator: AnswerGenerator,
        reranker: Optional[CrossEncoderReranker] = None,
        cfg: Optional[ExecutorConfig] = None,
        debug: bool = False,
    ):
        self.base_retriever = base_retriever
        # self.llm = generator
        self.llms = LLMBundle.from_single(generator)

        self.reranker = reranker
        self.cfg = cfg or ExecutorConfig()
        self.debug = debug

        # self.planner = JSONPlanner(llm=generator, debug=debug)
        self.planner = LinePlanner(llm=self.llms.planner, debug=debug)

        # debug outputs
        self.last_plan: Optional[Plan] = None
        self.last_answer: Optional[str] = None
        self.last_state: Dict[str, Any] = {}

        # retrieval cache
        self._cache: Dict[Tuple[str, int], List[Dict]] = {}

    # ------------------ core ops ------------------

    def _normalize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        LinePlanner отдаёт всё строками. Здесь приводим типы.
        """
        if not isinstance(args, dict):
            return {}

        out: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str):
                s = v.strip()

                # bool
                if s.lower() in ("true", "false"):
                    out[k] = (s.lower() == "true")
                    continue

                # int
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    try:
                        out[k] = int(s)
                        continue
                    except Exception:
                        pass

                # float
                try:
                    if any(ch in s for ch in (".", "e", "E")):
                        out[k] = float(s)
                        continue
                except Exception:
                    pass

                out[k] = s
            else:
                out[k] = v

        return out

    def _clamp(self, v, lo, hi, default):
        try:
            v = float(v)
        except Exception:
            return default
        v = max(lo, min(hi, v))
        return int(v) if isinstance(default, int) else v

    def _guard_step(self, s, state: Dict[str, Any]) -> bool:
        """
        1) Клэмпим опасные числа (top_k/max_fanout/max_entities/max_evidence/etc)
        2) Проверяем зависимости на наличие ключей в state
        """
        # ---- clamp ----
        if s.op == "retrieve":
            s.args["top_k"] = self._clamp(
                s.args.get("top_k"), 1, 50, self.cfg.default_top_k
            )

        elif s.op == "map_retrieve":
            s.args["top_k"] = self._clamp(s.args.get("top_k"), 1, 30, 15)
            s.args["max_fanout"] = self._clamp(
                s.args.get("max_fanout"), 1, 20, self.cfg.max_fanout
            )

        elif s.op == "extract_entities":
            s.args["max_entities"] = self._clamp(s.args.get("max_entities"), 1, 30, 20)

        elif s.op == "filter_ce":
            s.args["threshold"] = float(
                self._clamp(s.args.get("threshold"), 0.0, 1.0, self.cfg.ce_threshold)
            )
            s.args["top_per_entity"] = self._clamp(
                s.args.get("top_per_entity"), 1, 3, self.cfg.top_per_entity
            )

        elif s.op == "synthesize":
            s.args["max_evidence"] = self._clamp(
                s.args.get("max_evidence"), 1, 10, self.cfg.max_evidence
            )

        # ---- dependency checks ----
        dep_keys = []

        if s.op in ("extract_entities", "extract_answer", "synthesize"):
            dep_keys.append(s.args.get("from"))

        if s.op == "map_retrieve":
            dep_keys.append(s.args.get("from"))

        if s.op == "filter_ce":
            dep_keys += [s.args.get("rows_from"), s.args.get("hits_from")]

        if s.op == "compose_query":
            dep_keys.append(s.args.get("x_from"))

        if s.op == "retrieve":
            dep_keys.append(s.args.get("query_from"))

        for k in dep_keys:
            if k and k not in state:
                return False

        return True

    def _op_retrieve(self, *, query: str, top_k: int) -> List[Dict]:
        key = (query, int(top_k))
        if self.cfg.cache_enabled and key in self._cache:
            return self._cache[key]

        hits = self.base_retriever.retrieve(query)
        hits = hits[:top_k]
        hits = _dedup_hits_keep_order(hits)

        if self.cfg.cache_enabled:
            self._cache[key] = hits
        return hits

    def _op_extract_entities(self, *, question: str, from_hits: List[Dict], max_entities: int) -> List[Dict]:
        # 1) из вопроса
        ents = self.base_retriever.entity_extractor.extract(question) if self.base_retriever.entity_extractor else []
        # 2) можно добавить из топ-чанков (опционально)
        # for h in from_hits[:5]:
        #     ents += self.base_retriever.entity_extractor.extract(h.get("title","") + " " + h.get("text",""))
        # дедуп
        seen = set()
        out = []
        for e in ents:
            k = e.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append({"entity": e.strip()})
            if len(out) >= max_entities:
                break
        return out

    def _op_map_retrieve(
        self,
        *,
        rows: List[Dict],
        template: str,
        top_k: int,
        max_fanout: int,
    ) -> Dict[str, List[Dict]]:
        """
        Returns hits_by_entity: {entity -> hits}
        """
        hits_by: Dict[str, List[Dict]] = {}
        for row in rows[:max_fanout]:
            ent = (row.get("entity") or "").strip()
            if not ent:
                continue
            q = template.format(entity=ent)
            hits = self._op_retrieve(query=q, top_k=top_k)
            hits_by[ent] = hits
        return hits_by

    def _op_filter_ce(
        self,
        *,
        rows: List[Dict],
        hits_by_entity: Dict[str, List[Dict]],
        query_template: str,
        threshold: float,
        top_per_entity: int,
    ) -> List[Dict]:
        """
        For each entity, score candidate hits using cross-encoder
        and keep those above threshold (or top1 if none above? configurable).
        Returns verified rows:
          {entity, evidence_chunk_id, evidence_text, ce_score}
        """
        if self.reranker is None:
            # Fallback: no CE => pass through top evidence per entity
            out = []
            for r in rows:
                ent = (r.get("entity") or "").strip()
                hits = hits_by_entity.get(ent) or []
                for h in hits[:top_per_entity]:
                    out.append({
                        "entity": ent,
                        "chunk_id": h.get("chunk_id"),
                        "title": h.get("title", ""),
                        "text": h.get("text", ""),
                        "ce_score": None,
                    })
            return out

        verified: List[Dict] = []

        for r in rows:
            ent = (r.get("entity") or "").strip()
            if not ent:
                continue

            hits = hits_by_entity.get(ent) or []
            if not hits:
                continue

            q = query_template.format(entity=ent)

            # score with CE
            cand = [dict(h) for h in hits]  # copy
            scored = self.reranker.score(q, cand)
            scored = sorted(scored, key=lambda x: x.get("ce_score", 0.0), reverse=True)

            kept = [h for h in scored if h.get("ce_score", 0.0) >= threshold][:top_per_entity]

            # if none passed, keep top1 (optional behavior)
            if not kept and scored:
                kept = scored[:1]

            for h in kept:
                verified.append({
                    "entity": ent,
                    "chunk_id": h.get("chunk_id"),
                    "title": h.get("title", ""),
                    "text": h.get("text", ""),
                    "ce_score": float(h.get("ce_score", 0.0)) if h.get("ce_score") is not None else None,
                })

        return verified

    def _op_union_hits(self, lists: List[List[Dict]]) -> List[Dict]:
        out: List[Dict] = []
        seen = set()
        for hits in lists:
            for h in hits:
                cid = h.get("chunk_id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                out.append(h)
        return out

    def _op_intersect_hits(self, lists: List[List[Dict]]) -> List[Dict]:
        if not lists:
            return []
        sets = []
        by_id = {}
        for hits in lists:
            s = set()
            for h in hits:
                cid = h.get("chunk_id")
                if cid:
                    s.add(cid)
                    by_id[cid] = h
            sets.append(s)
        common = set.intersection(*sets) if sets else set()
        # сохраним порядок: пройдём по первому списку
        out = []
        for h in lists[0]:
            cid = h.get("chunk_id")
            if cid in common:
                out.append(by_id[cid])
        return out

    def _op_extract_answer(self, *, question: str, hits: List[Dict]) -> Dict[str, Any]:
        # возьмём небольшой контекст: top-6
        ctx = self.llms.synthesizer.build_context(
            hits[: self.cfg.extract_answer_hits_top],
            max_chars=self.cfg.extract_answer_ctx_chars,
        )

        system = (
            "Ты извлекаешь КОРОТКИЙ ответ (одну сущность/фразу) строго из контекста.\n"
            "Верни строго JSON без текста вокруг:\n"
            '{"value": "...", "chunk_id": "..."}\n'
            "Если в контексте нет ответа, верни:\n"
            '{"value": null, "chunk_id": null}'
        )
        user = f"Вопрос:\n{question}\n\nКонтекст:\n{ctx}\n\nJSON:"
        txt = self.llms.extractor.generate_chat(system=system, user=user,  max_new_tokens=self.cfg.extract_answer_max_new_tokens).strip()

        # грубый парсинг json
        import json as _json
        raw = None
        try:
            raw = _json.loads(txt)
        except Exception:
            if "{" in txt and "}" in txt:
                j = txt[txt.find("{"): txt.rfind("}") + 1]
                try:
                    raw = _json.loads(j)
                except Exception:
                    raw = None

        if not isinstance(raw, dict):
            return {"value": None, "chunk_id": None}

        val = raw.get("value")
        cid = raw.get("chunk_id")
        if isinstance(val, str):
            val = val.strip()
        if not val:
            val = None

        return {"value": val, "chunk_id": cid}

    def _op_compose_query(self, *, template: str, values: Dict[str, Any]) -> str:
        """
        Поддерживает template с любыми плейсхолдерами: {x}, {result_from_subq1}, и т.д.
        values: dict ключ -> значение для format(**values)
        """
        safe = {}
        for k, v in (values or {}).items():
            if v is None:
                safe[k] = ""
            elif isinstance(v, dict) and "value" in v:
                safe[k] = "" if v["value"] is None else str(v["value"])
            else:
                safe[k] = str(v)

        try:
            return template.format(**safe).strip()
        except KeyError:
            # fallback: если LLM вставил плейсхолдер, которого нет
            # попробуем хотя бы {x}
            x = safe.get("x", "")
            return template.replace("{x}", x).strip() if x else ""

    def _op_synthesize(
            self,
            *,
            question: str,
            evidence: List[Dict],
            max_evidence: int,
    ) -> str:
        ev = (evidence or [])[:max_evidence]

        # -----------------------------
        # Case A: это обычные hits (нет поля "entity")
        # -----------------------------
        if ev and "entity" not in ev[0]:
            context = self.llms.synthesizer.build_context(ev, max_chars=self.cfg.synth_hits_ctx_chars)

            system = (
                "Отвечай строго по контексту.\n"
                "Не выдумывай фактов.\n"
                "Если ответа нет — скажи, что недостаточно информации.\n"
                "Ответ 2–4 предложения, по-русски."
            )

            user = (
                f"Вопрос:\n{question}\n\n"
                f"Контекст:\n{context if context else 'NONE'}\n\n"
                "Ответ:"
            )

            return self.llms.synthesizer.generate_chat(system=system, user=user, max_new_tokens=self.cfg.synth_hits_max_new_tokens).strip()

        # -----------------------------
        # Case B: это evidence-rows (есть entity, chunk_id, text, ce_score)
        # -----------------------------
        blocks = []
        for i, row in enumerate(ev, start=1):
            ent = (row.get("entity") or "").strip()
            cid = (row.get("chunk_id") or "").strip()
            score = row.get("ce_score", None)

            text = (row.get("text") or "").strip()
            text = text[:800]

            blocks.append(f"[E{i}] entity={ent} chunk_id={cid} ce_score={score}\n{text}")

        context = "\n\n".join(blocks).strip()

        system = (
            "Ты отвечаешь строго по доказательствам (EVIDENCE).\n"
            "Не выдумывай фактов.\n"
            "Если в evidence нет ответа — скажи, что недостаточно информации.\n"
            "Ответ 2–4 предложения, по-русски."
        )

        user = f"""
    Вопрос:
    {question}

    EVIDENCE:
    {context if context else "NONE"}

    Сформируй ответ. Если возможно, упомяни сущности и опирайся на evidence.
    """.strip()

        return self.llms.synthesizer.generate_chat(system=system, user=user, max_new_tokens=160).strip()

    # ------------------ plan execution ------------------

    def retrieve(self, question: str) -> List[Dict]:
        self.last_answer = None
        self.last_state = {}
        self.last_plan = None

        plan = self.planner.plan(question)
        self.last_plan = plan

        state: Dict[str, Any] = {"question": question}

        steps = plan.steps[: self.cfg.max_steps]

        for s in steps:
            s.args = self._normalize_args(s.args or {})

            # guardrails: clamp + проверка зависимостей
            if not self._guard_step(s, state):
                if self.debug:
                    print(f"[Exec] skip step {s.id} op={s.op} out={s.out} args={s.args} (missing deps or unsafe args)")
                continue

            if self.debug:
                print(f"[Exec] step {s.id} op={s.op} out={s.out} args={s.args}")

            if s.op == "retrieve":
                # top_k = int(s.args.get("top_k") or self.cfg.default_top_k)
                top_k = int(s.args["top_k"])
                # 1) прямой query или main_query
                query = s.args.get("query") or s.args.get("main_query")
                # 2) query_from=<state_key>
                if not query:
                    qk = s.args.get("query_from")
                    if qk:
                        query = state.get(qk)
                # 3) если query — это строка, но совпадает с ключом state (LLM иногда так делает)
                if isinstance(query, str) and query in state:
                    query = state.get(query)
                query = query or question
                state[s.out] = self._op_retrieve(query=str(query), top_k=top_k)


            elif s.op == "extract_entities":
                src = s.args.get("from")
                max_entities = int(s.args.get("max_entities") or 20)
                hits = state.get(src) if src else None
                hits = hits if isinstance(hits, list) else []
                state[s.out] = self._op_extract_entities(from_hits=hits, max_entities=max_entities)

            elif s.op == "map_retrieve":
                src = s.args.get("from")
                template = s.args.get("template") or "{entity}"
                top_k = int(s.args.get("top_k") or 15)
                max_fanout = int(s.args.get("max_fanout") or self.cfg.max_fanout)
                rows = state.get(src) if src else None
                rows = rows if isinstance(rows, list) else []
                state[s.out] = self._op_map_retrieve(
                    rows=rows,
                    template=template,
                    top_k=top_k,
                    max_fanout=max_fanout,
                )

            elif s.op == "filter_ce":
                rows_from = s.args.get("rows_from")
                hits_from = s.args.get("hits_from")
                query_template = s.args.get("query_template") or "{entity}"
                threshold = float(s.args.get("threshold") or self.cfg.ce_threshold)
                top_per_entity = int(s.args.get("top_per_entity") or self.cfg.top_per_entity)

                rows = state.get(rows_from) if rows_from else []
                rows = rows if isinstance(rows, list) else []

                hits_by = state.get(hits_from) if hits_from else {}
                hits_by = hits_by if isinstance(hits_by, dict) else {}

                state[s.out] = self._op_filter_ce(
                    rows=rows,
                    hits_by_entity=hits_by,
                    query_template=query_template,
                    threshold=threshold,
                    top_per_entity=top_per_entity,
                )

            elif s.op == "synthesize":
                src = s.args.get("from")
                max_evidence = int(s.args.get("max_evidence") or self.cfg.max_evidence)
                evidence = state.get(src) if src else []
                evidence = evidence if isinstance(evidence, list) else []
                ans = self._op_synthesize(question=question, evidence=evidence, max_evidence=max_evidence)
                state[s.out] = ans
                self.last_answer = ans

            elif s.op == "extract_answer":
                src = s.args.get("from")
                q = s.args.get("question") or question
                hits = state.get(src) if src else []
                hits = hits if isinstance(hits, list) else []
                state[s.out] = self._op_extract_answer(question=str(q), hits=hits)


            elif s.op == "compose_query":
                template = s.args.get("template") or "{x}"
                x_from = s.args.get("x_from")
                x_obj = state.get(x_from) if x_from else None
                # values для format: и "x", и имя ключа x_from (чтобы работало {result_from_subq1})
                values = {"x": ""}
                if x_from:
                    values[x_from] = x_obj  # может быть dict {"value": ...}
                    if isinstance(x_obj, dict):
                        values["x"] = x_obj.get("value") or ""
                    else:
                        values["x"] = x_obj or ""
                state[s.out] = self._op_compose_query(template=template, values=values)


            elif s.op in ("union_hits", "intersect_hits"):
                from_arg = s.args.get("from") or ""
                keys = [k.strip() for k in from_arg.split(",") if k.strip()]
                lists = []
                for k in keys:
                    v = state.get(k)
                    lists.append(v if isinstance(v, list) else [])
                if s.op == "union_hits":
                    state[s.out] = self._op_union_hits(lists)
                else:
                    state[s.out] = self._op_intersect_hits(lists)


            else:
                raise ValueError(f"Unsupported op: {s.op}")

        self.last_state = state

        # return candidates for UI: if we have a retrieve step, return it; else empty
        # best effort: return "hits0" if exists
        hits = state.get("hits0")
        if isinstance(hits, list) and hits:
            return hits

        # fallback: if evidence rows exist, return their chunks as candidates
        ev = state.get("evidence")
        if isinstance(ev, list) and ev:
            # convert evidence rows to pseudo-hits for UI
            out = []
            for r in ev:
                out.append({
                    "chunk_id": r.get("chunk_id"),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "ce_score": r.get("ce_score"),
                })
            return out

        return []