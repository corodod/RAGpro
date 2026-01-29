# rag/multihop.py
from typing import List, Dict, Optional
from rag.retriever import Retriever
from rag.planner import SelfAskPlanner
from rag.generator import AnswerGenerator


class MultiHopRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        generator: AnswerGenerator,
        use_multihop: bool = True,
        max_hops: int = 4,
        debug: bool = False,
    ):
        self.base_retriever = base_retriever
        self.use_multihop = use_multihop
        self.debug = debug
        self.llm = generator

        if self.use_multihop:
            self.planner = SelfAskPlanner(llm=generator, max_hops=max_hops)
            self.max_hops = max_hops
        else:
            self.planner = None
            self.max_hops = 1

    def _dedup_docs_keep_order(self, docs: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for d in docs:
            cid = d.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(d)
        return out

    def _extract_notes(self, subquestion: str, docs: List[Dict]) -> List[str]:
        context = self.llm.build_context(docs, max_chars=2800)

        system = (
            "Ты извлекаешь факты из контекста.\n"
            "Пиши только факты, без рассуждений.\n"
            "Если релевантных фактов нет — напиши: NONE"
        )
        user = f"""
Под-вопрос:
{subquestion}

Контекст:
{context}

Выпиши 3–6 коротких фактов (каждый с новой строки).
""".strip()

        out = self.llm.generate_chat(system, user, max_new_tokens=140).strip()
        if not out:
            return []

        lines = [l.strip("-• \t") for l in out.splitlines() if l.strip()]
        if not lines or any(l.upper() == "NONE" for l in lines):
            return []

        return [l[:240] for l in lines[:6]]

    def _normalize(self, s: str) -> str:
        return " ".join((s or "").strip().lower().split())

    def retrieve(self, question: str) -> List[Dict]:
        if not self.use_multihop:
            return self.base_retriever.retrieve(question)

        all_docs: List[Dict] = []
        notes: List[str] = []
        previous_queries: List[str] = []

        # сущности один раз из исходного вопроса
        entities = []
        if getattr(self.base_retriever, "entity_extractor", None) is not None:
            entities = self.base_retriever.entity_extractor.extract(question)

        current_query = question
        final_answer: Optional[str] = None

        for hop in range(self.max_hops):
            docs = self.base_retriever.retrieve(current_query)
            all_docs.extend(docs)
            all_docs = self._dedup_docs_keep_order(all_docs)

            new_notes = self._extract_notes(current_query, docs)
            notes.extend(new_notes)

            previous_queries.append(current_query)

            plan = self.planner.plan(
                original_question=question,
                hop=hop,
                previous_queries=previous_queries,
                notes=notes,
                entities=entities,
            )

            if self.debug:
                print(f"[HOP {hop}] query={current_query}")
                print(f"[HOP {hop}] notes+={len(new_notes)}")
                print(f"[HOP {hop}] action={plan.get('action')}")

            if plan["action"] == "final":
                final_answer = plan.get("answer")
                break

            if plan["action"] == "stop":
                break

            next_query = plan["query"]

            # loop guard
            if self._normalize(next_query) in {self._normalize(q) for q in previous_queries}:
                if self.debug:
                    print(f"[HOP {hop}] loop detected, stopping")
                break

            current_query = next_query

        # финальный rerank под исходный вопрос (важно!)
        if self.base_retriever.reranker is not None and all_docs:
            docs_copy = [dict(d) for d in all_docs]
            scored = self.base_retriever.reranker.score(question, docs_copy)
            return sorted(scored, key=lambda x: x.get("ce_score", 0.0), reverse=True)

        return all_docs
