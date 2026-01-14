# rag/multihop.py

from typing import List
from rag.retriever import Retriever
from rag.planner import MultiHopPlanner
from rag.generator import AnswerGenerator

class MultiHopRetriever:
    """
    Обёртка над базовым Retriever, поддерживающая мультихоп.
    - use_multihop: True → используем MultiHopPlanner
    - max_hops: глубина мультихопа
    Если use_multihop=False, работает как обычный Retriever.
    """

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

        if self.use_multihop:
            self.planner = MultiHopPlanner(llm=generator, max_hops=max_hops)
            self.max_hops = max_hops
        else:
            self.planner = None
            self.max_hops = 1  # просто один шаг

    def _extract_facts(self, docs) -> List[str]:
        """
        Преобразует retrieved docs в короткие факты.
        """
        facts = []
        for d in docs:
            text = d.get("text", "").split(".")[0]
            title = d.get("title", "")
            facts.append(f"{title}: {text}")
        return facts

    def retrieve(self, question: str):
        """
        Возвращает список документов для генератора.
        Если use_multihop=False, сразу вызывает базовый retriever.
        """
        if not self.use_multihop:
            return self.base_retriever.retrieve(question)

        all_docs = []
        all_facts = []
        previous_queries = []

        current_query = question

        for hop in range(self.max_hops):
            docs = self.base_retriever.retrieve(current_query)
            all_docs.extend(docs)

            facts = self._extract_facts(docs)
            all_facts.extend(facts)

            previous_queries.append(current_query)

            next_query = self.planner.plan_next(
                original_question=question,
                hop=hop,
                previous_queries=previous_queries,
                retrieved_facts=all_facts,
            )

            if self.debug:
                print(f"[HOP {hop}] query = {current_query}")
                print(f"[HOP {hop}] next = {next_query}")

            if next_query == "STOP":
                break

            current_query = next_query

        # Финальный rerank по исходному вопросу через reranker базового Retriever
        # если есть cross-encoder, то применяем score к списку документов
        if self.base_retriever.reranker is not None:
            docs_copy = [dict(d) for d in all_docs]  # чтобы не мутировать оригиналы
            scored = self.base_retriever.reranker.score(question, docs_copy)
            # сортируем по ce_score
            scored_sorted = sorted(scored, key=lambda x: x.get("ce_score", 0.0), reverse=True)
            return scored_sorted

        return all_docs
