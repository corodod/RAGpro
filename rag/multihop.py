# rag/multihop.py

from typing import List
from rag.retriever import Retriever
from rag.planner import MultiHopPlanner


class MultiHopRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        planner: MultiHopPlanner,
        max_hops: int = 4,
        debug: bool = False,
    ):
        self.base_retriever = base_retriever
        self.planner = planner
        self.max_hops = max_hops
        self.debug = debug

    def _extract_facts(self, docs) -> List[str]:
        """
        Convert retrieved docs into short factual strings
        """
        facts = []
        for d in docs:
            text = d.get("text", "")
            text = text.split(".")[0]
            title = d.get("title", "")
            facts.append(f"{title}: {text}")
        return facts

    def retrieve(self, question: str):
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

        # финальный rerank по исходному вопросу
        return self.base_retriever.rerank(question, all_docs)
