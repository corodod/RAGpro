# rag/multihop.py

from typing import List, Dict, Optional
import numpy as np
from rag.retriever import Retriever
from rag.planner import MultiHopPlanner
from rag.generator import AnswerGenerator

class MultiHopRetriever:
    """
    Multi-hop retriever with answer-stability based STOP criterion.

    Core idea:
    - On each hop, probe a candidate answer from accumulated facts
    - STOP when the answer stabilizes (does not change semantically)
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
            self.max_hops = 1

        self.llm = generator

    # ------------------------------------------------------------------

    def _extract_facts(self, docs) -> List[str]:
        """
        Convert retrieved documents into short factual statements.
        """
        facts = []
        for d in docs:
            text = d.get("text", "").split(".")[0]
            title = d.get("title", "")
            facts.append(f"{title}: {text}")
        return facts

    # ------------------------------------------------------------------

    def retrieve(self, question: str):
        """
        Main multi-hop retrieval loop with answer-stability STOP.
        """
        if not self.use_multihop:
            return self.base_retriever.retrieve(question)

        all_docs: List[Dict] = []
        all_facts: List[str] = []
        all_fact_embs: List[np.ndarray] = []

        previous_queries: List[str] = []
        candidate_answers: List[Optional[str]] = []

        current_query = question

        for hop in range(self.max_hops):
            # ----------------------------------------------------------
            # Retrieval
            docs = self.base_retriever.retrieve(current_query)
            all_docs.extend(docs)

            facts = self._extract_facts(docs)
            all_facts.extend(facts)

            # ----------------------------------------------------------
            # Answer probe
            answer = self._probe_answer(question, all_facts)
            candidate_answers.append(answer)

            # ----------------------------------------------------------
            # Answer stability check
            answer_stable = False
            if hop > 0:
                prev_answer = candidate_answers[-2]
                answer_stable = self._answers_equivalent(prev_answer, answer)

            # ----------------------------------------------------------
            # Planner
            previous_queries.append(current_query)

            stop_allowed = bool(answer and answer_stable and hop > 0)

            next_query = self.planner.plan_next(
                original_question=question,
                hop=hop,
                previous_queries=previous_queries,
                retrieved_facts=all_facts,
                signals={
                    "stop_allowed": stop_allowed,
                    "answer_stable": answer_stable,
                },
            )

            if self.debug:
                print(f"[HOP {hop}] answer={answer}")
                print(f"[HOP {hop}] stable={answer_stable}")
                print(f"[HOP {hop}] next={next_query}")

            if next_query == "STOP":
                break

            current_query = next_query

        # --------------------------------------------------------------
        # Final rerank
        if self.base_retriever.reranker is not None:
            docs_copy = [dict(d) for d in all_docs]
            scored = self.base_retriever.reranker.score(question, docs_copy)
            return sorted(
                scored,
                key=lambda x: x.get("ce_score", 0.0),
                reverse=True,
            )

        return all_docs

    # ------------------------------------------------------------------

    def _probe_answer(self, question: str, facts: List[str]) -> Optional[str]:
        """
        Lightweight answer probe.
        Returns a short answer or None if not answerable.
        """
        facts_block = "\n".join(f"- {f}" for f in facts[-8:])

        prompt = f"""
You are given a question and known facts.

Question:
{question}

Facts:
{facts_block}

If the answer can be determined, give a SHORT answer (1 sentence max).
If not enough information, answer: UNKNOWN.

Answer:
""".strip()

        out = self.llm.generate(prompt, chunks=[]).strip()

        if not out or out.upper().startswith("UNKNOWN"):
            return None

        return out

    # ------------------------------------------------------------------

    def _answers_equivalent(
        self,
        a: Optional[str],
        b: Optional[str],
    ) -> bool:
        """
        Semantic equivalence check via embeddings.
        """
        if not a or not b:
            return False

        emb_a = self.base_retriever.dense.encode_passage(a)
        emb_b = self.base_retriever.dense.encode_passage(b)

        sim = float(emb_a @ emb_b)

        return sim >= 0.90
