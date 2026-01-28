# rag/planner.py

from typing import List, Dict
import re


class MultiHopPlanner:
    """
    Planner based on relation-gap reasoning.

    The planner:
    - Identifies missing relational facts needed to answer the question
    - Generates a targeted retrieval query for that missing fact
    - Decides STOP only when the answer is stable and allowed
    """

    def __init__(self, llm, max_hops: int = 4):
        self.llm = llm
        self.max_hops = max_hops

    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        original_question: str,
        hop: int,
        previous_queries: List[str],
        retrieved_facts: List[str],
        signals: Dict,
    ) -> str:
        facts_block = "\n".join(f"- {f}" for f in retrieved_facts[-8:])
        prev_q_block = "\n".join(previous_queries)

        return f"""
You are a reasoning planner for a multi-hop retrieval system.

Your goal:
Identify the SINGLE missing fact required to answer the question.

Think in terms of relations:
<Entity A> --[RELATION]--> <Entity B or VALUE>

You must:
- Identify what relation is missing
- Formulate a search query to retrieve that fact

Rules:
- Do NOT answer the question
- Do NOT explain reasoning
- Do NOT rephrase previous queries
- Generate ONE precise factual query

STOP is allowed ONLY if:
- The system explicitly allows STOP
- The answer is already stable

Original question:
{original_question}

Hop number:
{hop}

Previous queries:
{prev_q_block or "—"}

Known facts:
{facts_block or "—"}

Signals:
- Answer stable: {"YES" if signals.get("answer_stable") else "NO"}
- STOP allowed: {"YES" if signals.get("stop_allowed") else "NO"}

Output format:
MISSING_FACT: <Entity A> | <RELATION> | <Entity B or ?>
NEXT_QUERY: <query>

or

STOP
""".strip()

    # ------------------------------------------------------------------

    def plan_next(
        self,
        original_question: str,
        hop: int,
        previous_queries: List[str],
        retrieved_facts: List[str],
        signals: Dict,
    ) -> str:
        if hop >= self.max_hops:
            return "STOP"

        prompt = self._build_prompt(
            original_question=original_question,
            hop=hop,
            previous_queries=previous_queries,
            retrieved_facts=retrieved_facts,
            signals=signals,
        )

        response = self.llm.generate(prompt, chunks=[]).strip()

        # ---------------- STOP ----------------
        if response.upper().startswith("STOP"):
            return "STOP"

        # ---------------- NEXT_QUERY ----------------
        match = re.search(r"NEXT_QUERY:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return "STOP"
