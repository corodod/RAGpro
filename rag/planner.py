# rag/planner.py

from typing import List, Dict
import re


class MultiHopPlanner:
    """
    LLM-based planner that decides:
    - next query
    - or STOP
    """

    def __init__(self, llm, max_hops: int = 4):
        self.llm = llm
        self.max_hops = max_hops

    def _build_prompt(
        self,
        original_question: str,
        hop: int,
        previous_queries: List[str],
        retrieved_facts: List[str],
    ) -> str:
        facts_block = "\n".join(f"- {f}" for f in retrieved_facts[-6:])

        prev_q_block = "\n".join(previous_queries)

        return f"""
You are a retrieval planner for a multi-hop question answering system.

Your task:
- Decide what to search next
- Or decide that enough information is collected

Rules:
- If you already know the answer, output: STOP
- Otherwise output a short search query
- Do NOT answer the question
- Do NOT explain reasoning

Original question:
{original_question}

Hop number: {hop}

Previous queries:
{prev_q_block}

Known facts:
{facts_block}

Output format:
NEXT_QUERY: <query>
or
STOP
""".strip()

    def plan_next(
        self,
        original_question: str,
        hop: int,
        previous_queries: List[str],
        retrieved_facts: List[str],
    ) -> str:
        if hop >= self.max_hops:
            return "STOP"

        prompt = self._build_prompt(
            original_question,
            hop,
            previous_queries,
            retrieved_facts,
        )

        response = self.llm.generate(prompt, chunks=[]).strip()

        if response.upper().startswith("STOP"):
            return "STOP"

        match = re.search(r"NEXT_QUERY:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # fallback — safety stop
        return "STOP"
