# # rag/multihop.py
# from __future__ import annotations
#
# from typing import List, Dict, Optional
#
# from rag.retriever import Retriever
# from rag.planner import SelfAskPlanner
# from rag.generator import AnswerGenerator
#
#
# class MultiHopRetriever:
#     """
#     Multi-hop retrieval (Self-Ask style).
#
#     Key features:
#       - soft loop-guard (exact + semantic similarity)
#       - eval_retrieval mode:
#           store FINAL answer if produced, but continue hopping until max_hops
#           to accumulate more documents (useful for strict recall eval).
#     """
#
#     def __init__(
#         self,
#         base_retriever: Retriever,
#         generator: AnswerGenerator,
#         use_multihop: bool = True,
#         max_hops: int = 4,
#         debug: bool = False,
#         *,
#         eval_retrieval: bool = False,
#         loop_sim_threshold: float = 0.94,
#         loop_window: int = 6,
#     ):
#         self.base_retriever = base_retriever
#         self.use_multihop = use_multihop
#         self.debug = debug
#         self.llm = generator
#
#         self.eval_retrieval = bool(eval_retrieval)
#         self.loop_sim_threshold = float(loop_sim_threshold)
#         self.loop_window = int(loop_window)
#
#         # For external inspection/debugging
#         self.last_final_answer: Optional[str] = None
#         self.last_notes: List[str] = []
#         self.last_queries: List[str] = []
#
#         if self.use_multihop:
#             self.planner = SelfAskPlanner(llm=generator, max_hops=max_hops)
#             self.max_hops = max_hops
#         else:
#             self.planner = None
#             self.max_hops = 1
#
#     # ------------------------------- helpers -------------------------------
#
#     def _dedup_docs_keep_order(self, docs: List[Dict]) -> List[Dict]:
#         seen = set()
#         out: List[Dict] = []
#         for d in docs:
#             cid = d.get("chunk_id")
#             if not cid or cid in seen:
#                 continue
#             seen.add(cid)
#             out.append(d)
#         return out
#
#     def _extract_notes(self, subquestion: str, docs: List[Dict]) -> List[str]:
#         """
#         Extract short factual notes from docs (line-by-line).
#         Notes are used as planner context.
#         """
#         context = self.llm.build_context(docs, max_chars=2800)
#
#         system = (
#             "Ты извлекаешь факты из контекста.\n"
#             "Пиши только факты, без рассуждений.\n"
#             "Если релевантных фактов нет — напиши: NONE"
#         )
#         user = f"""
# Под-вопрос:
# {subquestion}
#
# Контекст:
# {context}
#
# Выпиши 3–6 коротких фактов (каждый с новой строки).
# """.strip()
#
#         out = self.llm.generate_chat(system, user, max_new_tokens=140).strip()
#         if not out:
#             return []
#
#         lines = [l.strip("-• \t") for l in out.splitlines() if l.strip()]
#         if not lines or any(l.upper() == "NONE" for l in lines):
#             return []
#
#         return [l[:240] for l in lines[:6]]
#
#     @staticmethod
#     def _normalize(s: str) -> str:
#         return " ".join((s or "").strip().lower().split())
#
#     def _is_loop(self, next_query: str, previous_queries: List[str]) -> bool:
#         """
#         Soft loop guard:
#           1) exact normalized match -> loop
#           2) semantic similarity > threshold -> loop (last N queries only)
#         """
#         nq = self._normalize(next_query)
#         if not nq:
#             return True
#
#         recent = previous_queries[-self.loop_window:] if self.loop_window > 0 else previous_queries
#         recent_norm = [self._normalize(q) for q in recent if q]
#
#         # 1) exact match
#         if nq in set(recent_norm):
#             return True
#
#         # 2) semantic similarity (only last N)
#         try:
#             nq_emb = self.base_retriever.dense.encode_passage(next_query)
#             for q in recent:
#                 if not q:
#                     continue
#                 q_emb = self.base_retriever.dense.encode_passage(q)
#                 sim = float(nq_emb @ q_emb)  # cosine if embeddings are normalized
#                 if sim >= self.loop_sim_threshold:
#                     return True
#         except Exception:
#             # fallback to exact-only (already checked)
#             return False
#
#         return False
#
#     # ------------------------------- main -------------------------------
#
#     def retrieve(self, question: str) -> List[Dict]:
#         """
#         Main multi-hop loop.
#         """
#         self.last_final_answer = None
#         self.last_notes = []
#         self.last_queries = []
#
#         if not self.use_multihop:
#             return self.base_retriever.retrieve(question)
#
#         all_docs: List[Dict] = []
#         notes: List[str] = []
#         previous_queries: List[str] = []
#
#         # entities once from original question (if available)
#         entities: List[str] = []
#         if getattr(self.base_retriever, "entity_extractor", None) is not None:
#             entities = self.base_retriever.entity_extractor.extract(question)
#
#         current_query = question
#
#         for hop in range(self.max_hops):
#             # -------- retrieve docs for current query
#             docs = self.base_retriever.retrieve(current_query)
#             all_docs.extend(docs)
#             all_docs = self._dedup_docs_keep_order(all_docs)
#
#             # -------- extract notes from this hop
#             new_notes = self._extract_notes(current_query, docs)
#             notes.extend(new_notes)
#
#             previous_queries.append(current_query)
#
#             # -------- planner step
#             plan = self.planner.plan(
#                 original_question=question,
#                 hop=hop,
#                 previous_queries=previous_queries,
#                 notes=notes,
#                 entities=entities,
#             )
#
#             if self.debug:
#                 print(f"[HOP {hop}] query={current_query}")
#                 print(f"[HOP {hop}] notes+={len(new_notes)}")
#                 print(f"[HOP {hop}] action={plan.get('action')}")
#
#             action = plan.get("action")
#
#             # ---------- FINAL ----------
#             if action == "final":
#                 if self.last_final_answer is None:
#                     self.last_final_answer = plan.get("answer")
#
#                 # Normal mode: stop at first confident final
#                 if not self.eval_retrieval:
#                     break
#
#                 # Eval-retrieval mode: force the next FOLLOWUP to keep collecting docs
#                 forced = self.planner.plan(
#                     original_question=question,
#                     hop=hop,  # same hop index is OK; planner has its own max_hops guard
#                     previous_queries=previous_queries,
#                     notes=notes,
#                     entities=entities,
#                     force_followup=True,
#                 )
#
#                 if self.debug:
#                     print(f"[HOP {hop}] final stored, eval_retrieval continues")
#                     print(f"[HOP {hop}] forced_action={forced.get('action')}")
#
#                 if forced.get("action") != "followup":
#                     break
#
#                 next_query = (forced.get("query") or "").strip()
#                 if not next_query:
#                     break
#
#                 if self._is_loop(next_query, previous_queries):
#                     if self.debug:
#                         print(f"[HOP {hop}] loop detected after FINAL (soft), stopping")
#                     break
#
#                 current_query = next_query
#                 continue
#
#             # ---------- STOP ----------
#             if action == "stop":
#                 break
#
#             # ---------- FOLLOWUP ----------
#             next_query = (plan.get("query") or "").strip()
#             if not next_query:
#                 break
#
#             if self._is_loop(next_query, previous_queries):
#                 if self.debug:
#                     print(f"[HOP {hop}] loop detected (soft), stopping")
#                 break
#
#             current_query = next_query
#
#         # expose debug info
#         self.last_notes = notes
#         self.last_queries = previous_queries
#
#         # Final rerank against original question (important!)
#         if self.base_retriever.reranker is not None and all_docs:
#             docs_copy = [dict(d) for d in all_docs]
#             scored = self.base_retriever.reranker.score(question, docs_copy)
#             return sorted(scored, key=lambda x: x.get("ce_score", 0.0), reverse=True)
#
#         return all_docs
