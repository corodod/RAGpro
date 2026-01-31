# # rag/planner.py
# from typing import List, Dict
# import re
#
#
# class SelfAskPlanner:
#     """
#     Self-Ask planner:
#       - emits FOLLOWUP or FINAL (single line) or STOP
#       - no JSON, line-based protocol
#
#     Supports:
#       - force_followup: disallow FINAL (useful for eval_retrieval mode)
#     """
#
#     def __init__(self, llm, max_hops: int = 4):
#         self.llm = llm
#         self.max_hops = max_hops
#
#     def _build_prompt(
#         self,
#         original_question: str,
#         hop: int,
#         previous_queries: List[str],
#         notes: List[str],
#         entities: List[str] | None = None,
#         *,
#         force_followup: bool = False,
#     ) -> str:
#         prev_block = "\n".join(f"- {q}" for q in previous_queries[-6:])
#         notes_block = "\n".join(f"- {n}" for n in notes[-12:])
#         ent_block = ", ".join(entities or [])
#
#         extra_rule = ""
#         if force_followup:
#             extra_rule = (
#                 "\nВАЖНО:\n"
#                 "- FORCE_FOLLOWUP = YES → FINAL ЗАПРЕЩЁН.\n"
#                 "- Выведи только FOLLOWUP или STOP.\n"
#             )
#
#         return f"""
# Ты решаешь вопрос через Self-Ask (многошаговый поиск по Википедии).
#
# Формат вывода — строго ОДНА строка:
# FOLLOWUP: <под-вопрос для поиска>
# или
# FINAL: <короткий финальный ответ>
# или
# STOP
# {extra_rule}
#
# Правила FOLLOWUP:
# - Это поисковый под-вопрос для Википедии, 1 строка.
# - Не повторяй предыдущие запросы (даже частично).
# - Старайся включать ключевые сущности из исходного вопроса.
# - Не отвечай на исходный вопрос в FOLLOWUP.
#
# Правила FINAL:
# - FINAL давай только если по NOTES можно уверенно ответить.
# - Ответ короткий (1–2 предложения), без рассуждений.
#
# Исходный вопрос:
# {original_question}
#
# Hop: {hop}
#
# FORCE_FOLLOWUP: {"YES" if force_followup else "NO"}
#
# Ключевые сущности:
# {ent_block or "—"}
#
# Предыдущие запросы:
# {prev_block or "—"}
#
# NOTES (из документов):
# {notes_block or "—"}
# """.strip()
#
#     def plan(
#         self,
#         original_question: str,
#         hop: int,
#         previous_queries: List[str],
#         notes: List[str],
#         entities: List[str] | None = None,
#         *,
#         force_followup: bool = False,
#     ) -> Dict:
#         if hop >= self.max_hops:
#             return {"action": "stop"}
#
#         prompt = self._build_prompt(
#             original_question,
#             hop,
#             previous_queries,
#             notes,
#             entities,
#             force_followup=force_followup,
#         )
#
#         resp = self.llm.generate_chat(
#             system="Ты планировщик Self-Ask для поиска по Википедии.",
#             user=prompt,
#             max_new_tokens=80,
#         ).strip()
#
#         if not resp:
#             return {"action": "stop"}
#
#         line = resp.splitlines()[0].strip()
#
#         if line.upper().startswith("STOP"):
#             return {"action": "stop"}
#
#         m = re.match(r"(?i)FOLLOWUP:\s*(.+)", line)
#         if m:
#             q = m.group(1).strip()
#             return {"action": "followup", "query": q} if q else {"action": "stop"}
#
#         m = re.match(r"(?i)FINAL:\s*(.+)", line)
#         if m:
#             # если мы форсим FOLLOWUP, то FINAL игнорим
#             if force_followup:
#                 return {"action": "stop"}
#             ans = m.group(1).strip()
#             return {"action": "final", "answer": ans} if ans else {"action": "stop"}
#
#         return {"action": "stop"}
