# rag/metrics_qa.py
import re
from collections import Counter


def normalize_ru(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"ё", "е", s)
    s = re.sub(r"[^a-zа-я0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_ru(pred) == normalize_ru(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_ru(pred).split()
    gold_toks = normalize_ru(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
'''
Что делает

Этот файл — чистая математика, он:
- не знает про retrieval
- не знает про LLM
- не знает про датасет

Он просто говорит:
“Два текста — это одинаковый ответ или нет?”

Метрики внутри
- Exact Match (EM)    “Совпадает ли ответ дословно?”
- Token F1            “Насколько ответы пересекаются по смысловым словам?”
'''