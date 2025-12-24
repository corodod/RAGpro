запуск
Windows / мощный ПК
- $env:GEN_BACKEND="cuda"      
- python -m uvicorn api.app:app

MacBook M1
- export GEN_BACKEND=cpu                                      
- export TOKENIZERS_PARALLELISM=false
- export OMP_NUM_THREADS=1
- export MKL_NUM_THREADS=1

- uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1

по адресу:

http://127.0.0.1:8000/


RAG(bm25 dense crossencoder)

================ RESULTS ================

n_queries = 1692

Recall@1: 0.3434 | MRR@1: 0.3434

Recall@3: 0.4663 | MRR@3: 0.3983

Recall@5: 0.5077 | MRR@5: 0.4079

Recall@10: 0.5384 | MRR@10: 0.4121

Recall@20: 0.5538 | MRR@20: 0.4132

- ≈34% вопросов: - релевантный документ найден первым
- ≈55% вопросов: - релевантный документ найден хотя бы в top-20
- ≈45% вопросов: - вообще ни один релевантный документ не найден
MRR почти не растёт после k=5, это очень важный сигнал.

Это значит:
- если релевантный документ найден,
- он обычно находится в top-5,
- если не найден там — дальше он почти не появляется.
- вообще ни один релевантный документ не найден