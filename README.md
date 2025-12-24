запуск
Windows / мощный ПК
$env:GEN_BACKEND="cuda"      
python -m uvicorn api.app:app

MacBook M1
export GEN_BACKEND=cpu                                      
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1

по адресу:

http://127.0.0.1:8000/
