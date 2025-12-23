запуск
Windows / мощный ПК
set GEN_BACKEND=cuda
python -m uvicorn api.app:app

MacBook M1
export GEN_BACKEND=mps
python -m uvicorn api.app:app

по адресу:

http://127.0.0.1:8000/
