# scripts/download_wiki.py
from datasets import load_dataset
import json
from tqdm import tqdm
from pathlib import Path

# 1. Находим корень проекта (папка RAGRPO)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. data/raw
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# 3. Путь к файлу
output_path = DATA_RAW / "wiki_corpus.jsonl"

# 4. Загружаем датасет
dataset = load_dataset("kaengreg/rubq", "corpus")

# 5. Записываем файл
with open(output_path, "w", encoding="utf-8") as f:
    for item in tqdm(dataset["train"]):
        record = {
            "doc_id": item["_id"],
            "title": item.get("title", ""),
            "text": item["text"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved to: {output_path}")
