import json
import os
from typing import Dict

class ColumnMapper:
    def __init__(self, mappings_dir: str = "./data_generator/templates"):
        self.mappings_dir = mappings_dir
        self.student_map = self._load_mapping("student_data.json")
        self.lessons_map = self._load_mapping("lessons.json")
        self.homeworks_map = self._load_mapping("homeworks.json")

    def _load_mapping(self, filename: str) -> Dict[str, str]:
        path = os.path.join(self.mappings_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mapping file {path} not found")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Преобразуем список словарей в словарь {russian_name: english_name}
        return {item["russian_name"]: item["english_name"] for item in data}