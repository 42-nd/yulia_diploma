import pandas as pd
import numpy as np
from typing import Tuple
import yaml
import os

class DataPreprocessor:
    """Предобработка данных для обучения моделей"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загрузка данных из CSV (students.csv, lessons.csv)"""
        data_dir = self.config['paths']['data_dir']

        students = pd.read_csv(f"{data_dir}/students.csv", encoding='utf-8')
        lessons = pd.read_csv(f"{data_dir}/lessons.csv", encoding='utf-8')
        homeworks_path = f"{data_dir}/homeworks.csv"
        if os.path.exists(homeworks_path):
            homeworks = pd.read_csv(homeworks_path, encoding='utf-8')
            homeworks['date'] = pd.to_datetime(homeworks['date'], errors='coerce')
        else:
            homeworks = pd.DataFrame()
        # Преобразование дат
        date_cols_lessons = ['date']
        for col in date_cols_lessons:
            if col in lessons.columns:
                lessons[col] = pd.to_datetime(lessons[col], errors='coerce')

        date_cols_students = ['created_date','first_attendance_date', 'last_attendance_date']
        for col in date_cols_students:
            if col in students.columns:
                students[col] = pd.to_datetime(students[col], errors='coerce')

        # Возвращаем пустые DataFrame для совместимости
        return students, lessons, homeworks, pd.DataFrame()

    def check_missing_values(self, students, lessons, homeworks, communications) -> dict:
        """Анализ пропусков в данных"""
        missing = {
            'students': students.isna().sum().to_dict(),
            'lessons': lessons.isna().sum().to_dict(),
            'homeworks': homeworks.isna().sum().to_dict(),
            'communications': communications.isna().sum().to_dict()
        }

        print("\nПропуски в данных:")
        for table, vals in missing.items():
            total_missing = sum(vals.values())
            if total_missing > 0:
                print(f"   {table}: {total_missing} пропусков")
            else:
                print(f"   {table}: нет пропусков")
        return missing

    def get_statistics(self, students, lessons, homeworks) -> dict:
        """Получение статистики по данным"""
        stats = {
            'students_count': len(students),
            'lessons_count': len(lessons),
            'homeworks_count': len(homeworks),
            'avg_age': students['age'].mean(),
            'avg_attendance_rate': students['attendance_rate'].mean() if 'attendance_rate' in students else 0,
            'avg_score': 0,  # теперь нет оценок
            'risk_distribution': students['risk'].value_counts().to_dict()
        }

        print("\nСтатистика данных:")
        print(f"Учеников: {stats['students_count']}")
        print(f"Уроков: {stats['lessons_count']}")
        print(f"Средний возраст: {stats['avg_age']:.1f} лет")
        print(f"Средняя посещаемость: {stats['avg_attendance_rate']*100:.1f}%")
        print(f"Риск (1): {stats['risk_distribution'].get(1, 0)} учеников")
        print(f"Риск (0): {stats['risk_distribution'].get(0, 0)} учеников")

        return stats

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    students, lessons, _, _ = preprocessor.load_data()
    preprocessor.check_missing_values(students, lessons, pd.DataFrame(), pd.DataFrame())
    preprocessor.get_statistics(students, lessons, pd.DataFrame())