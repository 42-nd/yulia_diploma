import pandas as pd
import numpy as np
import os
from typing import Tuple
import yaml
from src.data_mapper import ColumnMapper

class DataPreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.mapper = ColumnMapper(self.config['paths']['mappings_dir'])

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_dir = self.config['paths']['data_dir']
        students = pd.read_excel(os.path.join(data_dir, "student_data.xls"), sheet_name="student_data")
        lessons = pd.read_excel(os.path.join(data_dir, "lessons.xls"), sheet_name="lessons")
        homeworks = pd.read_excel(os.path.join(data_dir, "homeworks.xls"), sheet_name="homeworks")
        return students, lessons, homeworks

    def rename_columns(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Переименовывает колонки по маппингу"""
        rename_dict = {ru: en for ru, en in mapping.items() if ru in df.columns}
        return df.rename(columns=rename_dict)

    def clean_students(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка таблицы учеников"""
        df = self.rename_columns(df, self.mapper.student_map)
        # Преобразование дат
        date_cols = ['first_payment_date', 'last_payment_date', 'first_visit_date', 'last_visit_date', 'birth_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        # Числовые колонки
        numeric_cols = ['visits_count', 'balance', 'total_payments', 'debt', 'visit_duration_days', 'age']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Бинарные признаки
        for col in ['subscribed_to_newsletter', 'no_auto_notifications']:
            if col in df.columns:
                df[col] = df[col].map({'да': 1, 'нет': 0}).fillna(0).astype(int)
        return df

    def clean_lessons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка таблицы уроков"""
        df = self.rename_columns(df, self.mapper.lessons_map)
        # Дата
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # attendance_flag: пришел -> 1, иначе 0
        if 'attendance' in df.columns:
            df['attendance_flag'] = df['attendance'].apply(lambda x: 1 if x == 'пришел' else 0)
        # Оценки
        if 'lesson_grade' in df.columns:
            df['lesson_grade'] = pd.to_numeric(df['lesson_grade'], errors='coerce')
        if 'behavior_grade' in df.columns:
            df['behavior_grade'] = pd.to_numeric(df['behavior_grade'], errors='coerce')
        # Длительность
        if 'duration_academic_hours' in df.columns:
            df['duration_academic_hours'] = pd.to_numeric(df['duration_academic_hours'], errors='coerce')
        if 'duration_astronomical_hours' in df.columns:
            df['duration_astronomical_hours'] = pd.to_numeric(df['duration_astronomical_hours'], errors='coerce')
        return df

    def clean_homeworks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.rename_columns(df, self.mapper.homeworks_map)
        for col in ['assign_date', 'submit_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
        return df

    def create_target(self, students_df: pd.DataFrame, lessons_df: pd.DataFrame, homeworks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание целевой переменной academic_risk на основе успеваемости + шум.
        """
        df = students_df.copy()
        
        # Посещаемость
        att_rate = lessons_df.groupby('student_id')['attendance_flag'].mean()
        df['attendance_rate'] = df['student_id'].map(att_rate).fillna(0)
        
        # Средняя оценка за уроки
        avg_grade = lessons_df.groupby('student_id')['lesson_grade'].mean()
        df['avg_lesson_grade'] = df['student_id'].map(avg_grade).fillna(0)
        
        # Средняя оценка за ДЗ
        if not homeworks_df.empty:
            avg_hw = homeworks_df.groupby('student_id')['score'].mean()
            df['avg_hw_score'] = df['student_id'].map(avg_hw).fillna(0)
        else:
            df['avg_hw_score'] = 0
        
    # Нормализуем показатели к [0,1]
        risk_att = 1 - df['attendance_rate'].clip(0, 1)
        risk_grade = (5 - df['avg_lesson_grade'].clip(0, 5)) / 4
        risk_hw = (5 - df['avg_hw_score'].clip(0, 5)) / 4
        
        # Суммарный риск (веса можно подобрать)
        risk_score = 0.4 * risk_att + 0.3 * risk_grade + 0.3 * risk_hw
        
        # Добавляем шум, чтобы избежать идеального разделения
        np.random.seed(42)
        noise = np.random.normal(0, 0.15, size=len(df))
        risk_score_noisy = (risk_score + noise).clip(0, 1)
        
        # Снижаем порог, чтобы получить больше 1
        df['academic_risk'] = (risk_score_noisy > 0.4).astype(int)   # было 0.5
        
        print(f"\nРаспределение academic_risk: 0 = {(df['academic_risk']==0).sum()}, 1 = {(df['academic_risk']==1).sum()}")
        print(f"Средний риск-скор (без шума): {risk_score.mean():.3f}, с шумом: {risk_score_noisy.mean():.3f}")
        return df

    def get_cleaned_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Основной метод: загрузка, очистка, создание target"""
        students_raw, lessons_raw, homeworks_raw = self.load_raw_data()
        students = self.clean_students(students_raw)
        lessons = self.clean_lessons(lessons_raw)
        homeworks = self.clean_homeworks(homeworks_raw)
        students = self.create_target(students, lessons, homeworks)
        return students, lessons, homeworks