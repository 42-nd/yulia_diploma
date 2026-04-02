import pandas as pd
import numpy as np
from typing import List, Tuple
import yaml

class FeatureEngineer:
    """Генерация признаков для обучения моделей"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.reference_date = pd.to_datetime("now")

    def create_features(self, students_df: pd.DataFrame, lessons_df: pd.DataFrame, homeworks_df=None, communications_df=None) -> pd.DataFrame:
        """Создание признаков на основе students и lessons"""
        students = students_df.copy()
        lessons = lessons_df.copy()

        # Преобразование дат
        lessons['date'] = pd.to_datetime(lessons['date'], errors='coerce')
        students['created_date'] = pd.to_datetime(students['created_date'], errors='coerce')
        students['first_attendance_date'] = pd.to_datetime(students['first_attendance_date'], errors='coerce')
        students['last_attendance_date'] = pd.to_datetime(students['last_attendance_date'], errors='coerce')
        # Агрегированные признаки на уровне ученика
        # 1. attendance_rate (общий)
        students['attendance_rate'] = students['total_attendances'] / students['total_enrollments'].clip(lower=1)

        # 2. avg_lesson_attendance (средняя посещаемость на занятие)
        lesson_att = lessons.groupby('client_id')['attendance_flag'].mean()
        students['avg_lesson_attendance'] = students['client_id'].map(lesson_att).fillna(0)

        # 3. attendance_trend (тренд посещаемости по времени)
        def calc_trend(group):
            if len(group) < 2:
                return np.nan
            group = group.sort_values('date')
            # Убедимся, что даты корректны
            if group['date'].isna().any():
                return np.nan
            x = (group['date'] - group['date'].min()).dt.days.astype(float)
            y = group['attendance_flag'].astype(float)
            # Проверка на константность
            if x.nunique() < 2 or y.nunique() < 2:
                return 0.0
            # Защита от вырожденной матрицы
            try:
                slope = float(np.polyfit(x, y, 1)[0])
                return slope
            except (np.linalg.LinAlgError, ValueError):
                return 0.0

        trend = lessons.groupby('client_id').apply(calc_trend)
        students['attendance_trend'] = students['client_id'].map(trend).fillna(0)

        # 4. days_since_last_attendance
        last_att = lessons.groupby('client_id')['date'].max()
        students['days_since_last_attendance'] = (self.reference_date - students['client_id'].map(last_att)).dt.days.fillna(-1).astype(int)

        # 9. lifetime_days
        students['lifetime_days'] = (self.reference_date - students['created_date']).dt.days.fillna(0).astype(int)

        # 10. age_group
        students['age_group'] = pd.cut(students['age'], bins=[0, 12, 15, 18], labels=['10-12', '13-15', '16-18'])

        # 11. has_parent
        students['has_parent'] = students['parent_info'].notna() & (students['parent_info'] != '')

        # 12. subscribed
        students['subscribed'] = (students['subscribed_to_newsletter'] == 'Да').astype(int)

        # 13. no_auto_notifications
        students['no_auto_notifications'] = (students['no_auto_notifications'] == 'Да').astype(int)

        # 14. has_vk
        students['has_vk'] = students['vk'].notna() & (students['vk'] != '')

        # 15. communication_count – proxy: количество уникальных менеджеров (пока просто 1)
        students['communication_count'] = 1

        # 16. avg_duration_lesson
        avg_dur = lessons.groupby('client_id')['duration_academic_hours'].mean()
        students['avg_duration_lesson'] = students['client_id'].map(avg_dur).fillna(0)

        # Удаляем служебные столбцы, которые не нужны для обучения
        drop_cols = ['diligence', 'base_attendance', 'financial_activity', 'risk_score', 'risk']
        students.drop(columns=[c for c in drop_cols if c in students.columns], inplace=True)
        if homeworks_df is not None and not homeworks_df.empty:
            homeworks = homeworks_df.copy()
            homeworks['date'] = pd.to_datetime(homeworks['date'], errors='coerce')
            # Средний балл
            avg_score = homeworks.groupby('client_id')['score'].mean().round(2)
            students['avg_score'] = students['client_id'].map(avg_score).fillna(0)

            # Тренд оценок
            def calc_score_trend(group):
                if len(group) < 2:
                    return np.nan
                group = group.sort_values('date')
                x = (group['date'] - group['date'].min()).dt.days.astype(float)
                y = group['score']
                if x.nunique() < 2:
                    return 0.0
                return float(np.polyfit(x, y, 1)[0])
            trend = homeworks.groupby('client_id').apply(calc_score_trend)
            students['score_trend'] = students['client_id'].map(trend).fillna(0)

            # Вариативность оценок (std)
            std_score = homeworks.groupby('client_id')['score'].std().fillna(0)
            students['score_volatility'] = students['client_id'].map(std_score).fillna(0)
        else:
            students['avg_score'] = 0.0
            students['score_trend'] = 0.0
            students['score_volatility'] = 0.0
        # Возвращаем полный датафрейм (включая student_id и risk)
        return students

    def select_features(self, df: pd.DataFrame, n_features: int = 10) -> Tuple[List[str], pd.DataFrame]:
        """Отбор признаков — исключаем идентификаторы, текстовые поля и целевую переменную"""
        exclude_cols = self.config['features']['exclude_columns'].copy()
        target = self.config['features']['target']

        # Добавляем колонки для отображения (не для обучения)
        display_cols = ['client_id', 'full_name', 'age', 'birth_date', 'phone', 'email', 'responsible_manager',
                        'source', 'created_date', 
                        'first_attendance_date', 'last_attendance_date', 'student_status', 'branch',
                        'parent_info', 'vk', 'subscribed_to_newsletter', 'no_auto_notifications', 'age_group']

        # Исключаем из обучения
        training_exclude = exclude_cols + display_cols
        cols_to_drop = [c for c in training_exclude if c in df.columns and c != target]
        df_for_training = df.drop(columns=cols_to_drop, errors='ignore')

        numeric_cols = df_for_training.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target]

        if len(feature_cols) == 0:
            print("Внимание: не найдено числовых признаков для отбора!")
            return [], df

        correlations = df_for_training[feature_cols].corrwith(df_for_training[target]).abs().sort_values(ascending=False)
        selected = correlations.head(n_features).index.tolist()

        print(f"\nОтобрано {len(selected)} признаков для обучения:")
        for i, feat in enumerate(selected, 1):
            corr = correlations[feat]
            print(f"   {i}. {feat} (корреляция: {corr:.3f})")

        # Возвращаем полный датафрейм (с колонками для отображения)
        result_df = df.copy()
        return selected, result_df

if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    students, lessons, _, _ = preprocessor.load_data()
    engineer = FeatureEngineer()
    merged = engineer.create_features(students, lessons)
    print(f"Создано признаков: {merged.shape}")
    selected, reduced = engineer.select_features(merged)
    print(f"Отобранные признаки: {selected}")