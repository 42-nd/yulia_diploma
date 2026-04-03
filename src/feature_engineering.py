import pandas as pd
import numpy as np
from typing import List, Tuple
import yaml

class FeatureEngineer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.reference_date = pd.to_datetime("now")

    def _calc_trend(self, group, date_col: str, value_col: str):
        """Универсальный расчёт тренда"""
        if len(group) < 2:
            return np.nan
        group = group.sort_values(date_col)
        x = (group[date_col] - group[date_col].min()).dt.days.astype(float)
        y = group[value_col].fillna(0).astype(float)
        if x.nunique() < 2 or y.nunique() < 2:
            return 0.0
        try:
            return float(np.polyfit(x, y, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def create_features(self, students_df: pd.DataFrame, lessons_df: pd.DataFrame, homeworks_df: pd.DataFrame = None) -> pd.DataFrame:
        students = students_df.copy()
        lessons = lessons_df.copy()

        # Агрегации по урокам
        # 1. attendance_rate
        att_rate = lessons.groupby('student_id')['attendance_flag'].mean()
        students['attendance_rate'] = students['student_id'].map(att_rate).fillna(0)

        # 2. avg_lesson_attendance (дубликат)
        students['avg_lesson_attendance'] = students['attendance_rate']

        # 3. attendance_trend
        trend = lessons.groupby('student_id').apply(lambda g: self._calc_trend(g, 'date', 'attendance_flag'))
        students['attendance_trend'] = students['student_id'].map(trend).fillna(0)

        # 4. days_since_last_attendance
        last_att = lessons[lessons['attendance_flag'] == 1].groupby('student_id')['date'].max()
        students['days_since_last_attendance'] = (self.reference_date - students['student_id'].map(last_att)).dt.days.fillna(-1).astype(int)

        # 5. lifetime_days
        if 'first_visit_date' in students.columns:
            students['lifetime_days'] = (self.reference_date - students['first_visit_date']).dt.days.fillna(0).astype(int)
        else:
            students['lifetime_days'] = 0

        # 6. age_group
        students['age_group'] = pd.cut(students['age'], bins=[0, 12, 15, 18], labels=['10-12', '13-15', '16-18'])

        # 7. has_parent
        students['has_parent'] = students['parent_info'].notna() & (students['parent_info'] != '')

        # 8-9. subscribed и no_auto_notifications уже 0/1
        # 10. has_vk – нет данных
        students['has_vk'] = 0
        # 11. communication_count – нет данных
        students['communication_count'] = 1

        # 12. avg_duration_lesson
        avg_dur = lessons.groupby('student_id')['duration_academic_hours'].mean()
        students['avg_duration_lesson'] = students['student_id'].map(avg_dur).fillna(0)

        # 13. средний балл за урок
        avg_grade = lessons.groupby('student_id')['lesson_grade'].mean()
        students['avg_lesson_grade'] = students['student_id'].map(avg_grade).fillna(0)

        # 14. тренд оценок за урок
        grade_trend = lessons.groupby('student_id').apply(lambda g: self._calc_trend(g, 'date', 'lesson_grade'))
        students['grade_trend'] = students['student_id'].map(grade_trend).fillna(0)

        # 15. волатильность оценок за урок
        grade_std = lessons.groupby('student_id')['lesson_grade'].std().fillna(0)
        students['grade_volatility'] = students['student_id'].map(grade_std).fillna(0)

        # 16. Домашние задания
        if homeworks_df is not None and not homeworks_df.empty:
            hw = homeworks_df.copy()
            # средний балл
            avg_hw = hw.groupby('student_id')['score'].mean()
            students['avg_hw_score'] = students['student_id'].map(avg_hw).fillna(0)
            # тренд (используем assign_date)
            hw_trend = hw.groupby('student_id').apply(lambda g: self._calc_trend(g, 'assign_date', 'score'))
            students['hw_trend'] = students['student_id'].map(hw_trend).fillna(0)
            # волатильность
            hw_std = hw.groupby('student_id')['score'].std().fillna(0)
            students['hw_volatility'] = students['student_id'].map(hw_std).fillna(0)
        else:
            students['avg_hw_score'] = 0
            students['hw_trend'] = 0
            students['hw_volatility'] = 0

        return students

    def select_features(self, df: pd.DataFrame, n_features: int = 10) -> Tuple[List[str], pd.DataFrame]:
        exclude_cols = self.config['features']['exclude_columns'].copy()
        target = self.config['features']['target']
        display_cols = ['student_id', 'full_name', 'age', 'birth_date', 'phone', 'email', 'manager_id',
                        'source', 'first_payment_date', 'last_payment_date', 'first_visit_date',
                        'last_visit_date', 'student_status', 'parent_info', 'comments',
                        'subscribed_to_newsletter', 'no_auto_notifications', 'age_group']
        training_exclude = exclude_cols + display_cols
        cols_to_drop = [c for c in training_exclude if c in df.columns and c != target]
        df_for_training = df.drop(columns=cols_to_drop, errors='ignore')
        target_vals = df_for_training[target]
        if target_vals.nunique() < 2:
            print(f"Внимание: целевая переменная '{target}' имеет только {target_vals.nunique()} уникальных значений. Отбор признаков невозможен.")
            return [], df
        numeric_cols = df_for_training.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target]
        if len(feature_cols) == 0:
            return [], df
        correlations = df_for_training[feature_cols].corrwith(df_for_training[target]).abs().sort_values(ascending=False)
        selected = correlations.head(n_features).index.tolist()
        print(f"\nОтобрано {len(selected)} признаков для обучения:")
        for i, feat in enumerate(selected, 1):
            corr = correlations[feat]
            print(f"   {i}. {feat} (корреляция: {corr:.3f})")
        return selected, df