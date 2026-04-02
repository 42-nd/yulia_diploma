import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Tuple, Optional
import yaml
import os

class DataGenerator:
    """Генератор синтетических данных для прогнозирования ухода учеников"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.data_config = self.config['data']

    def generate(self, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Генерация двух таблиц: students (клиенты) и lessons (занятия)"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            seed = self.data_config.get('seed', 42)
            np.random.seed(seed)
            random.seed(seed)

        num_students = self.data_config['num_students']
        num_lessons_per_student_range = (3, 12)  # кол-во занятий на ученика

        # Генерация учеников
        students = []
        lessons = []

        # Сегменты для реалистичного распределения риска
        segments = np.random.choice(['high', 'mid', 'low'], size=num_students, p=[0.15, 0.60, 0.25])

        for i in range(num_students):
            segment = segments[i]

            # Параметры в зависимости от сегмента
            if segment == 'high':
                diligence = np.clip(np.random.beta(5, 1), 0.6, 0.99)
                base_attendance = np.clip(np.random.normal(0.85, 0.06), 0.70, 0.98)
                risk_bias = -0.3
            elif segment == 'low':
                diligence = np.clip(np.random.beta(1, 3), 0.05, 0.5)
                base_attendance = np.clip(np.random.normal(0.65, 0.12), 0.20, 0.85)
                risk_bias = 0.4
            else:  # mid
                diligence = np.clip(np.random.beta(2, 2), 0.3, 0.7)
                base_attendance = np.clip(np.random.normal(0.82, 0.07), 0.60, 0.95)
                risk_bias = 0.0
            has_declining = random.random() < 0.3
            daily_decline = np.random.uniform(0.01, 0.05) if has_declining else 0

            # Возраст и класс
            age = int(np.clip(np.random.normal(13, 2.5), 10, 17))
            grade = int(np.clip(age - 6, 5, 11))
            birth_date = datetime.now() - timedelta(days=age*365 + random.randint(0, 365))
            # Даты
            created_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
            first_attendance_date = created_date + timedelta(days=np.random.randint(1, 15))
            last_attendance_date = first_attendance_date + timedelta(days=np.random.randint(30, 90))

            # Дополнительные признаки
            source = random.choice(["Давно знаю", "Реклама", "Сайт", "Соцсети", "Друг"])
            responsible_manager = random.choice(["Юлиана", "Анна", "Ольга", "Дмитрий"])
            parent_info = "Елена" if random.random() > 0.3 else ""
            subscribed = random.choice(["Да", "Нет"])
            vk = random.choice(["vk.com/id123", ""])
            no_auto_notifications = random.choice(["Нет", "Да"]) if subscribed == "Да" else "Нет"

            # Целевая переменная risk (0 – не уйдёт, 1 – уйдёт)
            # Формируем на основе посещаемости, финансов и усердия
            risk_score = (
                (1 - base_attendance) * 0.7 +
                (1 - diligence) * 0.3 +
                risk_bias
            )
            risk_score = np.clip(risk_score + np.random.normal(0, 0.05), 0, 1)
            risk = 1 if risk_score > 0.45 else 0

            student = {
                'client_id': i + 1,
                'full_name': f"Ученик_{i+1}",
                'age': age,
                'birth_date': birth_date.strftime('%d.%m.%Y'),
                'phone': f"7{random.randint(9000000000, 9999999999)}",
                'email': f"student{i+1}@example.com",
                'responsible_manager': responsible_manager,
                'source': source,
                'created_date': created_date.strftime('%d.%m.%Y'),
                'total_enrollments': 0,          # заполнится после генерации уроков
                'total_attendances': 0,
                'first_attendance_date': first_attendance_date.strftime('%d.%m.%Y'),
                'last_attendance_date': last_attendance_date.strftime('%d.%m.%Y'),
                'attendance_days_span': (last_attendance_date - first_attendance_date).days if last_attendance_date else 0,
                'lifetime_from_creation_to_last_attendance': (last_attendance_date - created_date).days if last_attendance_date else 0,
                'student_status': 'Клиент' if risk == 0 else 'Бывший клиент',
                'branch': random.choice(["Основной", "Северный", "Южный"]),
                'parent_info': parent_info,
                'subscribed_to_newsletter': subscribed,
                'vk': vk,
                'no_auto_notifications': no_auto_notifications,
                'diligence': round(diligence, 3),          # скрытый параметр для генерации
                'has_declining':has_declining,
                'daily_decline':daily_decline,
                'base_attendance': round(base_attendance, 3),
                'risk': risk,
                'risk_score': round(risk_score, 3)
            }
            
            students.append(student)

        # Генерация занятий
        lesson_id = 1
        for student in students:
            client_id = student['client_id']
            num_lessons = np.random.randint(*num_lessons_per_student_range)
            start_date = datetime.strptime(student['first_attendance_date'], '%d.%m.%Y')
            base_att = student['base_attendance']
            diligence = student['diligence']

            enrolled_count = 0
            attended_count = 0

            for j in range(num_lessons):
                lesson_date = start_date + timedelta(days=j*7 + random.randint(-2, 5))
                if lesson_date > datetime.now():
                    lesson_date = datetime.now() - timedelta(days=random.randint(1, 30))

                # Признак посещения
                attendance_prob = np.clip(base_att + (diligence - 0.5)*0.1 + np.random.normal(0, 0.05), 0.05, 0.99)
                attendance = 1 if random.random() < attendance_prob else 0

                lesson = {
                    'lesson_id': lesson_id,
                    'client_id': client_id,
                    'date': lesson_date.strftime('%d.%m.%Y'),
                    'time': f"{random.randint(9,20)}:00",
                    'branch': student['branch'],
                    'program': random.choice(["Английский", "Математика", "Программирование"]),
                    'group': random.choice(["A1", "B2", "C1"]),
                    'teacher': random.choice(["Иванов", "Петров", "Сидорова"]),
                    'topic': random.choice(["Грамматика", "Лексика", "Разговор"]),
                    'enrolled': 1,   # для простоты всегда 1 (ученик записан)
                    'attended': attendance,
                    'total_absences': 0,   # можно заполнить позже
                    'excused_absences': 0,
                    'student_name': student['full_name'],
                    'enrollment_status': 'активна' if attendance == 1 else 'заморожена',
                    'attendance_flag': attendance,
                    'duration_academic_hours': random.choice([1, 1.5, 2]),
                    'duration_clock_hours': random.choice([45, 60, 90]),
                }
                lessons.append(lesson)
                lesson_id += 1

                if attendance:
                    attended_count += 1
                enrolled_count += 1

            # Обновляем поля в студенте
            student['total_enrollments'] = enrolled_count
            student['total_attendances'] = attended_count

        # Преобразуем в DataFrame

        homeworks = []
        for student in students:
            client_id = student['client_id']
            diligence = student['diligence']
            created_date = datetime.strptime(student['created_date'], '%d.%m.%Y')
            
            has_declining = student['has_declining']
            daily_decline = student['daily_decline']
            num_hw = np.random.randint(2, 8)
            for _ in range(num_hw):
                hw_date = created_date + timedelta(days=np.random.randint(1, 180))
                days = (hw_date - created_date).days
                base_score = np.clip(2 + 3 * diligence, 1, 5)
                if has_declining:
                    score_value = base_score - daily_decline * days
                else:
                    score_value = base_score
                score = np.clip(np.random.normal(score_value, 0.8), 1, 5)
                score = round(score, 1)
                homeworks.append({
                    'homework_id': len(homeworks) + 1,
                    'client_id': client_id,
                    'date': hw_date.strftime('%d.%m.%Y'),
                    'score': score,
                    'topic': random.choice(["Грамматика", "Лексика", "Письмо", "Аудирование"])
                })
        homeworks_df = pd.DataFrame(homeworks) if homeworks else pd.DataFrame(columns=['homework_id', 'client_id', 'date', 'score', 'topic'])
        for student in students:
            client_id = student['client_id']
            hw_student = homeworks_df[homeworks_df['client_id'] == client_id]
            if not hw_student.empty:
                avg = hw_student['score'].mean()
                # Тренд (наклон) по времени
                if len(hw_student) >= 2:
                    hw_sorted = hw_student.sort_values('date')
                    dates = pd.to_datetime(hw_sorted['date'], dayfirst=True)
                    x = (dates - dates.min()).dt.days.astype(float)
                    y = hw_sorted['score'].astype(float)
                    # Проверка, что есть хотя бы два уникальных дня
                    if x.nunique() >= 2:
                        slope = np.polyfit(x, y, 1)[0]
                    else:
                        slope = 0.0
                else:
                    slope = 0.0
                # Целевая переменная: риск скатиться в двоечники
                if avg < 2.5 or (slope < -0.01 and avg < 3.5):
                    academic_risk = 1
                else:
                    academic_risk = 0
            else:
                avg = 0.0
                slope = 0.0
                academic_risk = 0

            student['avg_score'] = round(avg, 2)
            student['score_trend'] = round(slope, 5)
            student['academic_risk'] = academic_risk

            # Удаляем служебные поля, чтобы не мешали обучению
            student.pop('has_declining', None)
            student.pop('daily_decline', None)
        
        communications_df = pd.DataFrame()  # не используем
        students_df = pd.DataFrame(students)
        lessons_df = pd.DataFrame(lessons)

        return students_df, lessons_df, homeworks_df, communications_df
    # data_generator.py (добавить методы в конец класса)

    def _prepare_export_df(self, students_df: pd.DataFrame) -> pd.DataFrame:
        """Преобразует students_df в формат export.csv (русские названия колонок) без финансовых полей"""
        df = students_df.copy()
        # Только нужные колонки (без финансовых)
        required_columns = [
            'ФИО', 'Телефон', 'Email', '№ клиента', 'Создан', 'Отв. менеджер',
            'Источник клиента', 'Записей', 'Посещений', 'Дата перв посещ кл', 'Дата посл посещ кл',
            'Срок посещ, дн', 'Срок жизни кл от созд до посл посещ, дн', 'Статус ученика',
            'Филиал ученика', 'Родитель (имя, телефон)', 'Подписан на рассылку', 'VK',
            'Не присылать автоуведомления', 'Возраст', 'Дата рождения'
        ]
        mapping = {
            'full_name': 'ФИО', 'phone': 'Телефон', 'email': 'Email', 'client_id': '№ клиента',
            'created_date': 'Создан', 'responsible_manager': 'Отв. менеджер', 'source': 'Источник клиента',
            'total_enrollments': 'Записей', 'total_attendances': 'Посещений',
            'first_attendance_date': 'Дата перв посещ кл', 'last_attendance_date': 'Дата посл посещ кл',
            'attendance_days_span': 'Срок посещ, дн',
            'lifetime_from_creation_to_last_attendance': 'Срок жизни кл от созд до посл посещ, дн',
            'student_status': 'Статус ученика', 'branch': 'Филиал ученика', 'parent_info': 'Родитель (имя, телефон)',
            'subscribed_to_newsletter': 'Подписан на рассылку', 'vk': 'VK', 'no_auto_notifications': 'Не присылать автоуведомления',
            'age': 'Возраст', 'birth_date': 'Дата рождения'
        }
        rename_dict = {eng: rus for eng, rus in mapping.items() if eng in df.columns}
        df_renamed = df.rename(columns=rename_dict)
        # Добавляем недостающие колонки (пустыми)
        for col in required_columns:
            if col not in df_renamed.columns:
                df_renamed[col] = ''
        # Оставляем только нужные колонки в порядке required_columns
        df_renamed = df_renamed[[c for c in required_columns if c in df_renamed.columns]]
        return df_renamed

    def _prepare_schedule_df(self, lessons_df: pd.DataFrame) -> pd.DataFrame:
        """Преобразует lessons_df в формат schedule.csv (русские названия) без финансовых полей"""
        df = lessons_df.copy()
        # Только нужные колонки (без 'Доходность' и др.)
        required_columns = [
            'Дата', 'Время', 'Филиал', 'Программа', 'Группа', 'Преподаватель',
            'Тема', 'Записано', 'Пришло', 'Пропусков всего', 'Пропусков по уваж. причине',
            'Ученик', 'Статус записи в группу', 'Посещение', 'Продолж. ас/ч', 'Продолж. ак/ч'
        ]
        mapping = {
            'date': 'Дата', 'time': 'Время', 'branch': 'Филиал', 'program': 'Программа',
            'group': 'Группа', 'teacher': 'Преподаватель', 'topic': 'Тема',
            'enrolled': 'Записано', 'attended': 'Пришло', 'total_absences': 'Пропусков всего',
            'excused_absences': 'Пропусков по уваж. причине', 'student_name': 'Ученик',
            'enrollment_status': 'Статус записи в группу', 'attendance_flag': 'Посещение',
            'duration_academic_hours': 'Продолж. ас/ч', 'duration_clock_hours': 'Продолж. ак/ч'
        }
        rename_dict = {eng: rus for eng, rus in mapping.items() if eng in df.columns}
        df_renamed = df.rename(columns=rename_dict)
        for col in required_columns:
            if col not in df_renamed.columns:
                df_renamed[col] = '' if col not in ['Записано', 'Пришло', 'Пропусков всего', 'Пропусков по уваж. причине', 'Посещение', 'Продолж. ас/ч', 'Продолж. ак/ч'] else 0
        # Оставляем только нужные колонки
        df_renamed = df_renamed[[c for c in required_columns if c in df_renamed.columns]]
        return df_renamed

    def save_crm_export(self, students_df: pd.DataFrame, filepath: str = "data/export.csv"):
        df_export = self._prepare_export_df(students_df)
        df_export.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Файл export.csv сохранён в {filepath}")

    def save_crm_schedule(self, lessons_df: pd.DataFrame, filepath: str = "data/schedule.csv"):
        df_schedule = self._prepare_schedule_df(lessons_df)
        df_schedule.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Файл schedule.csv сохранён в {filepath}")

    def save_data(self, students_df: pd.DataFrame, lessons_df: pd.DataFrame, homeworks_df=None, communications_df=None):
        data_dir = self.config['paths']['data_dir']
        os.makedirs(data_dir, exist_ok=True)
        students_df.to_csv(f"{data_dir}/students.csv", index=False, encoding='utf-8')
        lessons_df.to_csv(f"{data_dir}/lessons.csv", index=False, encoding='utf-8')
        if homeworks_df is not None:
            homeworks_df.to_csv(f"{data_dir}/homeworks.csv", index=False, encoding='utf-8')
        print(f"Данные сохранены в папку {data_dir}/")

    def save_to_excel(self, students_df: pd.DataFrame, lessons_df: pd.DataFrame, homeworks_df=None, communications_df=None,
                      filepath: str = "data/student_data.xlsx"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            students_df.to_excel(writer, sheet_name='Students', index=False)
            lessons_df.to_excel(writer, sheet_name='Lessons', index=False)
            if homeworks_df is not None:
                homeworks_df.to_excel(writer, sheet_name='Homeworks', index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name='Homeworks', index=False)
            # Communications не используем
            pd.DataFrame().to_excel(writer, sheet_name='Communications', index=False)
        print(f"Данные сохранены в {filepath}")

if __name__ == "__main__":
    generator = DataGenerator()
    students, lessons, _, _ = generator.generate()
    generator.save_data(students, lessons)