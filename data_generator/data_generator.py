import os
import json
import random
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Union

class DataGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        
        # Paths
        self.base_dir = Path(__file__).parent
        self.template_dir = self.base_dir / 'templates'
        self.corpus_dir = self.base_dir / 'corpus'
        self.output_dir = self.base_dir / 'generated_data'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load corpora
        self.names_corpus = self._load_json('student_names.json')
        self.topics_corpus = self._load_json('topics.json')
        self.comments_corpus = self._load_json('comments.json')
        self.parent_names_corpus = self._load_json('parent_names.json')
        self.email_domains = self._load_json('email.json')
        
        # Load column orders from CSV templates
        self.student_columns = self._load_template_columns('student_data_template.csv')
        self.lessons_columns = self._load_template_columns('lessons_template.csv')
        self.homeworks_columns = self._load_template_columns('homeworks_template.csv')
        
        # Data containers
        self.students_df = None
        self.lessons_df = None
        self.homeworks_df = None
        self.expulsion_map = {}
        
        # Predefined data from config
        self.sources = self.config.get('sources', ['интернет', 'друзья', 'реклама', 'соцсети', 'другое'])
    
    def _load_json(self, filename: str) -> Union[dict, list]:
        path = self.corpus_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Corpus file {path} not found")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_template_columns(self, template_name: str) -> list:
        template_path = self.template_dir / template_name
        if template_path.exists():
            df = pd.read_csv(template_path, nrows=0)
            return list(df.columns)
        else:
            raise FileNotFoundError(f"Template {template_path} not found")
    
    def generate(self):
        self._generate_students()
        self._generate_lessons()
        self._generate_homeworks()
        self._update_student_stats()
        self._save_to_xls()
        print("Data generation completed. Files saved in:", self.output_dir)
    
    def _generate_students(self):
        n = self.config['num_students']
        student_ids = [random.randint(10000, 99999) for _ in range(n)]
        data = []
        today = datetime.now().date()
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d').date()
        
        for i in range(n):
            # Choose random name from corpus
            name_entry = random.choice(self.names_corpus)
            first_name_ru = name_entry['first_name_ru']
            last_name_ru = name_entry['last_name_ru']
            first_name_en = name_entry['first_name_en']
            last_name_en = name_entry['last_name_en']
            full_name = f"{first_name_ru} {last_name_ru}"
            
            # Generate email: first letter of first name + last name (lowercase) @ domain
            email_local = (first_name_en[0] + "." + last_name_en).lower()
            email_domain = random.choice(self.email_domains)
            email = f"{email_local}@{email_domain}"
            
            phone = f"+7{random.randint(900, 999)}{random.randint(1000000, 9999999)}"
            student_id = student_ids[i]
            manager_id = random.randint(1000, 9999)
            source = random.choice(self.sources)
            
            total_payments = random.randint(10000, 300000)
            balance = random.randint(-50000, total_payments)
            debt = abs(balance) if balance < 0 else 0
            first_payment_date = self._random_date(start_date, end_date)
            last_payment_date = self._random_date(first_payment_date, end_date)
            
            birth_date = self._random_date(date(2006, 1, 1), date(2017, 12, 31))
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            # Expulsion
            expulsion_date = None
            if random.random() < self.config['expulsion_prob']:
                days_after_start = random.randint(self.config['expulsion_min_days_after_start'],
                                                  self.config['expulsion_max_days_after_start'])
                expulsion_date = start_date + timedelta(days=days_after_start)
                if expulsion_date > today:
                    expulsion_date = today - timedelta(days=random.randint(1, 30))
            self.expulsion_map[student_id] = expulsion_date
            student_status = 'отчислен' if expulsion_date and expulsion_date <= today else 'активен'
            
            # Parent info
            parent_name = random.choice(self.parent_names_corpus)
            parent_phone = f"+7{random.randint(900, 999)}{random.randint(1000000, 9999999)}"
            parent_info = f"{parent_name}, {parent_phone}"
            
            # Comments
            comment = random.choice(self.comments_corpus)
            
            row = {
                'ФИО': full_name,
                'Телефон': phone,
                'Email': email,
                '№ клиента': student_id,
                'Отв. менеджер': manager_id,
                'Источник клиента': source,
                'Посещений': 0,
                'Баланс': balance,
                'Приход (всего платежей)': total_payments,
                'Задолженность всего, сумма': debt,
                'Дата перв платежа': first_payment_date,
                'Дата посл платежа': last_payment_date,
                'Дата перв посещ кл': None,
                'Дата посл посещ кл': None,
                'Срок посещ, дн': 0,
                'Комментарии к клиенту': comment,
                'Статус ученика': student_status,
                'Договоры': random.randint(10000, 99999),
                'Справки': random.randint(10000, 99999),
                'Иные документы': random.randint(10000, 99999),
                'Возраст': age,
                'Дата рождения': birth_date,
                'Родитель (имя, телефон)': parent_info,
                'Подписан на рассылку': random.choice(['да', 'нет']),
                'Не присылать автоуведомления': random.choice(['да', 'нет']),
                'Согласие на получение рассылок': 'да',
                'Согласие с пользовательским соглашением': 'да',
                'Согласие на обработку персональных данных': 'да',
                'Согласие с политикой конфиденциальности': 'да',
                'Подтверждение возраста': 'да',
                'Согласие на обработку персональных данных несовершеннолетнего': 'да',
            }
            data.append(row)
        
        self.students_df = pd.DataFrame(data)
    
    def _generate_lessons(self):
        n_lessons = self.config['num_lessons']
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d').date()
        subjects = self.config['subjects']
        teacher = self.config['teacher']
        att_weights = self.config['attendance_weights']
        lesson_status_weights = self.config['lesson_status_weights']
        homework_prob = self.config['homework_assigned_prob']
        rate = self.config['rate_per_hour']
        astro_min = self.config['duration_astronomical_hours_min']
        astro_max = self.config['duration_astronomical_hours_max']
        
        lessons = []
        student_ids = self.students_df['№ клиента'].tolist()
        lesson_dates = []
        
        for _ in range(n_lessons):
            student_id = random.choice(student_ids)
            lesson_date = self._random_date(start_date, end_date)
            lesson_dates.append(lesson_date)
            
            expulsion_date = self.expulsion_map.get(student_id)
            if expulsion_date and lesson_date >= expulsion_date:
                student_status = 'отчислен'
            else:
                student_status = 'учится'
            
            subject = random.choice(subjects)
            topic = random.choice(self.topics_corpus[subject])
            
            attendance = random.choices(list(att_weights.keys()), weights=att_weights.values())[0]
            lesson_status = random.choices(list(lesson_status_weights.keys()), weights=lesson_status_weights.values())[0]
            
            astro_hours = round(random.uniform(astro_min, astro_max), 2)
            acad_hours = round(astro_hours * 60 / 45, 2)
            revenue = round(rate * astro_hours, 2)
            
            homework_assigned = 'yes' if random.random() < homework_prob else 'no'
            
            if attendance == 'пришел' and lesson_status == 'проведен':
                lesson_grade = random.randint(2, 5)
                behavior_grade = random.randint(2, 5)
            else:
                lesson_grade = ''
                behavior_grade = ''
            
            row = {
                'Статус': lesson_status,
                'Предмет': subject,
                'Преподаватель': teacher,
                'Тема': topic,
                'Описание': random.choice(['', 'Урок прошел хорошо', 'Было сложно', '']),
                'Присутствие': attendance,
                'Ученик': student_id,
                'Статус ученика': student_status,
                'Продолж. ас/ч': astro_hours,
                'Продолж. ак/ч': acad_hours,
                'Ставка': rate,
                'Доходность': revenue,
                'Задано дз': homework_assigned,
                'Оценка за урок': lesson_grade,
                'Оценка за поведение': behavior_grade,
            }
            lessons.append(row)
        
        self.lessons_df = pd.DataFrame(lessons)
        self.lessons_df['date'] = lesson_dates
    
    def _generate_homeworks(self):
        homeworks = []
        homework_id_counter = 100000
        for idx, lesson in self.lessons_df.iterrows():
            if lesson['Задано дз'] == 'yes':
                assign_date = lesson['date']
                submit_date = assign_date + timedelta(days=random.randint(1, 14))
                score = random.randint(2, 5)
                row = {
                    'ДЗ': homework_id_counter + idx,
                    'Ученик': lesson['Ученик'],
                    'Дата выдачи': assign_date,
                    'Дата сдачи': submit_date,
                    'Оценка': score,
                    'Тема': lesson['Тема'],
                }
                homeworks.append(row)
        self.homeworks_df = pd.DataFrame(homeworks)
    
    def _update_student_stats(self):
        attended = self.lessons_df[self.lessons_df['Присутствие'] == 'пришел']
        stats = attended.groupby('Ученик').agg(
            visits_count=('Ученик', 'size'),
            first_visit=('date', 'min'),
            last_visit=('date', 'max')
        ).reset_index()
        
        for _, row in stats.iterrows():
            student_id = row['Ученик']
            mask = self.students_df['№ клиента'] == student_id
            if mask.any():
                self.students_df.loc[mask, 'Посещений'] = row['visits_count']
                self.students_df.loc[mask, 'Дата перв посещ кл'] = row['first_visit']
                self.students_df.loc[mask, 'Дата посл посещ кл'] = row['last_visit']
                if pd.notnull(row['first_visit']) and pd.notnull(row['last_visit']):
                    duration = (row['last_visit'] - row['first_visit']).days
                    self.students_df.loc[mask, 'Срок посещ, дн'] = duration
    
    def _save_to_xls(self):
        lessons_export = self.lessons_df.drop(columns=['date'], errors='ignore')
        lessons_export = lessons_export.reindex(columns=self.lessons_columns, fill_value='')
        self.students_df = self.students_df.reindex(columns=self.student_columns, fill_value='')
        self.homeworks_df = self.homeworks_df.reindex(columns=self.homeworks_columns, fill_value='')
        
        with pd.ExcelWriter(self.output_dir / 'student_data.xls', engine='openpyxl') as writer:
            self.students_df.to_excel(writer, index=False, sheet_name='student_data')
        with pd.ExcelWriter(self.output_dir / 'lessons.xls', engine='openpyxl') as writer:
            lessons_export.to_excel(writer, index=False, sheet_name='lessons')
        with pd.ExcelWriter(self.output_dir / 'homeworks.xls', engine='openpyxl') as writer:
            self.homeworks_df.to_excel(writer, index=False, sheet_name='homeworks')
    
    def _random_date(self, start_date: date, end_date: date) -> date:
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)

if __name__ == '__main__':
    generator = DataGenerator('./data_generator/data_generator_config.yaml')
    generator.generate()