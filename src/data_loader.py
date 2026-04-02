import pandas as pd
from typing import Tuple
import os

class ExcelDataLoader:
    """
    Загрузка данных из Excel-файла для прогнозирования.
    Поддерживает формат с 4 листами: Students, Lessons, Homeworks, Communications
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загрузка всех таблиц из Excel"""
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Файл {self.filepath} не найден")
        
        students = pd.read_excel(self.filepath, sheet_name='Students')
        lessons = pd.read_excel(self.filepath, sheet_name='Lessons')
        homeworks = pd.read_excel(self.filepath, sheet_name='Homeworks')
        communications = pd.read_excel(self.filepath, sheet_name='Communications')
        
        # Конвертация дат
        if 'date' in lessons.columns:
            lessons['date'] = pd.to_datetime(lessons['date'], errors='coerce')
        if 'date' in communications.columns:
            communications['date'] = pd.to_datetime(communications['date'], errors='coerce')
        
        return students, lessons, homeworks, communications
    
    def validate(self, students, lessons, homeworks, communications) -> dict:
        validation = {
            'students_count': len(students),
            'lessons_count': len(lessons),
            'homeworks_count': len(homeworks),
            'communications_count': len(communications),
            'valid': True,
            'errors': []
        }
        
        if validation['students_count'] < 10:
            validation['valid'] = False
            validation['errors'].append('Недостаточно данных учеников (минимум 10)')
        
        if 'student_id' not in students.columns:
            validation['valid'] = False
            validation['errors'].append('Отсутствует колонка student_id')
        
        if 'academic_risk' not in students.columns:
            validation['valid'] = False
            validation['errors'].append('Отсутствует целевая переменная academic_risk')
        
        return validation


if __name__ == "__main__":
    loader = ExcelDataLoader('data/student_data.xlsx')
    students, lessons, homeworks, comms = loader.load()
    validation = loader.validate(students, lessons, homeworks, comms)
    print(validation)