import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
import sys
from datetime import datetime

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.dirname(__file__))
from src.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Система прогнозирования риска неуспеваемости учеников", layout="wide")

st.title("Система прогнозирования риска неуспеваемости учеников")
st.markdown("---")

# -------------------------------------------------------------------
# 1. Загрузка моделей и признаков
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    path_lr = "models/logreg_model.pkl"
    if os.path.exists(path_lr):
        try:
            models['logreg'] = joblib.load(path_lr)
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки логистической регрессии: {e}")
    path_cb = "models/catboost_model.pkl"
    if os.path.exists(path_cb):
        try:
            models['catboost'] = joblib.load(path_cb)
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки CatBoost: {e}")
    return models

@st.cache_data
def load_feature_names():
    path = "models/feature_names.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

models = load_models()
feature_names = load_feature_names()

# -------------------------------------------------------------------
# 2. Загрузка маппинга колонок из templates
# -------------------------------------------------------------------
def load_mapping_from_templates():
    """Загружает маппинг {русское: английское} из JSON-файлов в ./data_generator/templates"""
    base_dir = "./data_generator/templates"
    mappings = {}
    for fname in ["student_data.json", "lessons.json", "homeworks.json"]:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Преобразуем список словарей в словарь
                mapping = {item["russian_name"]: item["english_name"] for item in data}
                mappings[fname.replace(".json", "")] = mapping
        else:
            st.error(f"Файл маппинга не найден: {path}")
            st.stop()
    return mappings

mappings = load_mapping_from_templates()

def rename_by_mapping(df, mapping):
    """Переименовывает колонки DataFrame согласно маппингу {русское: английское}"""
    rename_dict = {ru: en for ru, en in mapping.items() if ru in df.columns}
    return df.rename(columns=rename_dict)

# -------------------------------------------------------------------
# 3. Вспомогательные функции для предобработки
# -------------------------------------------------------------------
def preprocess_dates(df, date_columns):
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def clean_loaded_data(students_df, lessons_df, homeworks_df):
    """Очистка загруженных данных с использованием маппинга"""
    students_df = rename_by_mapping(students_df, mappings['student_data'])
    lessons_df = rename_by_mapping(lessons_df, mappings['lessons'])
    homeworks_df = rename_by_mapping(homeworks_df, mappings['homeworks'])
    
    # Преобразование дат
    student_date_cols = ['first_payment_date', 'last_payment_date', 'first_visit_date', 'last_visit_date', 'birth_date']
    lessons_date_cols = ['date']
    homeworks_date_cols = ['assign_date', 'submit_date']
    students_df = preprocess_dates(students_df, student_date_cols)
    lessons_df = preprocess_dates(lessons_df, lessons_date_cols)
    homeworks_df = preprocess_dates(homeworks_df, homeworks_date_cols)
    
    # Числовые колонки
    numeric_student = ['visits_count', 'balance', 'total_payments', 'debt', 'visit_duration_days', 'age']
    for col in numeric_student:
        if col in students_df.columns:
            students_df[col] = pd.to_numeric(students_df[col], errors='coerce')
    
    # attendance_flag из lessons
    if 'attendance' in lessons_df.columns:
        lessons_df['attendance_flag'] = lessons_df['attendance'].apply(lambda x: 1 if x == 'пришел' else 0)
    
    # Оценки
    if 'lesson_grade' in lessons_df.columns:
        lessons_df['lesson_grade'] = pd.to_numeric(lessons_df['lesson_grade'], errors='coerce')
    if 'score' in homeworks_df.columns:
        homeworks_df['score'] = pd.to_numeric(homeworks_df['score'], errors='coerce')
    
    # Длительность уроков
    if 'duration_academic_hours' in lessons_df.columns:
        lessons_df['duration_academic_hours'] = pd.to_numeric(lessons_df['duration_academic_hours'], errors='coerce')
    
    # Бинарные признаки учеников
    for col in ['subscribed_to_newsletter', 'no_auto_notifications']:
        if col in students_df.columns:
            students_df[col] = students_df[col].map({'да': 1, 'нет': 0}).fillna(0).astype(int)
    
    return students_df, lessons_df, homeworks_df

# -------------------------------------------------------------------
# 4. Функция для построения гистограмм
# -------------------------------------------------------------------
def plot_histogram(data, xlabel, ylabel="Количество учеников", title=None, rot=45):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(data.index.astype(str), data.values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=rot, ha='right')
    st.pyplot(fig)

# -------------------------------------------------------------------
# 5. Основная логика приложения
# -------------------------------------------------------------------
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.probs = None
    st.session_state.raw_students = None
    st.session_state.raw_lessons = None
    st.session_state.raw_homeworks = None
    st.session_state.selected_student_index = 0  # для навигации

# Если данные ещё не загружены, показываем форму загрузки
if st.session_state.data is None:
    st.subheader("Загрузка данных из CRM")
    st.info("Загрузите три файла: **student_data.xls** (ученики), **lessons.xls** (занятия) и **homeworks.xls** (домашние задания).")

    export_file = st.file_uploader("Файл student_data (ученики)", type=['csv', 'xlsx', 'xls'], key='export')
    schedule_file = st.file_uploader("Файл lessons (занятия)", type=['csv', 'xlsx', 'xls'], key='schedule')
    homeworks_file = st.file_uploader("Файл homeworks (домашние задания)", type=['csv', 'xlsx', 'xls'], key='homeworks')

    if export_file is not None and schedule_file is not None and homeworks_file is not None:
        try:
            def read_file(file_obj):
                if file_obj.name.endswith('.csv'):
                    return pd.read_csv(file_obj, encoding='utf-8')
                else:
                    return pd.read_excel(file_obj)

            students_df = read_file(export_file)
            lessons_df = read_file(schedule_file)
            homeworks_df = read_file(homeworks_file)

            students_df, lessons_df, homeworks_df = clean_loaded_data(students_df, lessons_df, homeworks_df)

            st.success(f"Файлы загружены: {len(students_df)} учеников, {len(lessons_df)} занятий, {len(homeworks_df)} ДЗ")
            st.session_state.raw_students = students_df
            st.session_state.raw_lessons = lessons_df
            st.session_state.raw_homeworks = homeworks_df

            with st.spinner("Генерация признаков..."):
                engineer = FeatureEngineer()
                features_df = engineer.create_features(students_df, lessons_df, homeworks_df)

            # Выбор модели для прогноза (по умолчанию логистическая регрессия)
            selected_model_name = st.sidebar.selectbox("Модель для прогноза", ["logreg", "catboost"], index=0)
            if selected_model_name in models and models[selected_model_name] is not None:
                model = models[selected_model_name]
                if feature_names:
                    missing = [f for f in feature_names if f not in features_df.columns]
                    if missing:
                        st.error(f"Отсутствуют признаки, необходимые модели: {missing}")
                        st.stop()
                    X = features_df[feature_names].fillna(0)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)[:, 1]
                    else:
                        proba = model.predict(X)
                    features_df['risk'] = (proba >= 0.5).astype(int)
                    features_df['risk_prob'] = proba
                else:
                    features_df['risk'] = 0
                    features_df['risk_prob'] = 0.0
            else:
                st.warning(f"Модель {selected_model_name} не загружена. Используем логистическую регрессию, если доступна.")
                if 'logreg' in models and models['logreg'] is not None:
                    model = models['logreg']
                    if feature_names:
                        X = features_df[feature_names].fillna(0)
                        proba = model.predict_proba(X)[:, 1]
                        features_df['risk'] = (proba >= 0.5).astype(int)
                        features_df['risk_prob'] = proba
                    else:
                        features_df['risk'] = 0
                        features_df['risk_prob'] = 0.0
                else:
                    features_df['risk'] = 0
                    features_df['risk_prob'] = 0.0

            st.session_state.data = features_df
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка при обработке файлов: {e}")
            st.stop()
    else:
        st.info("Пожалуйста, загрузите все три файла для продолжения.")
        st.stop()

# -------------------------------------------------------------------
# 6. Отображение результатов
# -------------------------------------------------------------------
data = st.session_state.data

# Боковая панель с фильтрами и выбором модели
with st.sidebar:
    st.header("Фильтры")
    risk_filter = st.selectbox("Риск", ["Все", "Высокий риск (1)", "Низкий риск (0)"])
    if risk_filter == "Высокий риск (1)":
        filtered_data = data[data['risk'] == 1]
    elif risk_filter == "Низкий риск (0)":
        filtered_data = data[data['risk'] == 0]
    else:
        filtered_data = data

    st.header("Статистика")
    total_students = len(filtered_data)
    risk_count = filtered_data['risk'].sum() if 'risk' in filtered_data else 0
    st.metric("Всего учеников", total_students)
    st.metric("Высокий риск", int(risk_count))
    st.metric("Низкий риск", total_students - int(risk_count))
    if total_students > 0:
        st.metric("Доля риска", f"{risk_count/total_students*100:.1f}%")
    else:
        st.metric("Доля риска", "0%")

    st.header("Модель")
    model_choice = st.selectbox("Модель для анализа важности", ["logreg", "catboost"], index=0)

# Основная область
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Данные учеников")
    display_cols = ['student_id', 'full_name', 'age', 'attendance_rate', 'avg_lesson_grade', 'avg_hw_score', 'risk', 'risk_prob']
    available_display = [c for c in display_cols if c in filtered_data.columns]
    if not available_display:
        available_display = filtered_data.columns.tolist()
    st.dataframe(filtered_data[available_display].head(100), use_container_width=True)
    st.write(f"Всего учеников: {len(filtered_data)}")

with col2:
    st.subheader("Распределение риска")
    if 'risk' in filtered_data:
        risk_dist = filtered_data['risk'].value_counts().sort_index()
        risk_dist.index = risk_dist.index.map({0: "Низкий риск (0)", 1: "Высокий риск (1)"})
        plot_histogram(risk_dist, xlabel="Категория риска", ylabel="Количество учеников", title="Распределение риска", rot=0)
        st.caption("0 – низкий риск, 1 – высокий риск")

# Графики распределения показателей
st.subheader("Распределение ключевых показателей")
if 'attendance_rate' in filtered_data:
    att = filtered_data['attendance_rate'] * 100
    bins = np.linspace(0, 100, 21)
    counts = pd.cut(att, bins=bins).value_counts().sort_index()
    plot_histogram(counts, xlabel="Посещаемость, %", ylabel="Количество учеников", title="Распределение посещаемости", rot=45)

if 'avg_lesson_grade' in filtered_data:
    grades = filtered_data['avg_lesson_grade'].dropna()
    bins = np.arange(0, 5.5, 0.5)
    counts = pd.cut(grades, bins=bins).value_counts().sort_index()
    plot_histogram(counts, xlabel="Средняя оценка за урок", ylabel="Количество учеников", title="Распределение средних оценок", rot=45)

if 'avg_hw_score' in filtered_data:
    hw = filtered_data['avg_hw_score'].dropna()
    bins = np.arange(0, 5.5, 0.5)
    counts = pd.cut(hw, bins=bins).value_counts().sort_index()
    plot_histogram(counts, xlabel="Средняя оценка за ДЗ", ylabel="Количество учеников", title="Распределение оценок за ДЗ", rot=45)

# -------------------------------------------------------------------
# Прогнозирование для конкретного ученика (с навигацией по реальным ID)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Прогнозирование для конкретного ученика")

if not models or (model_choice not in models or models[model_choice] is None):
    st.warning("Модель не загружена. Запустите main.py для обучения.")
    st.stop()

# Пороговые значения для цветовой индикации
THRESHOLDS = {
    'attendance_rate': {'good': 0.75, 'warning': 0.5},
    'attendance_trend': {'good': 0.0001, 'warning': -0.0009},
}

if 'student_id' in filtered_data.columns:
    # Получаем отсортированный список уникальных ID
    student_ids = sorted(filtered_data['student_id'].unique())
    if not student_ids:
        st.error("Нет учеников для отображения")
        st.stop()
    
    # Индекс текущего выбранного ID в сессии
    if 'selected_student_index' not in st.session_state:
        st.session_state.selected_student_index = 0
    
    # Кнопки навигации
    col_select = st.columns([1, 1, 1])
    
    current_id = student_ids[st.session_state.selected_student_index]
    st.write(f"**Текущий ID:** {current_id}")
    
    # Можно также позволить выбрать ID из списка
    selected_id = st.selectbox(
        "Выберите ID ученика из списка",
        options=student_ids,
        index=st.session_state.selected_student_index,
        key="student_id_selector"
    )
    if selected_id != current_id:
        st.session_state.selected_student_index = student_ids.index(selected_id)
        st.rerun()
    
    student_data = filtered_data[filtered_data['student_id'] == current_id]
else:
    st.error("Нет колонки student_id")
    st.stop()

if len(student_data) > 0:
    student = student_data.iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**ФИО:** {student.get('full_name', 'N/A')}")
        st.info(f"**Возраст:** {student.get('age', 'N/A')} лет")
    with col2:
        att_rate = student.get('attendance_rate', 0)
        if att_rate < THRESHOLDS['attendance_rate']['warning']:
            st.error(f"**Посещаемость:** {att_rate*100:.1f}%")
        elif att_rate < THRESHOLDS['attendance_rate']['good']:
            st.warning(f"**Посещаемость:** {att_rate*100:.1f}%")
        else:
            st.info(f"**Посещаемость:** {att_rate*100:.1f}%")
        
        avg_grade = student.get('avg_lesson_grade', 0)
        if avg_grade < 3:
            st.error(f"**Средняя оценка за урок:** {avg_grade:.1f}")
        elif avg_grade < 4:
            st.warning(f"**Средняя оценка за урок:** {avg_grade:.1f}")
        else:
            st.info(f"**Средняя оценка за урок:** {avg_grade:.1f}")

    if st.button("Сделать прогноз"):
        prob = student.get('risk_prob', 0)
        risk = student.get('risk', 0)
        if risk == 1:
            st.error(f"**ВЫСОКИЙ РИСК** (вероятность {prob*100:.1f}%)")
        else:
            st.success(f"**НИЗКИЙ РИСК** (вероятность {prob*100:.1f}%)")
        
        st.write("**Рекомендации:**")
        recommendations = []
        if att_rate < 0.5:
            recommendations.append("Низкая посещаемость – необходимо усилить контроль и мотивацию.")
        elif att_rate < 0.75:
            recommendations.append("Посещаемость ниже нормы – рекомендуется связаться с родителями.")
        if avg_grade < 2.5:
            recommendations.append("Средний балл критически низкий – необходима срочная работа с учеником.")
        elif avg_grade < 3.5:
            recommendations.append("Средний балл ниже нормы – дополнительные занятия.")
        if not recommendations:
            recommendations.append("Показатели в норме, продолжайте в том же духе.")
        for rec in recommendations:
            st.write(rec)
else:
    st.warning("Ученик с таким ID не найден")

# -------------------------------------------------------------------
# Важность признаков для выбранной модели
# -------------------------------------------------------------------
st.markdown("---")
st.subheader(f"Важность признаков ({model_choice})")

if model_choice in models and models[model_choice] is not None:
    try:
        if model_choice == 'logreg':
            clf = models[model_choice].named_steps['clf']
            coef = clf.coef_.ravel()
            importance = np.abs(coef)
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.round(coef, 5),
                'importance': np.round(importance, 5)
            }).sort_values('importance', ascending=False)
        else:  # catboost
            importance = models[model_choice].get_feature_importance()
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.round(importance, 5)
            }).sort_values('importance', ascending=False)
        st.bar_chart(fi_df.set_index('feature')['importance'])
        st.dataframe(fi_df)
    except Exception as e:
        st.write(f"Не удалось получить важность признаков: {e}")
else:
    st.warning(f"Модель {model_choice} не загружена")

st.markdown("---")
st.caption("Дипломная работа | НГТУ | 2026 | Шевашкевич Ю.Д.")