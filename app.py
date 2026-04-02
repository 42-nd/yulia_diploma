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
from src.preprocessing import DataPreprocessor

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Система прогнозирования риска неуспеваемости учеников", layout="wide")

st.title("Система прогнозирования риска неуспеваемости учеников")
st.markdown("---")

# -------------------------------------------------------------------
# 1. Загрузка модели и признаков
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    path = "models/logreg_model.pkl"
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки модели: {e}")
            return None
    return None

@st.cache_data
def load_feature_names():
    path = "models/feature_names.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

model = load_model()
feature_names = load_feature_names()

# -------------------------------------------------------------------
# 2. Вспомогательные функции для преобразования имён колонок
# -------------------------------------------------------------------
def rename_export_columns(df):
    """Приводит названия колонок export.csv к английским именам, ожидаемым в FeatureEngineer"""
    mapping = {
        '№ клиента': 'client_id',
        'ФИО': 'full_name',
        'Возраст': 'age',
        'Дата рождения': 'birth_date',
        'Телефон': 'phone',
        'Email': 'email',
        'Отв. менеджер': 'responsible_manager',
        'Источник клиента': 'source',
        'Создан': 'created_date',
        'Записей': 'total_enrollments',
        'Посещений': 'total_attendances',
        'Дата перв посещ кл': 'first_attendance_date',
        'Дата посл посещ кл': 'last_attendance_date',
        'Срок посещ, дн': 'attendance_days_span',
        'Срок жизни кл от созд до посл посещ, дн': 'lifetime_from_creation_to_last_attendance',
        'Статус ученика': 'student_status',
        'Филиал ученика': 'branch',
        'Родитель (имя, телефон)': 'parent_info',
        'Подписан на рассылку': 'subscribed_to_newsletter',
        'VK': 'vk',
        'Не присылать автоуведомления': 'no_auto_notifications'
    }
    return df.rename(columns=mapping)

def rename_schedule_columns(df):
    """Приводит названия колонок schedule.csv к английским именам, ожидаемым в FeatureEngineer"""
    mapping = {
        'Дата': 'date',
        'Время': 'time',
        'Филиал': 'branch',
        'Программа': 'program',
        'Группа': 'group',
        'Преподаватель': 'teacher',
        'Тема': 'topic',
        'Записано': 'enrolled',
        'Пришло': 'attended',
        'Пропусков всего': 'total_absences',
        'Пропусков по уваж. причине': 'excused_absences',
        'Ученик': 'student_name',
        'Статус записи в группу': 'enrollment_status',
        'Посещение': 'attendance_flag',
        'Продолж. ас/ч': 'duration_academic_hours',
        'Продолж. ак/ч': 'duration_clock_hours',
        'Доходность': 'revenue'
    }
    return df.rename(columns=mapping)

def preprocess_dates(df, date_columns):
    """Преобразует строковые даты в datetime"""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def clean_data(students_df, lessons_df):
    """Выполняет предобработку: преобразование дат, типов, создание служебных полей"""
    # Преобразуем даты
    students_df = preprocess_dates(students_df, [
        'created_date', 'first_payment_date', 'last_payment_date',
        'first_attendance_date', 'last_attendance_date'
    ])
    lessons_df = preprocess_dates(lessons_df, ['date'])

    # Убедимся, что client_id и student_name совместимы
    if 'client_id' not in students_df.columns:
        students_df['client_id'] = students_df.index + 1
    if 'client_id' not in lessons_df.columns:
        # Пытаемся связать по имени ученика
        if 'student_name' in lessons_df.columns and 'full_name' in students_df.columns:
            # Создаём словарь соответствия имени и client_id
            name_to_id = dict(zip(students_df['full_name'], students_df['client_id']))
            lessons_df['client_id'] = lessons_df['student_name'].map(name_to_id)

    # Удаляем строки, где не удалось привязать client_id
    lessons_df = lessons_df.dropna(subset=['client_id'])
    lessons_df['client_id'] = lessons_df['client_id'].astype(int)

    # Преобразуем attendance_flag в число, если он в строке
    if 'attendance_flag' in lessons_df.columns:
        lessons_df['attendance_flag'] = lessons_df['attendance_flag'].astype(str).map({'1': 1, '0': 0, 'Да': 1, 'Нет': 0}).fillna(0).astype(int)
    # Добавим служебные поля (они нужны для create_features)
    # Если нет diligence, base_attendance, risk_score — они не обязательны, но create_features может их удалить
    # Для совместимости добавим их с нулями
    for col in ['diligence', 'base_attendance', 'risk_score']:
        if col not in students_df.columns:
            students_df[col] = 0.0

    return students_df, lessons_df

# -------------------------------------------------------------------
# Вспомогательная функция для построения гистограмм с подписями осей
# -------------------------------------------------------------------
def plot_histogram(data, xlabel, ylabel="Количество учеников", title=None, rot=45):
    """Рисует столбчатую диаграмму с подписанными осями и выводит через st.pyplot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(data.index.astype(str), data.values)  # индексы в строки для читаемости
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=rot, ha='right')
    st.pyplot(fig)

# -------------------------------------------------------------------
# 3. Основная логика приложения
# -------------------------------------------------------------------
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.probs = None
    st.session_state.raw_students = None
    st.session_state.raw_lessons = None

# Если данные ещё не загружены, показываем форму загрузки
if st.session_state.data is None:
    st.subheader("Загрузка данных из CRM")
    st.info("Загрузите два файла: **export.csv** (данные учеников) и **schedule.csv** (расписание занятий).")

    export_file = st.file_uploader("Файл export.csv (ученики)", type=['csv', 'xlsx'], key='export')
    schedule_file = st.file_uploader("Файл schedule.csv (занятия)", type=['csv', 'xlsx'], key='schedule')
    homeworks_file = st.file_uploader("Файл homeworks.csv (домашние задания, опционально)", type=['csv', 'xlsx'], key='homeworks')
    if export_file is not None and schedule_file is not None and homeworks_file is not None:
        try:
            # Чтение файлов
            if export_file.name.endswith('.csv'):
                export_df = pd.read_csv(export_file, encoding='utf-8')
            else:
                export_df = pd.read_excel(export_file)

            if schedule_file.name.endswith('.csv'):
                schedule_df = pd.read_csv(schedule_file, encoding='utf-8')
            else:
                schedule_df = pd.read_excel(schedule_file)
            if homeworks_file.name.endswith('.csv'):
                homeworks_df = pd.read_csv(homeworks_file, encoding='utf-8')
            else:
                homeworks_df = pd.read_excel(homeworks_file)
            homeworks_df['client_id'] = homeworks_df['client_id'].astype(int)
            homeworks_df['score'] = pd.to_numeric(homeworks_df['score'], errors='coerce')
            homeworks_df['date'] = pd.to_datetime(homeworks_df['date'], errors='coerce')
            # Переименование колонок
            export_df = rename_export_columns(export_df)
            schedule_df = rename_schedule_columns(schedule_df)

            # Предобработка
            export_df, schedule_df = clean_data(export_df, schedule_df)

            st.success(f"Файлы загружены: {len(export_df)} учеников, {len(schedule_df)} занятий")
            st.session_state.raw_students = export_df
            st.session_state.raw_lessons = schedule_df

            # Генерация признаков
            with st.spinner("Генерация признаков..."):
                engineer = FeatureEngineer()
                features_df = engineer.create_features(export_df, schedule_df, homeworks_df)

            # Применение модели для расчёта риска, если модель доступна
            if model is not None and feature_names:
                # Проверяем наличие всех необходимых признаков
                missing = [f for f in feature_names if f not in features_df.columns]
                if missing:
                    st.error(f"Отсутствуют признаки, необходимые модели: {missing}")
                    st.stop()
                X = features_df[feature_names].fillna(0)
                proba = model.predict_proba(X)[:, 1]
                features_df['risk'] = (proba >= 0.5).astype(int)
                features_df['risk_prob'] = proba
            else:
                features_df['risk'] = 0
                features_df['risk_prob'] = 0.0

            st.session_state.data = features_df
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка при обработке файлов: {e}")
            st.stop()
    else:
        st.info("Пожалуйста, загрузите оба файла для продолжения.")
        st.stop()

# -------------------------------------------------------------------
# 4. Отображение результатов
# -------------------------------------------------------------------
data = st.session_state.data

# Боковая панель с фильтрами
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
    risk_count = filtered_data['risk'].sum()
    st.metric("Всего учеников", total_students)
    st.metric("Высокий риск", int(risk_count))
    st.metric("Низкий риск", total_students - int(risk_count))
    st.metric("Доля риска", f"{risk_count/total_students*100:.1f}%" if total_students>0 else "0%")

    # Средние показатели
    st.write("Средние показатели:")
    avg_cols = {
        'attendance_rate': 'Посещаемость, %',
        'avg_score': 'Средний балл',
    }
    for col, label in avg_cols.items():
        if col in filtered_data.columns:
            val = filtered_data[col].mean()
            if 'rate' in col:
                st.write(f"- {label}: {val*100:.1f}%")
            else:
                st.write(f"- {label}: {val:,.0f}")

# Основная область
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Данные учеников")
    # Колонки для отображения
    display_cols = ['client_id', 'full_name', 'age', 'branch', 'attendance_rate', 'avg_score', 'risk', 'risk_prob']
    available_display = [c for c in display_cols if c in filtered_data.columns]
    if not available_display:
        available_display = filtered_data.columns.tolist()
    st.dataframe(filtered_data[available_display].head(100), use_container_width=True)
    st.write(f"Всего учеников: {len(filtered_data)}")

with col2:
    st.subheader("Распределение риска")
    risk_dist = filtered_data['risk'].value_counts().sort_index()
    # Переименовываем индексы для наглядности
    risk_dist.index = risk_dist.index.map({0: "Низкий риск (0)", 1: "Высокий риск (1)"})
    plot_histogram(risk_dist, xlabel="Категория риска", ylabel="Количество учеников", title="Распределение риска", rot=0)
    st.caption("0 – низкий риск, 1 – высокий риск")

st.subheader("Распределение ключевых показателей")
plot_config = {
    'attendance_rate': {'label': 'Посещаемость, %', 'mult': 100}
}

for col, cfg in plot_config.items():
    if col in filtered_data.columns:
        data_plot = filtered_data[col] * cfg['mult']
        # Строим гистограмму с 20 бинами
        counts = data_plot.value_counts(bins=20).sort_index()
        # Форматируем метки оси X
        labels = [f"{interval.right:.1f}%" for interval in counts.index]
        counts.index = labels
        plot_histogram(counts, xlabel=cfg['label'], ylabel="Количество учеников", title=f"Распределение {cfg['label']}", rot=45)
        st.caption(f"Распределение {cfg['label']}")
    else:
        st.info(f"Показатель {col} отсутствует в данных")

if 'avg_score' in filtered_data.columns:
    st.subheader("Распределение среднего балла")
    scores = filtered_data['avg_score']
    # Группируем по 0.5 балла
    bins = np.arange(0, 5.5, 0.5)
    labels = [f"{b:.1f}–{b+0.5:.1f}" for b in bins[:-1]]
    binned = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    counts = binned.value_counts().sort_index()
    plot_histogram(counts, xlabel="Средний балл (интервалы)", ylabel="Количество учеников", title="Распределение среднего балла", rot=45)
    st.caption("Распределение среднего балла (0–5)")

# Гистограмма тренда оценок
if 'score_trend' in filtered_data.columns:
    st.subheader("Распределение тренда оценок")
    trend_vals = filtered_data['score_trend']
    counts = trend_vals.value_counts(bins=20).sort_index()
    # Форматируем метки для наглядности
    labels = [f"{interval.left:.3f}–{interval.right:.3f}" for interval in counts.index]
    counts.index = labels
    plot_histogram(counts, xlabel="Тренд оценок (изменение в день)", ylabel="Количество учеников", title="Распределение тренда оценок", rot=45)
    st.caption("Распределение тренда оценок (изменение в день)")

# -------------------------------------------------------------------
# Прогнозирование для конкретного ученика (с цветными плашками)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Прогнозирование для конкретного ученика")

if model is None:
    st.warning("Модель не найдена. Запустите main.py для обучения.")
    st.stop()

# Пороговые значения для цветовой индикации
THRESHOLDS = { 
    'attendance_rate': {'good': 0.75, 'warning': 0.5},
    'attendance_trend': {'good': 0.0001, 'warning': -0.0009},
}

# Определяем min/max client_id
if 'client_id' in filtered_data.columns:
    min_id = int(filtered_data['client_id'].min())
    max_id = int(filtered_data['client_id'].max())
else:
    st.error("Нет колонки client_id")
    st.stop()

student_id = st.number_input(
    "ID ученика (client_id)",
    min_value=min_id,
    max_value=max_id,
    value=min_id,
    step=1
)

student_data = filtered_data[filtered_data['client_id'] == student_id]

if len(student_data) > 0:
    student = student_data.iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**ФИО:** {student.get('full_name', 'N/A')}")
        st.info(f"**Возраст:** {student.get('age', 'N/A')} лет")
        if 'grade' in student:
            st.info(f"**Класс:** {student.get('grade', 'N/A')}")
        if 'subject' in student:
            st.info(f"**Предмет:** {student.get('subject', 'N/A')}")

    with col2:
        st.subheader("Активность")
        att_rate = student.get('attendance_rate', 0)
        if att_rate < THRESHOLDS['attendance_rate']['warning']:
            st.error(f"**Посещаемость:** {att_rate*100:.1f}%")
        elif att_rate < THRESHOLDS['attendance_rate']['good']:
            st.warning(f"**Посещаемость:** {att_rate*100:.1f}%")
        else:
            st.info(f"**Посещаемость:** {att_rate*100:.1f}%")

        if 'avg_lesson_attendance' in student:
            avg_att = student['avg_lesson_attendance']
            if avg_att < THRESHOLDS['attendance_rate']['warning']:
                st.error(f"**Средняя посещаемость (урок):** {avg_att*100:.1f}%")
            elif avg_att < THRESHOLDS['attendance_rate']['good']:
                st.warning(f"**Средняя посещаемость (урок):** {avg_att*100:.1f}%")
            else:
                st.info(f"**Средняя посещаемость (урок):** {avg_att*100:.1f}%")

        if 'attendance_trend' in student:
            trend = student['attendance_trend']
            if abs(trend) < 1e-12: 
                trend = 0.0
            if trend > THRESHOLDS['attendance_trend']['good']:
                st.info(f"**Тренд посещаемости:** {trend:.3f}")
            elif trend > THRESHOLDS['attendance_trend']['warning']:
                st.warning(f"**Тренд посещаемости:** {trend:.3f}")
            else:
                st.error(f"**Тренд посещаемости:** {trend:.3f}")

        if 'subscribed' in student:
            if student['subscribed']:
                st.info(f"**Подписан на рассылку:** Да")
            else:
                st.warning(f"**Подписан на рассылку:** Нет")
        if 'avg_score' in student:
            avg_score = student['avg_score']
            if avg_score < 2.5:
                st.error(f"**Средний балл:** {avg_score:.1f} (высокий риск)")
            elif avg_score < 3.5:
                st.warning(f"**Средний балл:** {avg_score:.1f} (средний риск)")
            else:
                st.info(f"**Средний балл:** {avg_score:.1f} (норма)")

        if 'score_trend' in student:
            trend_score = student['score_trend']
            if trend_score < -0.01:
                st.error(f"**Тренд оценок:** {trend_score:.3f} (ухудшается)")
            elif trend_score < 0:
                st.warning(f"**Тренд оценок:** {trend_score:.3f} (слабое ухудшение)")
            else:
                st.info(f"**Тренд оценок:** {trend_score:.3f} (стабильно/растёт)")

    # Кнопка прогноза и рекомендации
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
        if 'avg_score' in student:
            if avg_score < 2.5:
                recommendations.append("Средний балл критически низкий – необходима срочная работа с учеником.")
            elif avg_score < 3.5:
                recommendations.append("Средний балл ниже нормы – дополнительные занятия.")
        if 'score_trend' in student:
            if trend_score < -0.01:
                recommendations.append("Заметное падение успеваемости – выяснить причины.")
        if not recommendations:
            recommendations.append("Показатели в норме, продолжайте в том же духе.")
        for rec in recommendations:
            st.write(rec)
else:
    st.warning("Ученик с таким ID не найден")

st.markdown("---")
st.subheader("Важность признаков (Логистическая регрессия)")

if model is not None and len(feature_names) > 0:
    try:
        clf = model.named_steps['clf']
        coefficients = clf.coef_.ravel()
        importance = np.abs(coefficients)
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': np.round(coefficients, 5),
            'importance': np.round(importance, 5)
        }).sort_values('importance', ascending=False)
        st.bar_chart(fi_df.set_index('feature')['importance'])
        st.write("Коэффициенты модели (чем выше по модулю, тем сильнее влияние):")
        st.dataframe(fi_df)
    except Exception as e:
        st.write(f"Не удалось получить коэффициенты: {e}")
else:
    st.warning("Модель или признаки не загружены")

st.markdown("---")
st.caption("Дипломная работа | НГТУ | 2025 | Шевашкевич Ю.Д.")