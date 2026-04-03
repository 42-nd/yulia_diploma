import os
import pandas as pd
import warnings
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.utils import Visualizer, print_results_table
import numpy as np
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("ДИПЛОМНЫЙ ПРОЕКТ: ПРОГНОЗИРОВАНИЕ РИСКА НЕУСПЕВАЕМОСТИ УЧЕНИКА")
    print("="*80)

    # 1. Загрузка и предобработка
    print("\nШАГ 1: Загрузка и предобработка данных...")
    preprocessor = DataPreprocessor()
    students, lessons, homeworks = preprocessor.get_cleaned_data()
    print(f"Загружено: учеников={len(students)}, уроков={len(lessons)}, ДЗ={len(homeworks)}")

    # 2. Сохранение очищенных таблиц в data_export
    export_dir = preprocessor.config['paths']['data_export_dir']
    os.makedirs(export_dir, exist_ok=True)
    students.to_csv(os.path.join(export_dir, "student_data.csv"), index=False, encoding='utf-8')
    lessons.to_csv(os.path.join(export_dir, "lessons.csv"), index=False, encoding='utf-8')
    homeworks.to_csv(os.path.join(export_dir, "homeworks.csv"), index=False, encoding='utf-8')
    print(f"Очищенные таблицы сохранены в {export_dir}")

    # 3. Генерация признаков
    print("\nШАГ 2: Генерация признаков...")
    engineer = FeatureEngineer()
    features_df = engineer.create_features(students, lessons, homeworks)
    selected_features, full_df = engineer.select_features(features_df, n_features=10)
    print(f"Сформировано признаков: {features_df.shape[1]}")

    # 4. Подготовка к обучению
    print("\nШАГ 3: Обучение моделей...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(full_df)

    # Логистическая регрессия
    trainer.train_logistic_regression(X_train, y_train)
    trainer.evaluate_model(trainer.models['logreg'], X_test, y_test, 'logreg')

    # CatBoost
    trainer.train_catboost(X_train, y_train)
    trainer.evaluate_model(trainer.models['catboost'], X_test, y_test, 'catboost')

    # 5. Сохранение моделей
    trainer.save_models()

    # 6. Визуализация
    visualizer = Visualizer()
    print("\nШАГ 4: Визуализация...")
    numeric_for_corr = full_df.select_dtypes(include='number')
    visualizer.plot_correlation_heatmap(numeric_for_corr, "correlation_heatmap")

    for model_name in ['logreg', 'catboost']:
        fi_df = trainer.get_feature_importance(model_name)
        if not fi_df.empty:
            visualizer.plot_feature_importance(fi_df, model_name, top_n=10)
            print(f"\nВажность признаков ({model_name}):")
            print(fi_df.head(10).to_string(index=False))
        cm = trainer.results[model_name]['confusion_matrix']
        visualizer.plot_confusion_matrix(cm, model_name)

    # 7. Сохранение метрик в txt
    metrics_path = os.path.join(visualizer.reports_dir, "model_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("Сравнение моделей\n")
        f.write("="*80 + "\n")
        f.write(f"{'Модель':<20} {'ROC-AUC':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-"*80 + "\n")
        for name, metrics in trainer.results.items():
            roc = metrics['roc_auc']
            if roc is None or np.isnan(roc):
                roc = 0.0
            f.write(f"{name:<20} {roc:<12.3f} {metrics['accuracy']:<12.3f} "
                    f"{metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1']:<12.3f}\n")
        f.write("="*80 + "\n")
    print(f"Метрики сохранены в {metrics_path}")

    # 8. Прогнозы для всех учеников
    print("\nШАГ 5: Формирование итоговой таблицы с прогнозами...")
    X_full = full_df[trainer.feature_names].copy()
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_full_imputed = imputer.fit_transform(X_full)
    preds_logreg = trainer.models['logreg'].predict_proba(X_full_imputed)[:, 1]
    preds_catboost = trainer.models['catboost'].predict_proba(X_full_imputed)[:, 1]
    
    # Оставляем только нужные колонки: student_id, target, признаки, прогнозы
    keep_cols = ['student_id', 'academic_risk'] + trainer.feature_names
    output_df = full_df[keep_cols].copy()
    output_df['pred_risk_logreg'] = preds_logreg
    output_df['pred_risk_catboost'] = preds_catboost
    
    # Округление float до 4 знаков
    for col in output_df.select_dtypes(include='float').columns:
        output_df[col] = output_df[col].round(4)
    
    output_df.to_csv(os.path.join(export_dir, "features_with_predictions.csv"), index=False, encoding='utf-8')
    print(f"Итоговая таблица сохранена: features_with_predictions.csv")

    # 9. Итоговая таблица результатов
    print_results_table(trainer.results)

    print("\n" + "="*80)
    print("ПРОЕКТ УСПЕШНО ЗАВЕРШЁН!")
    print("="*80)
    print("\nРезультаты сохранены в папках:")
    print(f"data_export/ - очищенные таблицы и прогнозы")
    print(f"models/ - обученные модели (logreg, catboost)")
    print(f"reports/ - графики и метрики")

if __name__ == "__main__":
    main()