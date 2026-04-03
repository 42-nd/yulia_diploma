from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.utils import Visualizer
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("ДИПЛОМНЫЙ ПРОЕКТ: ПРОГНОЗИРОВАНИЕ РИСКА НЕУСПЕВАЕМОСТИ УЧЕНИКА")
    print("="*80)

    print("\nШАГ 1: Предобработка данных...")
    preprocessor = DataPreprocessor()
    students, lessons, homeworks, _ = preprocessor.load_data()  # load_data теперь должна загружать и homeworks
    preprocessor.check_missing_values(students, lessons, pd.DataFrame(), pd.DataFrame())  # вместо None
    preprocessor.get_statistics(students, lessons, pd.DataFrame())                       # вместо None

    print("\nШАГ 2: Генерация признаков...")
    engineer = FeatureEngineer()
    merged = engineer.create_features(students, lessons, homeworks_df=homeworks)  # только students и lessons
    selected, reduced = engineer.select_features(merged, n_features=10)

    # Сохраняем обработанные данные
    reduced.to_csv('data/processed_students.csv', index=False, encoding='utf-8')
    print(f"Обработанные данные сохранены: {reduced.shape}")

    visualizer = Visualizer()
    visualizer.plot_correlation_heatmap(reduced, "correlation_heatmap")

    print("\nШАГ 3: Обучение модели (Логистическая регрессия)...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(reduced)

    trainer.train_logistic_regression(X_train, y_train)
    trainer.evaluate_model(trainer.model, X_test, y_test)

    print("\nШАГ 4: Важность признаков...")
    fi_df = trainer.get_feature_importance()
    if not fi_df.empty:
        visualizer.plot_feature_importance(fi_df, "logreg")
        print(fi_df.head(10))

    print("\nШАГ 6: Сохранение модели...")
    trainer.save_model()

    print("\n" + "="*80)
    print("ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
    print("="*80)
    print("\nРезультаты сохранены в папках:")
    print("data/ - данные")
    print("models/ - обученная модель")
    print("reports/ - графики и визуализации")
    print("\nДля запуска интерфейса выполните: streamlit run app.py")

if __name__ == "__main__":
    main()