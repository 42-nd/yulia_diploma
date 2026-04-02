import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib
import os
import yaml

class ModelTrainer:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.results = {}
        self.feature_names = []
    
    def prepare_data(self, df: pd.DataFrame, target: str = 'academic_risk'):
        """Подготовка данных для обучения"""
        # Исключаемые колонки (идентификаторы, текстовые, даты)
        exclude_cols = self.config['features'].get('exclude_columns', [])
        # Добавляем все нечисловые колонки, которые могут быть в данных
        extra_exclude = [
            'client_id', 'full_name', 'responsible_manager', 'source',
            'created_date', 'first_attendance_date', 'last_attendance_date', 'student_status',
            'branch', 'parent_info', 'vk', 'subscribed_to_newsletter',
            'no_auto_notifications', 'age_group'
        ]
        exclude_cols = list(set(exclude_cols + extra_exclude))

        # Выбираем колонки, не входящие в исключения и не являющиеся целевой
        candidate_cols = [c for c in df.columns if c not in exclude_cols and c != target]
        # Оставляем только числовые колонки
        numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_cols].copy()
        y = df[target].astype(int).copy()

        self.feature_names = numeric_cols
        test_size = self.config['models']['test_size']
        random_state = self.config['models']['random_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, numeric_cols
    
    def train_logistic_regression(self, X_train, y_train):
        """Обучение логистической регрессии"""
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
        ])
        
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10]
        }
        
        gs = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        self.model = gs.best_estimator_
        print(f"Логистическая регрессия: ROC-AUC = {gs.best_score_:.3f}")
        
        return gs.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test) -> dict:
        """Оценка качества модели"""
        scores = model.predict_proba(X_test)[:, 1]
        preds = (scores >= 0.5).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, scores),
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, preds),
            'classification_report': classification_report(y_test, preds, zero_division=0, output_dict=True)
        }
        
        print(f"\nРезультаты на тестовой выборке:")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        
        self.results = metrics
        return metrics
    
    def save_model(self):
        """Сохранение обученной модели"""
        models_dir = self.config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        joblib.dump(self.model, f"{models_dir}/logreg_model.pkl")
        
        # Сохраняем названия признаков
        import json
        with open(f"{models_dir}/feature_names.json", 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False)
        
        print(f"Модель сохранена в {models_dir}/logreg_model.pkl")
    
    def load_model(self):
        """Загрузка обученной модели"""
        models_dir = self.config['paths']['models_dir']
        path = f"{models_dir}/logreg_model.pkl"
        
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Модель загружена из {path}")
            
            # Загружаем названия признаков
            feature_path = f"{models_dir}/feature_names.json"
            if os.path.exists(feature_path):
                import json
                with open(feature_path, 'r', encoding='utf-8') as f:
                    self.feature_names = json.load(f)
            
            return True
        return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Получение важности признаков (коэффициенты логистической регрессии)"""
        if self.model is None:
            return pd.DataFrame()
        
        try:
            clf = self.model.named_steps['clf']
            importance = np.abs(clf.coef_.ravel())
            
            fi_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance,
                'coefficient': clf.coef_.ravel()
            }).sort_values('importance', ascending=False)
            
            return fi_df
        except Exception as e:
            print(f"Ошибка получения важности признаков: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    preprocessor = DataPreprocessor()
    students, lessons, homeworks, comms = preprocessor.load_data()
    
    engineer = FeatureEngineer()
    merged = engineer.create_features(students, lessons, homeworks, comms)
    selected, reduced = engineer.select_features(merged)
    
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(reduced)
    
    trainer.train_logistic_regression(X_train, y_train)
    trainer.evaluate_model(trainer.model, X_test, y_test)
    trainer.save_model()
    
    # Важность признаков
    fi_df = trainer.get_feature_importance()
    print(f"\nВажность признаков:")
    print(fi_df.head(10))