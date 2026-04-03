import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier
import joblib
import os
import yaml
import json

class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.models = {}
        self.results = {}
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame, target: str = 'academic_risk'):
        """Подготовка данных для обучения"""
        exclude_cols = self.config['features'].get('exclude_columns', [])
        extra_exclude = [
            'student_id', 'full_name', 'manager_id', 'source', 'first_payment_date',
            'last_payment_date', 'first_visit_date', 'last_visit_date', 'student_status',
            'parent_info', 'comments', 'subscribed_to_newsletter', 'no_auto_notifications', 'age_group'
        ]
        exclude_cols = list(set(exclude_cols + extra_exclude))
        candidate_cols = [c for c in df.columns if c not in exclude_cols and c != target]
        numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].copy()
        y = df[target].astype(int).copy()
        self.feature_names = numeric_cols
        test_size = self.config['models']['test_size']
        random_state = self.config['models']['random_state']
        
        # Если в одном из классов меньше 2 образцов, стратификацию не используем
        from collections import Counter
        counts = Counter(y)
        min_class_size = min(counts.values())
        stratify = y if min_class_size >= 2 else None
        if stratify is None:
            print(f"Внимание: один из классов имеет {min_class_size} образец(ов). Стратификация отключена.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))   # <-- замена liblinear на lbfgs
        ])
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
        gs = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        gs.fit(X_train, y_train)
        self.models['logreg'] = gs.best_estimator_
        print(f"Логистическая регрессия: ROC-AUC (CV) = {gs.best_score_:.3f}")
        return gs.best_estimator_

    def train_catboost(self, X_train, y_train):
        model = CatBoostClassifier(
            iterations=self.config['models']['catboost_iterations'],
            depth=self.config['models']['catboost_depth'],
            learning_rate=self.config['models']['catboost_learning_rate'],
            random_seed=self.config['models']['random_state'],
            verbose=False,
            cat_features=[]  # все признаки числовые
        )
        model.fit(X_train, y_train)
        self.models['catboost'] = model
        return model

    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.predict(X_test)
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
        self.results[model_name] = metrics
        print(f"\nРезультаты {model_name} на тестовой выборке:")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        return metrics

    def save_models(self):
        models_dir = self.config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f"{models_dir}/{name}_model.pkl")
        # Сохраняем названия признаков
        with open(f"{models_dir}/feature_names.json", 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False)
        print(f"Модели сохранены в {models_dir}")

    def load_models(self):
        models_dir = self.config['paths']['models_dir']
        for name in ['logreg', 'catboost']:
            path = f"{models_dir}/{name}_model.pkl"
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Модель {name} загружена из {path}")
        feature_path = f"{models_dir}/feature_names.json"
        if os.path.exists(feature_path):
            with open(feature_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
        return len(self.models) > 0

    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        if model_name not in self.models:
            return pd.DataFrame()
        model = self.models[model_name]
        if model_name == 'logreg':
            try:
                clf = model.named_steps['clf']
                importance = np.abs(clf.coef_.ravel())
                coef = clf.coef_.ravel()
            except:
                return pd.DataFrame()
        elif model_name == 'catboost':
            importance = model.get_feature_importance()
            coef = importance  # для CatBoost нет коэффициентов, только важность
        else:
            return pd.DataFrame()
        fi_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'coefficient': coef if model_name=='logreg' else np.nan
        }).sort_values('importance', ascending=False)
        return fi_df