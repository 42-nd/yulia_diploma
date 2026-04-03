import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Optional
import yaml

class Visualizer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.reports_dir = self.config['paths']['reports_dir']
        os.makedirs(self.reports_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12

    def plot_correlation_heatmap(self, df: pd.DataFrame, save_name: str = "correlation_heatmap"):
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
        plt.title('Корреляционная матрица признаков')
        plt.tight_layout()
        if self.config['visualization']['save_figures']:
            plt.savefig(f"{self.reports_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        if self.config['visualization']['show_figures']:
            plt.show()
        plt.close()

    def plot_feature_importance(self, fi_df: pd.DataFrame, model_name: str, top_n: int = 10, save_name: str = None):
        top_features = fi_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'].values[::-1])
        plt.yticks(range(len(top_features)), top_features['feature'].values[::-1])
        plt.xlabel('Важность')
        plt.title(f'Важность признаков - {model_name}')
        plt.tight_layout()
        if save_name is None:
            save_name = f"feature_importance_{model_name}"
        if self.config['visualization']['save_figures']:
            plt.savefig(f"{self.reports_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        if self.config['visualization']['show_figures']:
            plt.show()
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, save_name: str = None):
        if save_name is None:
            save_name = f"confusion_matrix_{model_name}"
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Предсказано')
        plt.ylabel('Фактически')
        plt.title(f'Матрица ошибок - {model_name}')
        plt.tight_layout()
        if self.config['visualization']['save_figures']:
            plt.savefig(f"{self.reports_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        if self.config['visualization']['show_figures']:
            plt.show()
        plt.close()

def print_results_table(results: dict):
    print("\n" + "="*80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*80)
    print(f"{'Модель':<20} {'ROC-AUC':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['roc_auc']:<12.3f} {metrics['accuracy']:<12.3f} "
              f"{metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1']:<12.3f}")
    print("="*80)