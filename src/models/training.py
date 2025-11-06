"""
Módulo de Entrenamiento de Modelos
Gestiona el entrenamiento del modelo, la optimización de hiperparámetros y la evaluación
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Clase para gestionar el entrenamiento y la evaluación del modelo"""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Inicializa el entrenador de modelos

        Parámetros:
        -----------
        model_type : str
            Tipo de modelo a usar: 'gradient_boosting' o 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.metrics = {}

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Dividir los datos en conjuntos de entrenamiento y prueba

        Parámetros:
        -----------
        X : pd.DataFrame
            Características (features)
        y : pd.Series
            Variable objetivo
        test_size : float
            Proporción del conjunto de prueba
        random_state : int
            Semilla aleatoria
        """
        print("\nDividiendo datos...")

        # Eliminar 'Anio' si está presente (valor constante, sin poder predictivo)
        if 'Anio' in X.columns:
            X = X.drop(columns=['Anio'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.feature_names = X.columns.tolist()

        print(f"✅ Conjunto de entrenamiento: {self.X_train.shape[0]:,} filas")
        print(f"✅ Conjunto de prueba: {self.X_test.shape[0]:,} filas")
        print(f"✅ Características: {len(self.feature_names)}")

    def train_base_model(self):
        """Entrenar un modelo base sin optimización de hiperparámetros"""
        print(f"\nEntrenando modelo base {self.model_type}...")

        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                oob_score=True,
                verbose=0
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError(f"Tipo de modelo desconocido: {self.model_type}")

        self.model.fit(self.X_train, self.y_train)
        print("✅ Entrenamiento del modelo base completado")

        # Evaluate base model
        self._evaluate_model(prefix="base")

    def optimize_hyperparameters(self, n_iter: int = 30, cv: int = 5):
        """
        Optimizar hiperparámetros usando RandomizedSearchCV

        Parámetros:
        -----------
        n_iter : int
            Número de combinaciones de parámetros a probar
        cv : int
            Número de particiones para validación cruzada
        """
        print(f"\nOptimizando hiperparámetros para {self.model_type}...")
        print(f"  n_iter={n_iter}, cv={cv}")
        print("  Esto puede tomar varios minutos...")

        if self.model_type == 'random_forest':
            param_distributions = {
                'n_estimators': randint(100, 300),
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', 0.5, 0.7],
                'max_samples': [0.7, 0.8, 0.9, None],
                'bootstrap': [True]
            }
            base_estimator = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)

        elif self.model_type == 'gradient_boosting':
            param_distributions = {
                'n_estimators': randint(100, 300),
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', 0.5, 0.7]
            }
            base_estimator = GradientBoostingRegressor(random_state=42)

        else:
            raise ValueError(f"Tipo de modelo desconocido: {self.model_type}")

        # Búsqueda aleatoria
        random_search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        random_search.fit(self.X_train, self.y_train)

        print("\n✅ ¡Optimización completada!")
        print("\n📊 Mejores hiperparámetros:")
        for param, value in random_search.best_params_.items():
            print(f"  • {param}: {value}")

        print(f"\n✅ Mejor puntaje R² en CV: {random_search.best_score_:.4f}")

        # Usar el mejor modelo
        self.model = random_search.best_estimator_
        print("\n🔄 Entrenando modelo final con los mejores hiperparámetros...")
        self.model.fit(self.X_train, self.y_train)
        print("✅ Entrenamiento del modelo optimizado completado")

        # Evaluar modelo optimizado
        self._evaluate_model(prefix="optimized")

    def _evaluate_model(self, prefix: str = ""):
        """
        Evaluar el rendimiento del modelo en los conjuntos de entrenamiento y prueba

        Parámetros:
        -----------
        prefix : str
            Prefijo para las claves de las métricas
        """
        # Predicciones
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Métricas - Entrenamiento
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.y_train, y_pred_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)

        # Métricas - Pruebas
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.y_test, y_pred_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)

        # Guardar métricas
        self.metrics[f'{prefix}_train'] = {
            'R2': r2_train,
            'MAE': mae_train,
            'MSE': mse_train,
            'RMSE': rmse_train
        }
        self.metrics[f'{prefix}_test'] = {
            'R2': r2_test,
            'MAE': mae_test,
            'MSE': mse_test,
            'RMSE': rmse_test
        }

        # Imprimir resultados
        print("\n" + "="*60)
        print(f"📊 MODELO {prefix.upper()} - CONJUNTO DE ENTRENAMIENTO")
        print("="*60)
        print(f"Puntaje R²:                         {r2_train:.4f}")
        print(f"MAE (Error Absoluto Medio):         {mae_train:.4f}%")
        print(f"MSE (Error Cuadrático Medio):       {mse_train:.4f}")
        print(f"RMSE (Raíz del ECM):                {rmse_train:.4f}%")

        print("\n" + "="*60)
        print(f"📊 MODELO {prefix.upper()} - CONJUNTO DE PRUEBA")
        print("="*60)
        print(f"Puntaje R²:                         {r2_test:.4f}")
        print(f"MAE (Error Absoluto Medio):         {mae_test:.4f}%")
        print(f"MSE (Error Cuadrático Medio):       {mse_test:.4f}")
        print(f"RMSE (Raíz del ECM):                {rmse_test:.4f}%")

        # Análisis de sobreajuste
        print("\n" + "="*60)
        print("🔍 ANÁLISIS DE SOBREAJUSTE")
        print("="*60)
        diff_r2 = r2_train - r2_test
        diff_mae = mae_test - mae_train

        print(f"R² diferencia (Train - Test):       {diff_r2:.4f}")
        print(f"Diferencia MAE (Test - Train):      {diff_mae:.4f}%")

        if diff_r2 > 0.13:
            print("⚠️  Posible overfitting detectado (R² diferencia > 0.1)")
        else:
            print("✅ El modelo generaliza bien")
        print("="*60)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Obtener la importancia de características del modelo entrenado

        Parámetros:
        -----------
        top_n : int
            Número de características principales a devolver

        Retorna:
        --------
        pd.DataFrame : DataFrame de importancia de características
        """
        if self.model is None:
            raise ValueError("El modelo aún no ha sido entrenado")

        importances = self.model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False
        ).head(top_n)

        return feature_importance_df

    def save_model(self, filepath: str):
        """
        Guardar el modelo entrenado en disco

        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("El modelo aún no ha sido entrenado")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Modelo guardado: {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """
        Cargar un modelo entrenado desde disco

        Parámetros:
        -----------
        filepath : str
            Ruta del modelo guardado

        Retorna:
        --------
        dict : Datos del modelo incluyendo el modelo, nombres de características y métricas
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        print(f"✅ Modelo cargado: {filepath}")
        print(f"  Tipo de modelo: {model_data['model_type']}")
        print(f"  Características: {len(model_data['feature_names'])}")

        return model_data

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones usando el modelo entrenado

        Parámetros:
        -----------
        X : pd.DataFrame
            Características para la predicción

        Retorna:
        --------
        np.ndarray : Predicciones
        """
        if self.model is None:
            raise ValueError("El modelo aún no ha sido entrenado")

        # Asegurar que las características coinciden con los datos de entrenamiento
        if set(X.columns) != set(self.feature_names):
            raise ValueError("Las características no coinciden con los datos de entrenamiento")

        # Reordenar columnas para coincidir con los datos de entrenamiento
        X = X[self.feature_names]

        return self.model.predict(X)

    def plot_results(self, save_dir: str = 'docs'):
        """
        Generar y guardar gráficos de evaluación

        Parámetros:
        -----------
        save_dir : str
            Directorio donde guardar los gráficos
        """
        if self.model is None:
            raise ValueError("El modelo aún no ha sido entrenado")

        # Predicciones
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # 1. Real vs Predicho
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Conjunto de entrenamiento
        axes[0].scatter(self.y_train, y_pred_train, alpha=0.5, s=10, color='steelblue')
        axes[0].plot([self.y_train.min(), self.y_train.max()],
                    [self.y_train.min(), self.y_train.max()],
                    'r--', lw=2, label='Predicción perfecta')
        axes[0].set_xlabel('Valores reales', fontweight='bold')
        axes[0].set_ylabel('Predicciones', fontweight='bold')
        axes[0].set_title(f'Conjunto de Entrenamiento (R² = {self.metrics.get("optimized_train", {}).get("R2", 0):.4f})',
                         fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Conjunto de prueba
        axes[1].scatter(self.y_test, y_pred_test, alpha=0.5, s=10, color='darkorange')
        axes[1].plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2, label='Predicción perfecta')
        axes[1].set_xlabel('Valores reales', fontweight='bold')
        axes[1].set_ylabel('Predicciones', fontweight='bold')
        axes[1].set_title(f'Conjunto de Prueba (R² = {self.metrics.get("optimized_test", {}).get("R2", 0):.4f})',
                         fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Importancia de características
        feature_importance_df = self.get_feature_importance(top_n=15)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance_df)),
                feature_importance_df['Importance'],
                color='darkorange', alpha=0.8)
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
        plt.xlabel('Importancia', fontweight='bold')
        plt.title(f'Top 15 Características Más Importantes ({self.model_type})',
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Gráficos guardados en el directorio '{save_dir}/'")
