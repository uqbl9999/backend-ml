"""
Model Training Module
Handles model training, hyperparameter tuning, and evaluation
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
    """Class to handle model training and evaluation"""

    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize the model trainer

        Parameters:
        -----------
        model_type : str
            Type of model to use: 'gradient_boosting' or 'random_forest'
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
        Split data into training and testing sets

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        test_size : float
            Proportion of test set
        random_state : int
            Random seed
        """
        print("\nSplitting data...")

        # Remove 'Anio' if present (constant value, no predictive power)
        if 'Anio' in X.columns:
            X = X.drop(columns=['Anio'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.feature_names = X.columns.tolist()

        print(f"✅ Training set: {self.X_train.shape[0]:,} rows")
        print(f"✅ Test set: {self.X_test.shape[0]:,} rows")
        print(f"✅ Features: {len(self.feature_names)}")

    def train_base_model(self):
        """Train a base model without hyperparameter tuning"""
        print(f"\nTraining base {self.model_type} model...")

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
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(self.X_train, self.y_train)
        print("✅ Base model training completed")

        # Evaluate base model
        self._evaluate_model(prefix="base")

    def optimize_hyperparameters(self, n_iter: int = 30, cv: int = 5):
        """
        Optimize hyperparameters using RandomizedSearchCV

        Parameters:
        -----------
        n_iter : int
            Number of parameter combinations to try
        cv : int
            Number of cross-validation folds
        """
        print(f"\nOptimizing hyperparameters for {self.model_type}...")
        print(f"  n_iter={n_iter}, cv={cv}")
        print("  This may take several minutes...")

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
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Random Search
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

        print("\n✅ Optimization completed!")
        print("\n📊 Best hyperparameters:")
        for param, value in random_search.best_params_.items():
            print(f"  • {param}: {value}")

        print(f"\n✅ Best CV R² score: {random_search.best_score_:.4f}")

        # Use the best model
        self.model = random_search.best_estimator_
        print("\n🔄 Training final model with best hyperparameters...")
        self.model.fit(self.X_train, self.y_train)
        print("✅ Optimized model training completed")

        # Evaluate optimized model
        self._evaluate_model(prefix="optimized")

    def _evaluate_model(self, prefix: str = ""):
        """
        Evaluate model performance on train and test sets

        Parameters:
        -----------
        prefix : str
            Prefix for metrics keys
        """
        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Metrics - Training
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.y_train, y_pred_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)

        # Metrics - Testing
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.y_test, y_pred_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)

        # Store metrics
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

        # Print results
        print("\n" + "="*60)
        print(f"📊 {prefix.upper()} MODEL - TRAINING SET")
        print("="*60)
        print(f"R² Score:                           {r2_train:.4f}")
        print(f"MAE (Mean Absolute Error):          {mae_train:.4f}%")
        print(f"MSE (Mean Squared Error):           {mse_train:.4f}")
        print(f"RMSE (Root Mean Squared Error):     {rmse_train:.4f}%")

        print("\n" + "="*60)
        print(f"📊 {prefix.upper()} MODEL - TEST SET")
        print("="*60)
        print(f"R² Score:                           {r2_test:.4f}")
        print(f"MAE (Mean Absolute Error):          {mae_test:.4f}%")
        print(f"MSE (Mean Squared Error):           {mse_test:.4f}")
        print(f"RMSE (Root Mean Squared Error):     {rmse_test:.4f}%")

        # Overfitting analysis
        print("\n" + "="*60)
        print("🔍 OVERFITTING ANALYSIS")
        print("="*60)
        diff_r2 = r2_train - r2_test
        diff_mae = mae_test - mae_train

        print(f"R² difference (Train - Test):       {diff_r2:.4f}")
        print(f"MAE difference (Test - Train):      {diff_mae:.4f}%")

        if diff_r2 > 0.1:
            print("⚠️  Possible overfitting detected (R² difference > 0.1)")
        else:
            print("✅ Model generalizes well")
        print("="*60)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from the trained model

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

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
        Save the trained model to disk

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Model saved: {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """
        Load a trained model from disk

        Parameters:
        -----------
        filepath : str
            Path to the saved model

        Returns:
        --------
        dict : Model data including model, feature names, and metrics
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        print(f"✅ Model loaded: {filepath}")
        print(f"  Model type: {model_data['model_type']}")
        print(f"  Features: {len(model_data['feature_names'])}")

        return model_data

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model

        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction

        Returns:
        --------
        np.ndarray : Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Ensure features match training data
        if set(X.columns) != set(self.feature_names):
            raise ValueError("Features don't match training data")

        # Reorder columns to match training data
        X = X[self.feature_names]

        return self.model.predict(X)

    def plot_results(self, save_dir: str = 'docs'):
        """
        Generate and save evaluation plots

        Parameters:
        -----------
        save_dir : str
            Directory to save plots
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # 1. Actual vs Predicted
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Training set
        axes[0].scatter(self.y_train, y_pred_train, alpha=0.5, s=10, color='steelblue')
        axes[0].plot([self.y_train.min(), self.y_train.max()],
                    [self.y_train.min(), self.y_train.max()],
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontweight='bold')
        axes[0].set_ylabel('Predictions', fontweight='bold')
        axes[0].set_title(f'Training Set (R² = {self.metrics.get("optimized_train", {}).get("R2", 0):.4f})',
                         fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Test set
        axes[1].scatter(self.y_test, y_pred_test, alpha=0.5, s=10, color='darkorange')
        axes[1].plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Values', fontweight='bold')
        axes[1].set_ylabel('Predictions', fontweight='bold')
        axes[1].set_title(f'Test Set (R² = {self.metrics.get("optimized_test", {}).get("R2", 0):.4f})',
                         fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Feature Importance
        feature_importance_df = self.get_feature_importance(top_n=15)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance_df)),
                feature_importance_df['Importance'],
                color='darkorange', alpha=0.8)
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
        plt.xlabel('Importance', fontweight='bold')
        plt.title(f'Top 15 Most Important Features ({self.model_type})',
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Plots saved to '{save_dir}/' directory")
