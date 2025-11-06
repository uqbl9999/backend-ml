"""
Script de Entrenamiento
Ejecuta este script para entrenar el modelo desde cero
"""

import sys
import os
import argparse

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation import DataPreparation
from src.models.training import ModelTrainer


def main(data_path: str, model_type: str = 'random_forest', optimize: bool = True):
    """
    Flujo principal de entrenamiento

    Parámetros:
    -----------
    data_path : str
        Ruta al archivo CSV de datos crudos
    model_type : str
        Tipo de modelo: 'gradient_boosting' o 'random_forest'
    optimize : bool
        Si se debe realizar optimización de hiperparámetros
    """
    print("="*80)
    print("MENTAL HEALTH SCREENING PREDICTION - MODEL TRAINING")
    print("="*80)

    # =========================================================================
    # FASE 1: PREPARACIÓN DE DATOS
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA PREPARATION")
    print("="*80)

    data_prep = DataPreparation(data_path)
    X_balanced, y_balanced = data_prep.prepare_full_pipeline(save_intermediate=True)

    print(f"\n✅ Data preparation completed!")
    print(f"   Final dataset: {X_balanced.shape[0]:,} rows x {X_balanced.shape[1]} columns")

    # =========================================================================
    # FASE 2: ENTRENAMIENTO DEL MODELO
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: MODEL TRAINING")
    print("="*80)

    trainer = ModelTrainer(model_type=model_type)

    # Dividir datos
    trainer.split_data(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Entrenar modelo base
    trainer.train_base_model()

    # Optimizar hiperparámetros
    if optimize:
        trainer.optimize_hyperparameters(n_iter=30, cv=5)

    # =========================================================================
    # FASE 3: GUARDAR MODELO
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: SAVE MODEL")
    print("="*80)

    # Crear el directorio 'models' si no existe
    os.makedirs('models', exist_ok=True)

    # Guardar modelo
    model_filename = f'models/trained_model.pkl'
    trainer.save_model(model_filename)

    # =========================================================================
    # FASE 4: GENERAR GRÁFICOS DE EVALUACIÓN
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: GENERATE EVALUATION PLOTS")
    print("="*80)

    # Crear el directorio 'docs' si no existe
    os.makedirs('docs', exist_ok=True)

    # Generar gráficos
    trainer.plot_results(save_dir='docs')

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print("\n📊 Summary:")
    print(f"   Model type: {model_type}")
    print(f"   Training samples: {trainer.X_train.shape[0]:,}")
    print(f"   Test samples: {trainer.X_test.shape[0]:,}")
    print(f"   Features: {len(trainer.feature_names)}")

    if 'optimized_test' in trainer.metrics:
        test_metrics = trainer.metrics['optimized_test']
        print(f"\n📈 Test Set Performance:")
        print(f"   R² Score: {test_metrics['R2']:.4f}")
        print(f"   MAE: {test_metrics['MAE']:.4f}%")
        print(f"   RMSE: {test_metrics['RMSE']:.4f}%")

    print(f"\n💾 Files saved:")
    print(f"   Model: {model_filename}")
    print(f"   Plots: docs/evaluation_*.png")
    print(f"   Data: data/dataset_*.csv")

    print("\n✅ Ready to use! Start the API with: uvicorn api.main:app --reload")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train mental health screening prediction model')
    parser.add_argument('--data', type=str, default='tamizajes.csv',
                       help='Path to raw data CSV file')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['gradient_boosting', 'random_forest'],
                       help='Type of model to train')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Skip hyperparameter optimization (faster training)')

    args = parser.parse_args()

    main(
        data_path=args.data,
        model_type=args.model,
        optimize=not args.no_optimize
    )
