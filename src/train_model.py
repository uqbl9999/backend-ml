"""
Training Script
Run this script to train the model from scratch
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation import DataPreparation
from src.models.training import ModelTrainer


def main(data_path: str, model_type: str = 'gradient_boosting', optimize: bool = True):
    """
    Main training pipeline

    Parameters:
    -----------
    data_path : str
        Path to the raw data CSV file
    model_type : str
        Type of model: 'gradient_boosting' or 'random_forest'
    optimize : bool
        Whether to perform hyperparameter optimization
    """
    print("="*80)
    print("MENTAL HEALTH SCREENING PREDICTION - MODEL TRAINING")
    print("="*80)

    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA PREPARATION")
    print("="*80)

    data_prep = DataPreparation(data_path)
    X_balanced, y_balanced = data_prep.prepare_full_pipeline(save_intermediate=True)

    print(f"\n✅ Data preparation completed!")
    print(f"   Final dataset: {X_balanced.shape[0]:,} rows x {X_balanced.shape[1]} columns")

    # =========================================================================
    # PHASE 2: MODEL TRAINING
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: MODEL TRAINING")
    print("="*80)

    trainer = ModelTrainer(model_type=model_type)

    # Split data
    trainer.split_data(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Train base model
    trainer.train_base_model()

    # Optimize hyperparameters
    if optimize:
        trainer.optimize_hyperparameters(n_iter=30, cv=5)

    # =========================================================================
    # PHASE 3: SAVE MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: SAVE MODEL")
    print("="*80)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save model
    model_filename = f'models/trained_model.pkl'
    trainer.save_model(model_filename)

    # =========================================================================
    # PHASE 4: GENERATE EVALUATION PLOTS
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: GENERATE EVALUATION PLOTS")
    print("="*80)

    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)

    # Generate plots
    trainer.plot_results(save_dir='docs')

    # =========================================================================
    # SUMMARY
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
    parser.add_argument('--model', type=str, default='gradient_boosting',
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
