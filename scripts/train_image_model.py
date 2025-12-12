#!/usr/bin/env python
"""
Script de entrenamiento para modelo de reconocimiento de imágenes médicas.

Uso:
    python scripts/train_image_model.py --epochs 50 --batch-size 32

Este script entrena un modelo CNN para clasificar rayos X en:
- COVID19
- NORMAL
- PNEUMONIA
- TUBERCULOSIS
"""

import sys
import os
import argparse

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.image_training import ImageModelTrainer


def main():
    """Función principal de entrenamiento."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo CNN para clasificación de rayos X'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/images',
        help='Directorio con datos (train/val/test). Default: data/images'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas de entrenamiento. Default: 50'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño del batch. Default: 32'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0005,
        help='Tasa de aprendizaje. Default: 0.0005'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models/image_models/best_model.keras',
        help='Ruta de salida del modelo. Default: models/image_models/best_model.keras'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO CNN - CLASIFICACIÓN DE RAYOS X")
    print("=" * 70)
    print(f"\nParámetros:")
    print(f"  Data directory:  {args.data_dir}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Output model:    {args.output}")
    print()

    # Verificar que exista el directorio de datos
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Directorio de datos no encontrado: {args.data_dir}")
        print("\nAsegúrate de que el directorio existe y contiene:")
        print("  - data/images/train/")
        print("  - data/images/val/")
        print("  - data/images/test/")
        print("\nCada subdirectorio debe contener carpetas para cada clase:")
        print("  - COVID19/")
        print("  - NORMAL/")
        print("  - PNEUMONIA/")
        print("  - TUBERCULOSIS/")
        sys.exit(1)

    # Verificar subdirectorios
    required_dirs = ['train', 'val', 'test']
    missing_dirs = []
    for dir_name in required_dirs:
        full_path = os.path.join(args.data_dir, dir_name)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"❌ Error: Faltan subdirectorios: {', '.join(missing_dirs)}")
        print(f"\nEl directorio {args.data_dir} debe contener:")
        for dir_name in required_dirs:
            print(f"  - {dir_name}/")
        sys.exit(1)

    try:
        # Crear trainer
        trainer = ImageModelTrainer(args.data_dir, image_size=(224, 224))

        # Cargar datasets
        trainer.load_datasets(batch_size=args.batch_size)

        # Construir modelo
        trainer.build_model(learning_rate=args.learning_rate)

        # Entrenar
        trainer.train(epochs=args.epochs, save_path=args.output)

        # Evaluar
        metrics = trainer.evaluate()

        # Guardar modelo y metadata
        trainer.save_model(args.output, metrics=metrics)

        # Generar visualizaciones
        output_dir = os.path.dirname(args.output)
        trainer.plot_training_history(
            save_path=os.path.join(output_dir, 'training_history.png')
        )

        trainer.plot_confusion_matrix(
            cm=metrics['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )

        print("\n" + "=" * 70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"\n📊 Resultados:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"  Test Loss:     {metrics['test_loss']:.4f}")
        print(f"\n📁 Archivos generados:")
        print(f"  - Modelo:             {args.output}")
        print(f"  - Metadata:           {output_dir}/model_metadata.json")
        print(f"  - Training history:   {output_dir}/training_history.png")
        print(f"  - Confusion matrix:   {output_dir}/confusion_matrix.png")
        print(f"\n🚀 Siguiente paso: Copiar el modelo a producción y hacer deploy")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
