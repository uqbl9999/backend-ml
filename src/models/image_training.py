"""
Módulo de entrenamiento para modelo CNN de clasificación de rayos X.

Este módulo proporciona la clase ImageModelTrainer para entrenar modelos CNN
que clasifican imágenes de rayos X en: COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class ImageModelTrainer:
    """
    Clase para entrenar modelo CNN de clasificación de rayos X.

    Gestiona carga de datos, construcción del modelo, entrenamiento,
    evaluación y guardado de resultados.
    """

    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Inicializar el entrenador.

        Parameters:
        -----------
        data_dir : str
            Directorio raíz con subdirectorios train/val/test.
        image_size : tuple
            Tamaño de las imágenes (altura, ancho).
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_classes = 4
        self.class_names = None
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def load_datasets(self, batch_size: int = 32) -> Tuple:
        """
        Cargar datasets de entrenamiento, validación y test.

        Parameters:
        -----------
        batch_size : int
            Tamaño del batch.

        Returns:
        --------
        tuple
            (train_generator, val_generator, test_generator)
        """
        # Data augmentation para training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Solo rescale para validación y test
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Train generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        # Validation generator
        self.val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Test generator
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Guardar nombres de clases
        self.class_names = list(self.train_generator.class_indices.keys())

        print(f"✅ Datasets cargados:")
        print(f"   Train: {self.train_generator.samples} imágenes")
        print(f"   Val:   {self.val_generator.samples} imágenes")
        print(f"   Test:  {self.test_generator.samples} imágenes")
        print(f"   Clases: {self.class_names}")

        return self.train_generator, self.val_generator, self.test_generator

    def build_model(self, learning_rate: float = 0.0005) -> tf.keras.Model:
        """
        Construir arquitectura CNN.

        Parameters:
        -----------
        learning_rate : float
            Tasa de aprendizaje para el optimizador Adam.

        Returns:
        --------
        tf.keras.Model
            Modelo compilado.
        """
        model = Sequential([
            # Bloque 1
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Bloque 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Bloque 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Bloque 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Clasificador
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compilar
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("✅ Modelo construido")
        model.summary()

        return model

    def train(
        self,
        epochs: int = 50,
        save_path: str = 'models/image_models/best_model.keras'
    ):
        """
        Entrenar el modelo con callbacks.

        Parameters:
        -----------
        epochs : int
            Número de épocas.
        save_path : str
            Ruta para guardar el mejor modelo.
        """
        if self.model is None:
            raise ValueError("Modelo no construido. Llama primero a build_model()")

        if self.train_generator is None:
            raise ValueError("Datasets no cargados. Llama primero a load_datasets()")

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("\n🚀 Iniciando entrenamiento...")
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )

        print("\n✅ Entrenamiento completado")

    def evaluate(self) -> Dict:
        """
        Evaluar el modelo en el test set.

        Returns:
        --------
        dict
            Métricas de evaluación.
        """
        if self.model is None:
            raise ValueError("Modelo no disponible")

        if self.test_generator is None:
            raise ValueError("Test generator no disponible")

        print("\n📊 Evaluando modelo en test set...")

        # Evaluar
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)

        # Predicciones para classification report
        self.test_generator.reset()
        y_pred_probs = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.test_generator.classes

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Extraer métricas por clase
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1": report[class_name]["f1-score"]
            }

        metrics = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist()
        }

        print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        return metrics

    def save_model(self, model_path: str, metrics: Dict = None):
        """
        Guardar modelo y metadata.

        Parameters:
        -----------
        model_path : str
            Ruta para guardar el modelo (.keras).
        metrics : dict
            Métricas de evaluación.
        """
        if self.model is None:
            raise ValueError("Modelo no disponible")

        # Guardar modelo
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"\n✅ Modelo guardado en: {model_path}")

        # Guardar metadata
        if metrics:
            metadata_path = os.path.join(
                os.path.dirname(model_path),
                'model_metadata.json'
            )

            metadata = {
                "model_type": "CNN - Convolutional Neural Network",
                "framework": "TensorFlow/Keras",
                "version": tf.__version__,
                "input_shape": [*self.image_size, 3],
                "num_classes": self.num_classes,
                "classes": self.class_names,
                "architecture": {
                    "total_params": int(self.model.count_params()),
                    "trainable_params": int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
                    "layers": len(self.model.layers)
                },
                "training_info": {
                    "epochs": len(self.history.history['loss']),
                    "batch_size": self.train_generator.batch_size,
                    "optimizer": "Adam",
                    "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
                    "loss": "categorical_crossentropy"
                },
                "metrics": metrics,
                "class_descriptions": {
                    "COVID19": "Neumonía viral causada por SARS-CoV-2. Patrones típicos: opacidades en vidrio esmerilado bilateral.",
                    "NORMAL": "Radiografía de tórax sin hallazgos patológicos significativos.",
                    "PNEUMONIA": "Infección pulmonar bacteriana o viral. Consolidaciones y opacidades.",
                    "TUBERCULOSIS": "Infección por Mycobacterium tuberculosis. Cavitaciones, nódulos, infiltrados."
                }
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            print(f"✅ Metadata guardada en: {metadata_path}")

    def plot_training_history(self, save_path: str = 'training_history.png'):
        """
        Visualizar accuracy y loss durante entrenamiento.

        Parameters:
        -----------
        save_path : str
            Ruta para guardar el gráfico.
        """
        if self.history is None:
            raise ValueError("No hay historial de entrenamiento")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Gráfico de entrenamiento guardado en: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = 'confusion_matrix.png'):
        """
        Visualizar matriz de confusión.

        Parameters:
        -----------
        cm : np.ndarray
            Matriz de confusión.
        save_path : str
            Ruta para guardar el gráfico.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Matriz de confusión guardada en: {save_path}")
        plt.close()
