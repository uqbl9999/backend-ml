"""
Módulo de predicción con modelo CNN para clasificación de rayos X.

Este módulo gestiona la carga del modelo entrenado y realiza predicciones
sobre imágenes de rayos X para diagnosticar: COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

# Configurar TensorFlow para reducir logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.models.image_preprocessing import ImagePreprocessor


class ImagePredictor:
    """
    Clase para realizar predicciones con modelo CNN de clasificación de rayos X.

    Gestiona la carga del modelo, preprocesamiento de imágenes y generación
    de predicciones con interpretación de confianza.
    """

    def __init__(self, model_path: str):
        """
        Inicializar el predictor.

        Parameters:
        -----------
        model_path : str
            Ruta al archivo del modelo (.keras o .h5).
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        self.metadata = None
        self.preprocessor = ImagePreprocessor()

        # Cargar modelo y metadata
        self.load_model()

    def load_model(self):
        """
        Cargar modelo Keras y metadata asociada.

        Raises:
        -------
        FileNotFoundError
            Si el modelo o metadata no existen.
        Exception
            Si hay error al cargar el modelo.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        try:
            # Cargar modelo Keras
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Modelo CNN cargado desde: {self.model_path}")

            # Cargar metadata
            metadata_path = os.path.join(
                os.path.dirname(self.model_path),
                'model_metadata.json'
            )

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"✅ Metadata cargada desde: {metadata_path}")
            else:
                print(f"⚠️  Advertencia: Metadata no encontrada en {metadata_path}")
                self.metadata = self._create_default_metadata()

        except Exception as e:
            raise Exception(f"Error al cargar modelo: {str(e)}")

    def _create_default_metadata(self) -> Dict:
        """
        Crear metadata por defecto si no existe.

        Returns:
        --------
        dict
            Metadata básica del modelo.
        """
        return {
            "model_type": "CNN - Convolutional Neural Network",
            "framework": "TensorFlow/Keras",
            "input_shape": [224, 224, 3],
            "num_classes": 4,
            "classes": self.class_names,
            "class_descriptions": {
                "COVID19": "Neumonía viral causada por SARS-CoV-2. Patrones típicos: opacidades en vidrio esmerilado bilateral.",
                "NORMAL": "Radiografía de tórax sin hallazgos patológicos significativos.",
                "PNEUMONIA": "Infección pulmonar bacteriana o viral. Consolidaciones y opacidades.",
                "TUBERCULOSIS": "Infección por Mycobacterium tuberculosis. Cavitaciones, nódulos, infiltrados."
            }
        }

    def predict_from_file(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Realizar predicción desde archivo upload.

        Parameters:
        -----------
        file_bytes : bytes
            Bytes del archivo de imagen.
        filename : str
            Nombre del archivo.

        Returns:
        --------
        dict
            Resultado de la predicción.
        """
        start_time = time.time()

        # Cargar imagen desde bytes
        image = self.preprocessor.load_image_from_bytes(file_bytes)

        # Validar imagen
        validation = self.preprocessor.validate_image(image)
        if not validation['is_valid']:
            raise ValueError(f"Imagen inválida: {', '.join(validation['errors'])}")

        # Preprocesar
        img_array = self.preprocessor.preprocess_for_prediction(image)

        # Predicción
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Interpretar resultados
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Calcular tiempo de procesamiento
        processing_time_ms = (time.time() - start_time) * 1000

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'interpretation': self._interpret_prediction(predicted_class, confidence),
            'all_probabilities': {
                class_name: float(predictions[i])
                for i, class_name in enumerate(self.class_names)
            },
            'metadata': {
                'image_size': list(image.size),
                'processing_time_ms': round(processing_time_ms, 2),
                'filename': filename
            }
        }

    def predict_from_url(self, url: str) -> Dict:
        """
        Realizar predicción desde URL.

        Parameters:
        -----------
        url : str
            URL de la imagen.

        Returns:
        --------
        dict
            Resultado de la predicción.
        """
        start_time = time.time()

        # Descargar imagen
        image = self.preprocessor.load_image_from_url(url)

        # Validar imagen
        validation = self.preprocessor.validate_image(image)
        if not validation['is_valid']:
            raise ValueError(f"Imagen inválida: {', '.join(validation['errors'])}")

        # Preprocesar
        img_array = self.preprocessor.preprocess_for_prediction(image)

        # Predicción
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Interpretar resultados
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Calcular tiempo de procesamiento
        processing_time_ms = (time.time() - start_time) * 1000

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'interpretation': self._interpret_prediction(predicted_class, confidence),
            'all_probabilities': {
                class_name: float(predictions[i])
                for i, class_name in enumerate(self.class_names)
            },
            'metadata': {
                'image_size': list(image.size),
                'processing_time_ms': round(processing_time_ms, 2),
                'source': 'url',
                'url': url
            }
        }

    def predict_from_path(self, path: str) -> Dict:
        """
        Realizar predicción desde ruta local (para testing).

        Parameters:
        -----------
        path : str
            Ruta al archivo de imagen.

        Returns:
        --------
        dict
            Resultado de la predicción.
        """
        start_time = time.time()

        # Cargar imagen
        image = self.preprocessor.load_image_from_path(path)

        # Validar imagen
        validation = self.preprocessor.validate_image(image)
        if not validation['is_valid']:
            raise ValueError(f"Imagen inválida: {', '.join(validation['errors'])}")

        # Preprocesar
        img_array = self.preprocessor.preprocess_for_prediction(image)

        # Predicción
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Interpretar resultados
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Calcular tiempo de procesamiento
        processing_time_ms = (time.time() - start_time) * 1000

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'interpretation': self._interpret_prediction(predicted_class, confidence),
            'all_probabilities': {
                class_name: float(predictions[i])
                for i, class_name in enumerate(self.class_names)
            },
            'metadata': {
                'image_size': list(image.size),
                'processing_time_ms': round(processing_time_ms, 2),
                'source': 'local',
                'path': path
            }
        }

    @staticmethod
    def _interpret_prediction(class_name: str, confidence: float) -> str:
        """
        Interpretar predicción según nivel de confianza.

        Parameters:
        -----------
        class_name : str
            Clase predicha.
        confidence : float
            Nivel de confianza (0-1).

        Returns:
        --------
        str
            Interpretación del resultado.
        """
        if confidence >= 0.90:
            return f"Alta confianza - {class_name}"
        elif confidence >= 0.70:
            return f"Confianza moderada - {class_name} (recomendado confirmar)"
        else:
            return f"Baja confianza - {class_name} (requiere revisión por especialista)"

    def get_model_info(self) -> Dict:
        """
        Obtener información del modelo.

        Returns:
        --------
        dict
            Información sobre arquitectura y métricas del modelo.
        """
        if self.metadata is None:
            return {
                "model_type": "CNN - Convolutional Neural Network",
                "framework": "TensorFlow/Keras",
                "status": "Metadata no disponible"
            }

        return {
            "model_type": self.metadata.get("model_type", "CNN"),
            "framework": self.metadata.get("framework", "TensorFlow/Keras"),
            "input_shape": self.metadata.get("input_shape", [224, 224, 3]),
            "num_classes": self.metadata.get("num_classes", 4),
            "classes": self.metadata.get("classes", self.class_names),
            "architecture": self.metadata.get("architecture", {}),
            "training_info": self.metadata.get("training_info", {})
        }

    def get_class_descriptions(self) -> List[Dict]:
        """
        Obtener descripciones médicas de las clases.

        Returns:
        --------
        list
            Lista de diccionarios con información de cada clase.
        """
        descriptions = self.metadata.get("class_descriptions", {}) if self.metadata else {}

        return [
            {
                "class_name": class_name,
                "description": descriptions.get(
                    class_name,
                    f"Diagnóstico: {class_name}"
                )
            }
            for class_name in self.class_names
        ]
