"""
Servicio de reconocimiento de imágenes médicas.

Este módulo proporciona una interfaz de alto nivel para el reconocimiento
de imágenes de rayos X, integrando predicción y explicaciones XAI.
"""

import os
import json
from typing import Dict, List, Optional
from src.models.image_prediction import ImagePredictor


class ImageRecognitionService:
    """
    Servicio para reconocimiento de imágenes médicas.

    Proporciona métodos para predicción, estadísticas y generación de
    explicaciones médicas usando IA explicable.
    """

    def __init__(self, model_path: str):
        """
        Inicializar el servicio.

        Parameters:
        -----------
        model_path : str
            Ruta al modelo entrenado (.keras).
        """
        self.model_path = model_path
        self.predictor = ImagePredictor(model_path)
        self.metadata_path = os.path.join(
            os.path.dirname(model_path),
            'model_metadata.json'
        )

    def predict_from_upload(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Realizar predicción desde archivo upload con validación.

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
        try:
            result = self.predictor.predict_from_file(file_bytes, filename)
            return result
        except Exception as e:
            raise Exception(f"Error en predicción desde upload: {str(e)}")

    def predict_from_url(self, url: str) -> Dict:
        """
        Realizar predicción desde URL con validación.

        Parameters:
        -----------
        url : str
            URL de la imagen.

        Returns:
        --------
        dict
            Resultado de la predicción.
        """
        try:
            result = self.predictor.predict_from_url(url)
            return result
        except Exception as e:
            raise Exception(f"Error en predicción desde URL: {str(e)}")

    def generate_explanation_with_xai(
        self,
        prediction_result: Dict,
        xai_service
    ) -> Dict:
        """
        Generar explicación médica usando servicio XAI.

        Parameters:
        -----------
        prediction_result : dict
            Resultado de la predicción.
        xai_service : XAIService
            Instancia del servicio XAI.

        Returns:
        --------
        dict
            {'success': bool, 'explanation': dict}
        """
        try:
            # Usar el método específico para imágenes médicas del servicio XAI
            xai_result = xai_service.generate_medical_image_explanation(
                predicted_class=prediction_result['predicted_class'],
                confidence=prediction_result['confidence'],
                all_probabilities=prediction_result['all_probabilities']
            )

            return xai_result

        except Exception as e:
            print(f"Error generando explicación XAI: {str(e)}")
            return self._get_fallback_explanation(prediction_result)

    def _get_fallback_explanation(self, prediction_result: Dict) -> Dict:
        """
        Generar explicación por defecto si XAI falla.

        Parameters:
        -----------
        prediction_result : dict
            Resultado de la predicción.

        Returns:
        --------
        dict
            Explicación fallback.
        """
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']

        # Recomendaciones por clase
        recommendations_by_class = {
            'COVID19': [
                "Solicitar prueba PCR para SARS-CoV-2",
                "Evaluar saturación de oxígeno y función respiratoria",
                "Considerar aislamiento preventivo según protocolos"
            ],
            'NORMAL': [
                "Mantener seguimiento rutinario",
                "Fomentar hábitos saludables",
                "Consultar si aparecen síntomas respiratorios"
            ],
            'PNEUMONIA': [
                "Realizar cultivos para identificar agente etiológico",
                "Iniciar tratamiento antibiótico empírico",
                "Monitorear respuesta clínica y parámetros vitales"
            ],
            'TUBERCULOSIS': [
                "Solicitar baciloscopia y cultivo de esputo",
                "Evaluar contactos cercanos del paciente",
                "Referir a programa de control de tuberculosis"
            ]
        }

        context_by_class = {
            'COVID19': "Patrón compatible con neumonía viral por COVID-19, requiere confirmación con PCR.",
            'NORMAL': "Radiografía sin hallazgos patológicos significativos.",
            'PNEUMONIA': "Patrón sugestivo de proceso infeccioso pulmonar, requiere correlación clínica.",
            'TUBERCULOSIS': "Hallazgos compatibles con tuberculosis pulmonar, requiere confirmación bacteriológica."
        }

        considerations = []
        if confidence >= 0.90:
            considerations.append(f"Confianza alta en diagnóstico de {predicted_class}")
        elif confidence >= 0.70:
            considerations.append(f"Confianza moderada, confirmar con estudios adicionales")
        else:
            considerations.append("Baja confianza, revisión por especialista necesaria")

        # Verificar diagnósticos diferenciales
        all_probs = prediction_result['all_probabilities']
        differentials = [
            cls for cls, prob in all_probs.items()
            if cls != predicted_class and prob > 0.20
        ]
        if differentials:
            considerations.append(f"Considerar diagnósticos diferenciales: {', '.join(differentials)}")

        return {
            'success': False,
            'explanation': {
                'contexto_clinico': context_by_class.get(
                    predicted_class,
                    f"Diagnóstico sugerido: {predicted_class}"
                ),
                'recomendaciones': recommendations_by_class.get(
                    predicted_class,
                    ["Consultar con especialista", "Realizar estudios complementarios", "Seguimiento clínico"]
                ),
                'consideraciones': considerations
            }
        }

    def get_statistics(self) -> Dict:
        """
        Obtener estadísticas del modelo desde metadata.

        Returns:
        --------
        dict
            Estadísticas del modelo (accuracy, métricas por clase, etc.).
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Metadata del modelo no encontrada")

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            metrics = metadata.get('metrics', {})

            return {
                'test_accuracy': metrics.get('test_accuracy', 0.0),
                'test_loss': metrics.get('test_loss', 0.0),
                'per_class_metrics': metrics.get('per_class_metrics', {}),
                'confusion_matrix': metrics.get('confusion_matrix', [])
            }

        except Exception as e:
            raise Exception(f"Error cargando estadísticas: {str(e)}")

    def get_class_info(self) -> List[Dict]:
        """
        Obtener información de las clases con descripciones médicas.

        Returns:
        --------
        list
            Lista de diccionarios con información de cada clase.
        """
        return self.predictor.get_class_descriptions()


# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_image_service_instance: Optional[ImageRecognitionService] = None


def get_image_service(model_path: Optional[str] = None) -> Optional[ImageRecognitionService]:
    """
    Obtener instancia singleton del servicio de imágenes.

    Parameters:
    -----------
    model_path : str, optional
        Ruta al modelo. Si no se especifica, se busca en rutas predeterminadas.

    Returns:
    --------
    ImageRecognitionService or None
        Instancia del servicio o None si no se encuentra el modelo.
    """
    global _image_service_instance

    if _image_service_instance is None:
        if model_path is None:
            model_path = _find_model_path()

        if model_path and os.path.exists(model_path):
            try:
                _image_service_instance = ImageRecognitionService(model_path)
            except Exception as e:
                print(f"Error al inicializar servicio de imágenes: {e}")
                return None
        else:
            print("⚠️  Modelo de imágenes no encontrado. Los endpoints de imágenes no estarán disponibles.")
            return None

    return _image_service_instance


def _find_model_path() -> Optional[str]:
    """
    Buscar modelo en rutas predeterminadas.

    Returns:
    --------
    str or None
        Ruta al modelo si se encuentra, None en caso contrario.
    """
    # Rutas posibles para desarrollo y producción
    possible_paths = [
        # Desarrollo local (relativo al archivo)
        os.path.join(
            os.path.dirname(__file__),
            '../../models/image_models/best_model.keras'
        ),
        # Producción (Docker/Render)
        '/app/models/image_models/best_model.keras',
        # CWD
        os.path.join(os.getcwd(), 'models/image_models/best_model.keras'),
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"✅ Modelo de imágenes encontrado en: {abs_path}")
            return abs_path

    return None
