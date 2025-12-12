"""
Servicio de reconocimiento de imágenes médicas.

Este módulo proporciona una interfaz de alto nivel para el reconocimiento
de imágenes de rayos X, integrando predicción y explicaciones XAI.

MIGRACIÓN: Ahora actúa como proxy hacia Hugging Face Space.
El modelo TensorFlow reside en HF para reducir memoria del backend.
"""

import os
import json
from typing import Dict, List, Optional
from src.services.huggingface_client import HuggingFaceImageClient


class ImageRecognitionService:
    """
    Servicio para reconocimiento de imágenes médicas.

    Ahora actúa como proxy hacia Hugging Face Space.
    Proporciona métodos para predicción, estadísticas y generación de
    explicaciones médicas usando IA explicable.
    """

    def __init__(self, hf_space_url: str):
        """
        Inicializar el servicio con cliente HF.

        Parameters:
        -----------
        hf_space_url : str
            URL del Hugging Face Space con el modelo.
        """
        self.hf_client = HuggingFaceImageClient(hf_space_url)

        # Mantener metadata_path local como fallback
        self.metadata_path = os.path.join(
            os.path.dirname(__file__),
            '../../models/image_models/model_metadata.json'
        )

    def predict_from_upload(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Realizar predicción desde archivo upload usando HF Space.

        Parameters:
        -----------
        file_bytes : bytes
            Bytes del archivo de imagen.
        filename : str
            Nombre del archivo.

        Returns:
        --------
        dict
            Resultado de la predicción desde HF Space.
        """
        try:
            result = self.hf_client.predict_from_upload(file_bytes, filename)
            return result
        except Exception as e:
            raise Exception(f"Error en predicción desde upload: {str(e)}")

    def predict_from_url(self, url: str) -> Dict:
        """
        Realizar predicción desde URL usando HF Space.

        Parameters:
        -----------
        url : str
            URL de la imagen.

        Returns:
        --------
        dict
            Resultado de la predicción desde HF Space.
        """
        try:
            result = self.hf_client.predict_from_url(url)
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
        Obtener estadísticas del modelo desde HF Space.
        Usa metadata local como fallback.

        Returns:
        --------
        dict
            Estadísticas del modelo (accuracy, métricas por clase, etc.).
        """
        try:
            # Intentar obtener desde HF Space primero
            return self.hf_client.get_statistics()
        except Exception as e:
            # Fallback a metadata local
            print(f"⚠️ No se pudo obtener stats de HF Space: {e}. Usando metadata local.")

            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError("Metadata del modelo no encontrada ni en HF ni localmente")

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

            except Exception as fallback_error:
                raise Exception(f"Error cargando estadísticas: {str(fallback_error)}")

    def get_class_info(self) -> List[Dict]:
        """
        Obtener información de las clases desde HF Space.

        Returns:
        --------
        list
            Lista de diccionarios con información de cada clase.
        """
        try:
            return self.hf_client.get_classes()
        except Exception as e:
            raise Exception(f"Error obteniendo clases: {str(e)}")

    def get_model_info(self) -> Dict:
        """
        Obtener información del modelo desde HF Space.

        Returns:
        --------
        dict
            Información del modelo (arquitectura, training info, etc.).
        """
        try:
            return self.hf_client.get_model_info()
        except Exception as e:
            raise Exception(f"Error obteniendo info del modelo: {str(e)}")


# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_image_service_instance: Optional[ImageRecognitionService] = None


def get_image_service(hf_space_url: Optional[str] = None) -> Optional[ImageRecognitionService]:
    """
    Obtener instancia singleton del servicio de imágenes.

    MIGRACIÓN: Ahora usa HF Space URL en lugar de model_path.

    Parameters:
    -----------
    hf_space_url : str, optional
        URL del Hugging Face Space. Si no se especifica, se obtiene de env var HF_SPACE_URL.

    Returns:
    --------
    ImageRecognitionService or None
        Instancia del servicio o None si no se puede conectar al HF Space.
    """
    global _image_service_instance

    if _image_service_instance is None:
        if hf_space_url is None:
            hf_space_url = os.getenv('HF_SPACE_URL')

        if not hf_space_url:
            print("⚠️  HF_SPACE_URL no configurada. Los endpoints de imágenes no estarán disponibles.")
            print("    Configura la variable de entorno HF_SPACE_URL con la URL de tu Space.")
            return None

        try:
            _image_service_instance = ImageRecognitionService(hf_space_url)

            # Health check al HF Space
            health = _image_service_instance.hf_client.health_check()
            if health.get('status') == 'healthy':
                print(f"✅ Servicio de imágenes conectado a HF Space: {hf_space_url}")
            else:
                print(f"⚠️  HF Space no saludable: {health}")
                print("    Verifica que el Space esté corriendo en Hugging Face.")
                return None

        except Exception as e:
            print(f"Error al conectar con HF Space: {e}")
            return None

    return _image_service_instance


# MIGRACIÓN: _find_model_path() eliminada - ya no se busca modelo local
# El modelo ahora reside en Hugging Face Space
