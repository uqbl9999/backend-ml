"""
Cliente para comunicarse con Hugging Face Space
Proxy HTTP para enviar imágenes al modelo alojado en HF
"""

import requests
import io
import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HuggingFaceImageClient:
    """
    Cliente HTTP para comunicarse con el modelo TensorFlow en HF Space.

    Maneja el envío de imágenes y recepción de predicciones desde el Space.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Inicializar cliente HF.

        Parameters:
        -----------
        base_url : str
            URL base del HF Space (ej: "https://uqbl9999-mi-modelo-vision.hf.space")
        timeout : int
            Timeout en segundos para requests (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Backend-ML-Client/1.0'
        })

    def predict_from_upload(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Enviar imagen al HF Space para predicción.

        Parameters:
        -----------
        file_bytes : bytes
            Bytes de la imagen
        filename : str
            Nombre del archivo original

        Returns:
        --------
        dict
            Resultado de predicción del HF Space

        Raises:
        -------
        Exception
            Si hay error de timeout o en la comunicación
        """
        start_time = time.time()

        try:
            # Preparar multipart/form-data
            files = {'file': (filename, io.BytesIO(file_bytes), 'image/jpeg')}

            # Enviar request al HF Space
            response = self.session.post(
                f"{self.base_url}/predict",
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"HF prediction success: {latency_ms:.2f}ms, file={filename}")

            return response.json()

        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"HF prediction timeout after {latency_ms:.2f}ms")
            raise Exception("HF Space timeout (modelo demorando mucho en procesar)")

        except requests.exceptions.HTTPError as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"HF prediction HTTP error {e.response.status_code} after {latency_ms:.2f}ms")

            # Intentar obtener mensaje de error del Space
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)

            raise Exception(f"Error en HF Space: {error_detail}")

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"HF prediction request failed after {latency_ms:.2f}ms: {e}")
            raise Exception(f"Error de conexión con HF Space: {str(e)}")

    def predict_from_url(self, url: str) -> Dict:
        """
        Solicitar predicción desde URL de imagen.

        Parameters:
        -----------
        url : str
            URL de la imagen

        Returns:
        --------
        dict
            Resultado de predicción del HF Space

        Raises:
        -------
        Exception
            Si hay error en la comunicación
        """
        start_time = time.time()

        try:
            payload = {"image_url": url}

            response = self.session.post(
                f"{self.base_url}/predict-url",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"HF prediction from URL success: {latency_ms:.2f}ms")

            return response.json()

        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"HF prediction from URL timeout after {latency_ms:.2f}ms")
            raise Exception("HF Space timeout")

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"HF prediction from URL failed after {latency_ms:.2f}ms: {e}")
            raise Exception(f"Error en HF Space: {str(e)}")

    def get_model_info(self) -> Dict:
        """
        Obtener información del modelo desde HF Space.

        Returns:
        --------
        dict
            Información del modelo (arquitectura, training info, etc.)

        Raises:
        -------
        Exception
            Si falla la obtención de información
        """
        try:
            response = self.session.get(
                f"{self.base_url}/model-info",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise Exception(f"No se pudo obtener info del modelo: {str(e)}")

    def get_classes(self) -> list:
        """
        Obtener información de clases desde HF Space.

        Returns:
        --------
        list
            Lista de diccionarios con clase y descripción

        Raises:
        -------
        Exception
            Si falla la obtención de clases
        """
        try:
            response = self.session.get(
                f"{self.base_url}/classes",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get classes: {e}")
            raise Exception(f"No se pudo obtener clases: {str(e)}")

    def get_statistics(self) -> Dict:
        """
        Obtener estadísticas del modelo desde HF Space.

        Returns:
        --------
        dict
            Estadísticas (accuracy, métricas por clase, confusion matrix)

        Raises:
        -------
        Exception
            Si falla la obtención de estadísticas
        """
        try:
            response = self.session.get(
                f"{self.base_url}/statistics",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise Exception(f"No se pudo obtener estadísticas: {str(e)}")

    def health_check(self) -> Dict:
        """
        Verificar salud del HF Space.

        Returns:
        --------
        dict
            Estado del servicio {"status": "healthy"|"unhealthy", ...}
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"HF Space health check: {result}")
            return result
        except Exception as e:
            logger.error(f"HF Space health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False
            }
