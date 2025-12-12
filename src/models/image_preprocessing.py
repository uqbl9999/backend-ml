"""
Módulo de preprocesamiento de imágenes para reconocimiento médico.

Este módulo proporciona funcionalidades para cargar, validar y preprocesar
imágenes de rayos X antes de pasarlas al modelo CNN.
"""

import os
import io
import requests
import numpy as np
from typing import Dict, Optional, Tuple
from PIL import Image

# Constantes
IMAGE_SIZE = (224, 224)
ALLOWED_FORMATS = ['PNG', 'JPEG', 'JPG']
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


class ImagePreprocessor:
    """
    Clase para preprocesamiento de imágenes médicas.

    Proporciona métodos para cargar imágenes desde diferentes fuentes
    (archivo local, bytes, URL) y preprocesarlas para el modelo CNN.
    """

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Inicializar el preprocesador.

        Parameters:
        -----------
        target_size : tuple
            Tamaño objetivo (altura, ancho) para redimensionar imágenes.
        """
        self.target_size = target_size

    def load_image_from_path(self, path: str) -> Image.Image:
        """
        Cargar imagen desde ruta local.

        Parameters:
        -----------
        path : str
            Ruta al archivo de imagen.

        Returns:
        --------
        PIL.Image
            Imagen cargada.

        Raises:
        -------
        FileNotFoundError
            Si el archivo no existe.
        ValueError
            Si el archivo no es una imagen válida.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        try:
            image = Image.open(path)
            return image
        except Exception as e:
            raise ValueError(f"Error al cargar imagen: {str(e)}")

    def load_image_from_bytes(self, file_bytes: bytes) -> Image.Image:
        """
        Cargar imagen desde bytes (upload).

        Parameters:
        -----------
        file_bytes : bytes
            Bytes de la imagen.

        Returns:
        --------
        PIL.Image
            Imagen cargada.

        Raises:
        -------
        ValueError
            Si los bytes no representan una imagen válida.
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            return image
        except Exception as e:
            raise ValueError(f"Error al cargar imagen desde bytes: {str(e)}")

    def load_image_from_url(self, url: str, timeout: int = 10) -> Image.Image:
        """
        Descargar y cargar imagen desde URL.

        Parameters:
        -----------
        url : str
            URL de la imagen.
        timeout : int
            Timeout en segundos para la descarga.

        Returns:
        --------
        PIL.Image
            Imagen descargada.

        Raises:
        -------
        requests.exceptions.RequestException
            Si hay error en la descarga.
        ValueError
            Si la respuesta no es una imagen válida.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Verificar content-type
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL no apunta a una imagen (Content-Type: {content_type})")

            image = Image.open(io.BytesIO(response.content))
            return image

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error descargando imagen desde URL: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error procesando imagen desde URL: {str(e)}")

    def validate_image(self, image: Image.Image) -> Dict:
        """
        Validar que la imagen cumple con los requisitos.

        Parameters:
        -----------
        image : PIL.Image
            Imagen a validar.

        Returns:
        --------
        dict
            {'is_valid': bool, 'errors': list, 'warnings': list}
        """
        errors = []
        warnings = []

        # Validar formato
        if image.format not in ALLOWED_FORMATS:
            errors.append(f"Formato no soportado: {image.format}. Formatos permitidos: {ALLOWED_FORMATS}")

        # Validar dimensiones mínimas
        width, height = image.size
        if width < 100 or height < 100:
            errors.append(f"Imagen demasiado pequeña ({width}x{height}). Mínimo: 100x100")

        # Validar dimensiones máximas
        if width > 4000 or height > 4000:
            warnings.append(f"Imagen muy grande ({width}x{height}). Podría tardar en procesar.")

        # Validar modo (RGB, L, etc.)
        if image.mode not in ['RGB', 'L', 'RGBA']:
            warnings.append(f"Modo de color inusual: {image.mode}. Se convertirá a RGB.")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesar imagen para el modelo CNN.

        Pasos:
        1. Convertir a RGB si es necesario
        2. Redimensionar a target_size
        3. Convertir a array numpy
        4. Normalizar píxeles a rango [0, 1]

        Parameters:
        -----------
        image : PIL.Image
            Imagen a preprocesar.

        Returns:
        --------
        np.ndarray
            Array numpy con shape (224, 224, 3) y valores en [0, 1].
        """
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionar
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Convertir a numpy array
        img_array = np.array(image, dtype=np.float32)

        # Normalizar a [0, 1]
        img_array = img_array / 255.0

        return img_array

    def preprocess_for_prediction(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesar imagen y agregar dimensión de batch para predicción.

        Parameters:
        -----------
        image : PIL.Image
            Imagen a preprocesar.

        Returns:
        --------
        np.ndarray
            Array numpy con shape (1, 224, 224, 3).
        """
        img_array = self.preprocess(image)
        # Agregar dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
