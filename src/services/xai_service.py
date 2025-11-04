"""
Servicio de IA Explicable (Explainable AI) usando Perplexity API
Genera explicaciones concisas sobre las predicciones del modelo
"""

import os
import json
import requests
from typing import Dict, Optional


class XAIService:
    """Servicio para generar explicaciones de IA explicable usando Perplexity"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el servicio XAI con Perplexity API

        Args:
            api_key: API key de Perplexity (si no se provee, se obtiene de variable de entorno)
        """
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("Perplexity API key no configurada. Establecer PERPLEXITY_API_KEY en variables de entorno.")

        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Mapeo de rangos de riesgo
        self.risk_ranges = {
            "Riesgo Bajo": (0, 15),
            "Riesgo Moderado": (15, 25),
            "Riesgo Alto": (25, 35),
            "Riesgo Muy Alto": (35, 100)
        }

    def _build_context_prompt(self, params: Dict, prediction: float, interpretation: str) -> str:
        """
        Construye el prompt contextual para el modelo

        Args:
            params: Parámetros de entrada usados para la predicción
            prediction: Tasa de positividad predicha
            interpretation: Interpretación del nivel de riesgo

        Returns:
            Prompt formateado
        """
        # Mapeo de nombres amigables
        etapa_map = {
            '< 1': 'menores de 1 año',
            '1 - 4': '1 a 4 años',
            '5 - 9': '5 a 9 años',
            '10 - 11': '10 a 11 años',
            '12 - 14': '12 a 14 años',
            '15 - 17': '15 a 17 años',
            '18 - 24': '18 a 24 años',
            '25 - 29': '25 a 29 años',
            '30 - 39': '30 a 39 años',
            '40 - 59': '40 a 59 años',
            '60 - 79': '60 a 79 años',
            '80  +': '80 años o más'
        }

        sexo_map = {
            'F': 'Femenino',
            'M': 'Masculino'
        }

        mes_map = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }

        etapa = etapa_map.get(params.get('Etapa', ''), params.get('Etapa', ''))
        sexo = sexo_map.get(params.get('Sexo', ''), params.get('Sexo', ''))
        mes = mes_map.get(params.get('NroMes', 0), str(params.get('NroMes', '')))

        prompt = f"""Eres un experto en salud mental pública en Perú. Debes explicar de manera breve y clara por qué una predicción de riesgo de salud mental tiene cierto nivel.

CONTEXTO DE LA PREDICCIÓN:
- Departamento: {params.get('Departamento', 'N/A')}
- Provincia: {params.get('Provincia', 'N/A')}
- Mes: {mes}
- Sexo: {sexo}
- Edad: {etapa}
- Tipo de tamizaje: {params.get('DetalleTamizaje', 'N/A')}

RESULTADO:
- Tasa de positividad estimada: {prediction:.2f}%
- Nivel de riesgo: {interpretation}

Tu tarea es generar SOLO 3 elementos en formato JSON:

1. "contexto_situacional": Una frase corta (máximo 25 palabras) explicando por qué la tasa está en ese rango para esta combinación de factores.

2. "acciones": Array con 3 acciones preventivas concretas y específicas (cada una máximo 10 palabras).

3. "factores_clave": Array con 2-3 factores principales que influyen en esta predicción (cada uno máximo 8 palabras).

IMPORTANTE:
- Sé EXTREMADAMENTE CONCISO
- NO uses lenguaje técnico complejo
- NO repitas la información del contexto
- Enfócate en insights accionables
- Las acciones deben ser específicas para este contexto

Responde SOLO con el JSON, sin texto adicional:
{{
  "contexto_situacional": "...",
  "acciones": ["acción 1", "acción 2", "acción 3"],
  "factores_clave": ["factor 1", "factor 2"]
}}"""

        return prompt

    def generate_explanation(
        self,
        params: Dict,
        prediction: float,
        interpretation: str,
        model: str = "sonar",
        temperature: float = 0.7
    ) -> Dict:
        """
        Genera una explicación de IA explicable para una predicción

        Args:
            params: Parámetros de entrada usados para la predicción
            prediction: Tasa de positividad predicha
            interpretation: Interpretación del nivel de riesgo
            model: Modelo de Perplexity a usar (default: sonar - lightweight y económico)
            temperature: Temperatura para la generación (default: 0.7)

        Returns:
            Dict con la explicación estructurada
        """
        try:
            prompt = self._build_context_prompt(params, prediction, interpretation)

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un experto en salud mental pública. Generas explicaciones concisas y accionables en formato JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": 500
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            response_data = response.json()

            # Extraer el contenido de la respuesta
            content = response_data['choices'][0]['message']['content']

            # Limpiar el contenido si viene con markdown o texto adicional
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            explanation = json.loads(content)

            return {
                "success": True,
                "explanation": explanation,
                "tokens_used": response_data.get('usage', {}).get('total_tokens', 0)
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Error de conexión con Perplexity: {str(e)}",
                "explanation": {
                    "contexto_situacional": "No se pudo generar la explicación automática.",
                    "acciones": [
                        "Reforzar acciones preventivas y seguimiento",
                        "Monitorear indicadores críticos semanalmente",
                        "Coordinar intervención con equipos territoriales"
                    ],
                    "factores_clave": [
                        "Combinación de factores demográficos y geográficos",
                        "Patrón estacional y tipo de tamizaje"
                    ]
                }
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Error al parsear JSON: {str(e)}",
                "explanation": {
                    "contexto_situacional": "No se pudo generar la explicación automática.",
                    "acciones": [
                        "Reforzar acciones preventivas y seguimiento",
                        "Monitorear indicadores críticos semanalmente",
                        "Coordinar intervención con equipos territoriales"
                    ],
                    "factores_clave": [
                        "Combinación de factores demográficos y geográficos",
                        "Patrón estacional y tipo de tamizaje"
                    ]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "explanation": {
                    "contexto_situacional": "No se pudo generar la explicación automática.",
                    "acciones": [
                        "Reforzar acciones preventivas y seguimiento",
                        "Monitorear indicadores críticos semanalmente",
                        "Coordinar intervención con equipos territoriales"
                    ],
                    "factores_clave": [
                        "Combinación de factores demográficos y geográficos",
                        "Patrón estacional y tipo de tamizaje"
                    ]
                }
            }

    def generate_batch_explanations(
        self,
        predictions: list[Dict],
        model: str = "sonar",
        temperature: float = 0.7
    ) -> list[Dict]:
        """
        Genera explicaciones para múltiples predicciones

        Args:
            predictions: Lista de predicciones con sus parámetros
            model: Modelo de Perplexity a usar
            temperature: Temperatura para la generación

        Returns:
            Lista de explicaciones
        """
        explanations = []

        for pred in predictions:
            explanation = self.generate_explanation(
                params=pred.get('input_data', {}),
                prediction=pred.get('tasa_positividad_predicha', 0),
                interpretation=pred.get('interpretacion', ''),
                model=model,
                temperature=temperature
            )
            explanations.append(explanation)

        return explanations


# Singleton instance
_xai_service_instance: Optional[XAIService] = None


def get_xai_service(api_key: Optional[str] = None) -> Optional[XAIService]:
    """
    Obtiene la instancia singleton del servicio XAI

    Args:
        api_key: API key de Perplexity (opcional)

    Returns:
        Instancia de XAIService o None si no se puede inicializar
    """
    global _xai_service_instance

    if _xai_service_instance is None:
        try:
            _xai_service_instance = XAIService(api_key=api_key)
        except ValueError:
            # API key no configurada
            return None

    return _xai_service_instance


def reset_xai_service():
    """Resetea la instancia singleton (útil para testing)"""
    global _xai_service_instance
    _xai_service_instance = None
