# Guía de IA Explicable (XAI) con Perplexity

## 📖 Introducción

La funcionalidad de **Explainable AI (XAI)** del sistema proporciona explicaciones claras y accionables sobre las predicciones del modelo de riesgo de salud mental. Utilizando **Perplexity AI** con modelos Llama 3.1 Sonar, el sistema genera contexto situacional, acciones recomendadas y factores clave para cada predicción.

## 🎯 Objetivo

El módulo XAI tiene como objetivos:

1. **Transparencia**: Que los usuarios comprendan por qué se obtiene cierto nivel de riesgo
2. **Acción**: Proporcionar recomendaciones específicas y aplicables
3. **Confianza**: Aumentar la confianza en las predicciones del modelo mediante explicaciones
4. **UI-friendly**: Generar explicaciones concisas que se integren perfectamente en interfaces de usuario

## 🔧 Configuración

### Requisitos

- API Key de Perplexity (obtén una en [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api))
- Python 3.8+
- Paquete `requests` (incluido en requirements.txt)

### Configurar API Key

**Opción 1: Variable de entorno**

```bash
# Linux/Mac
export PERPLEXITY_API_KEY="pplx-..."

# Windows CMD
set PERPLEXITY_API_KEY=pplx-...

# Windows PowerShell
$env:PERPLEXITY_API_KEY="pplx-..."
```

**Opción 2: Archivo .env**

1. Copia `.env.example` a `.env`
2. Edita `.env` y agrega tu API key:
   ```
   PERPLEXITY_API_KEY=pplx-your-actual-key-here
   ```

## 🚀 Uso

### Endpoint Principal

```bash
POST /predict/explain
```

### Ejemplo de Request

```bash
curl -X POST "http://localhost:8000/predict/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "NroMes": 1,
    "Departamento": "ANCASH",
    "Provincia": "AIJA",
    "Sexo": "F",
    "Etapa": "< 1",
    "DetalleTamizaje": "SINDROME Y/O TRASTORNO PSICOTICO"
  }'
```

### Ejemplo de Response

```json
{
  "tasa_positividad_predicha": 24.02,
  "interpretacion": "Riesgo Muy Alto - Intervención urgente requerida",
  "input_data": {
    "NroMes": 1,
    "Departamento": "ANCASH",
    "Provincia": "AIJA",
    "Sexo": "F",
    "Etapa": "< 1",
    "DetalleTamizaje": "SINDROME Y/O TRASTORNO PSICOTICO",
    "ubigeo": 20201
  },
  "explicacion": {
    "contexto_situacional": "La tasa se encuentra en un rango moderado respecto a la media histórica. Se recomienda fortalecer la detección temprana y reforzar los protocolos de derivación.",
    "acciones": [
      "Reforzar acciones preventivas y seguimiento",
      "Monitorear indicadores críticos semanalmente",
      "Coordinar intervención con equipos territoriales"
    ],
    "factores_clave": [
      "Combinación específica de ubicación geográfica y grupo etario",
      "Mes del año y tipo de tamizaje específico"
    ]
  }
}
```

## 📊 Estructura de la Explicación

### 1. Contexto Situacional

- **Qué es**: Una frase corta (máximo 25 palabras) que explica el nivel de riesgo
- **Propósito**: Dar contexto sobre por qué la tasa está en ese rango
- **Ejemplo**: "La tasa se encuentra en un rango moderado respecto a la media histórica..."

### 2. Acciones Recomendadas

- **Qué es**: Array con 3 acciones preventivas concretas
- **Formato**: Cada acción tiene máximo 10 palabras
- **Propósito**: Proporcionar pasos accionables específicos para el contexto
- **Ejemplo**:
  - "Reforzar acciones preventivas y seguimiento"
  - "Monitorear indicadores críticos semanalmente"
  - "Coordinar intervención con equipos territoriales"

### 3. Factores Clave

- **Qué es**: Array con 2-3 factores principales que influyen en la predicción
- **Formato**: Cada factor tiene máximo 8 palabras
- **Propósito**: Identificar qué características tienen mayor impacto
- **Ejemplo**:
  - "Combinación específica de ubicación geográfica y grupo etario"
  - "Mes del año y tipo de tamizaje específico"

## 🔍 Cómo Funciona

### Flujo de Proceso

```
1. Usuario envía request → /predict/explain
                ↓
2. Sistema valida entrada y calcula ubigeo
                ↓
3. Modelo ML genera predicción
                ↓
4. XAI Service construye prompt contextual
                ↓
5. Perplexity AI analiza y genera explicación
                ↓
6. Sistema combina predicción + explicación
                ↓
7. Response con datos completos al usuario
```

### Prompt Engineering

El servicio XAI construye un prompt específico que incluye:

- Contexto geográfico (Departamento, Provincia)
- Características demográficas (Sexo, Edad)
- Información temporal (Mes)
- Tipo de tamizaje
- Resultado de la predicción
- Nivel de riesgo interpretado

Perplexity AI recibe instrucciones para generar:
- Explicaciones extremadamente concisas
- Acciones específicas al contexto
- Formato JSON estructurado

## 💰 Consideraciones de Costo

### Modelo Utilizado

- **Por defecto**: `sonar` (basado en Llama 3.3 70B)
- **Motivo**: Ligero, económico y con acceso a información actualizada
- **Tokens promedio**: ~500 tokens por explicación
- **Lanzamiento**: Febrero 2025 (última generación)

### Estimación de Costos

A precio actual de Perplexity AI:
- **Costo por 1M tokens**: Aproximadamente $1 USD
- **Costo por explicación**: ~$0.0005 (medio centavo)
- **Muy económico**: Más barato que GPT-4o-mini

### Optimización

Para reducir costos aún más:
1. Implementar caché de explicaciones similares
2. Reducir max_tokens si las explicaciones son muy largas
3. Usar temperature más baja (menos creatividad = menos tokens)

## 🛡️ Manejo de Errores

### Escenarios de Fallback

El servicio XAI incluye manejo robusto de errores:

1. **API Key no configurada**: Retorna error 503 indicando configuración faltante
2. **Error de Perplexity**: Retorna explicación genérica predeterminada
3. **Timeout**: Explicación de fallback sin interrumpir el servicio

### Explicación de Fallback

Si Perplexity AI falla, el sistema retorna:

```json
{
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
```

## 🔬 Personalización

### Cambiar el Modelo

Puedes usar un modelo diferente editando `xai_service.py`:

```python
# Modelos disponibles en Perplexity (2025):
# - sonar (ligero, económico, recomendado para producción)
# - sonar-pro (avanzado, mejor calidad con grounding mejorado)

explanation = xai_service.generate_explanation(
    params=params,
    prediction=prediction,
    interpretation=interpretation,
    model="sonar-pro"  # Modelo más potente
)
```

### Ajustar Temperature

```python
# Más determinístico (menos creatividad)
explanation = xai_service.generate_explanation(
    params=params,
    prediction=prediction,
    interpretation=interpretation,
    temperature=0.3  # Más consistente
)

# Más creativo (más variedad)
explanation = xai_service.generate_explanation(
    params=params,
    prediction=prediction,
    interpretation=interpretation,
    temperature=1.0  # Más variado
)
```

## 📝 Mejores Prácticas

### Uso en Producción

1. **Monitorear costos**: Revisa uso en Perplexity dashboard
2. **Implementar rate limiting**: Evita sobrecostos por uso excesivo
3. **Caché inteligente**: Guarda explicaciones para parámetros similares
4. **Logging**: Registra todas las llamadas para análisis posterior
5. **Validación médica**: Revisa con expertos la calidad de las explicaciones

### Uso Responsable

1. **Disclaimer**: Las explicaciones son orientativas, no diagnósticos
2. **Revisión humana**: Expertos en salud deben validar recomendaciones
3. **Privacidad**: No incluyas información sensible en los prompts
4. **Transparencia**: Indica claramente que las explicaciones son generadas por IA

## 🧪 Testing

### Test Manual

```bash
# Con API key configurada
curl -X POST "http://localhost:8000/predict/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "NroMes": 6,
    "Departamento": "LIMA",
    "Provincia": "LIMA",
    "Sexo": "M",
    "Etapa": "18 - 24",
    "DetalleTamizaje": "TRASTORNO DEPRESIVO"
  }'
```

### Verificar Configuración

```bash
# Verificar que la API key está configurada (Linux/Mac)
echo $PERPLEXITY_API_KEY

# Windows CMD
echo %PERPLEXITY_API_KEY%

# Windows PowerShell
echo $env:PERPLEXITY_API_KEY
```

## ❓ Troubleshooting

### Error: "XAI service not available"

**Causa**: Variable de entorno `PERPLEXITY_API_KEY` no configurada

**Solución**:
```bash
export PERPLEXITY_API_KEY="pplx-..."
# Reinicia la API
```

### Error: "Authentication failed"

**Causa**: API key inválida o sin créditos

**Solución**:
1. Verifica tu API key en Perplexity dashboard
2. Confirma que tienes créditos disponibles
3. Regenera la API key si es necesario

### Explicaciones genéricas

**Causa**: Prompts no suficientemente específicos

**Solución**:
- Revisa y ajusta los prompts en `xai_service.py`
- Reduce la temperature para más consistencia
- Considera usar un modelo más potente (llama-3.1-sonar-large)

### Error: Respuesta no es JSON válido

**Causa**: Perplexity a veces incluye texto markdown alrededor del JSON

**Solución**:
- El servicio ya incluye limpieza automática de markdown
- Si persiste, revisa el contenido raw en los logs

## 🔗 Referencias

- [Perplexity API Documentation](https://docs.perplexity.ai/)
- [Perplexity API Pricing](https://www.perplexity.ai/settings/api)
- [Best Practices for Prompt Engineering](https://docs.perplexity.ai/guides/prompting)

## 💡 Ventajas de Perplexity AI

1. **Acceso a información actualizada**: Modelos "online" con búsqueda web
2. **Costo competitivo**: Más económico que alternativas similares
3. **Modelos Llama 3.1**: Modelos de código abierto de alta calidad
4. **Sin censura excesiva**: Mejor para temas de salud mental
5. **API REST simple**: Uso directo con requests, sin dependencias pesadas

## 📧 Soporte

Para preguntas o problemas con XAI:
1. Revisa esta guía completa
2. Verifica los logs de la API
3. Consulta la documentación de Perplexity
4. Abre un issue en el repositorio del proyecto
