# Mental Health Screening Prediction API

Sistema de predicción de tasas de positividad de tamizajes de salud mental en Perú utilizando Machine Learning.

## 📋 Descripción del Proyecto

Este proyecto desarrolla un modelo de Machine Learning para predecir la tasa de positividad de tamizajes de salud mental basándose en características demográficas, geográficas y temporales. El objetivo es optimizar la asignación de recursos hospitalarios y personal médico especializado.

### Características Principales

- **Predicción de Tasa de Positividad**: Predice el porcentaje de casos positivos en tamizajes
- **API REST con FastAPI**: Endpoints para predicciones individuales y en lote
- **IA Explicable (XAI)**: Explicaciones generadas por GPT sobre las predicciones
- **Modelos ML**: Gradient Boosting y Random Forest con optimización de hiperparámetros
- **Balanceo de Datos**: Implementación de SMOTE para manejo de clases desbalanceadas
- **Interpretación de Resultados**: Clasificación automática de niveles de riesgo

## 🏗️ Estructura del Proyecto

```
backend-ml/
│
├── data/                          # Datos (NO incluir datasets completos en Git)
│   ├── dataset_limpio.csv         # Datos después de limpieza
│   ├── df_clean_to_model.csv      # Datos codificados
│   ├── dataset_balanceado.csv     # Datos balanceados
│   └── TB_UBIGEOS.csv            # Tabla de ubigeos del Perú
│
├── src/                           # Código fuente
│   ├── data_preparation.py        # Preparación y limpieza de datos
│   ├── train_model.py            # Script de entrenamiento
│   ├── models/                    # Módulos del modelo
│   │   ├── __init__.py
│   │   ├── training.py            # Entrenamiento del modelo
│   │   └── prediction.py          # Predicciones
│   └── services/                  # Servicios adicionales
│       ├── __init__.py
│       ├── ubigeo_service.py      # Mapeo Departamento+Provincia→Ubigeo
│       └── xai_service.py         # Servicio de IA Explicable (XAI)
│
├── api/                           # API REST
│   └── main.py                    # FastAPI application
│
├── models/                        # Modelos entrenados
│   └── trained_model.pkl          # Modelo serializado
│
├── docs/                          # Documentación y visualizaciones
│   ├── evaluation_actual_vs_predicted.png
│   └── evaluation_feature_importance.png
│
├── tests/                         # Pruebas unitarias
│
├── notebooks/                     # Jupyter notebooks experimentales
│   └── parcialfinal.ipynb        # Notebook original
│
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar el Repositorio

```bash
git clone <repository-url>
cd backend-ml
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Preparar los Datos

Asegúrate de tener el archivo `tamizajes.csv` en el directorio raíz del proyecto.

### Paso 5: (Opcional) Configurar IA Explicable

Para habilitar las funcionalidades de **Explainable AI (XAI)**, configura tu API key de Perplexity:

```bash
# Linux/Mac
export PERPLEXITY_API_KEY="tu-api-key-aqui"

# Windows CMD
set PERPLEXITY_API_KEY=tu-api-key-aqui

# Windows PowerShell
$env:PERPLEXITY_API_KEY="tu-api-key-aqui"
```

**Nota:** Esta configuración es opcional. La API funcionará normalmente sin ella, pero el endpoint `/predict/explain` no estará disponible.

## 🎯 Uso

### 1. Entrenar el Modelo

```bash
# Entrenamiento completo con optimización de hiperparámetros
python src/train_model.py

# Opciones adicionales:
python src/train_model.py --data tamizajes.csv --model gradient_boosting

# Entrenamiento rápido sin optimización:
python src/train_model.py --no-optimize

# Con Random Forest:
python src/train_model.py --model random_forest
```

Este proceso generará:
- `models/trained_model.pkl`: Modelo entrenado
- `data/dataset_*.csv`: Datasets procesados
- `docs/evaluation_*.png`: Gráficos de evaluación

### 2. Iniciar la API

```bash
# Modo desarrollo (con recarga automática)
uvicorn api.main:app --reload

# Modo producción
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`

### 3. Documentación Interactiva de la API

Una vez iniciada la API, accede a:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🗺️ Mapeo Automático de Ubicación

El sistema utiliza un **servicio de mapeo automático** que convierte la combinación de **Departamento + Provincia** en el código **Ubigeo** correspondiente. Esto simplifica el uso de la API para aplicaciones frontend.

### Cómo Funciona

1. **Usuario envía**: Departamento + Provincia
2. **Sistema mapea**: Busca el ubigeo correspondiente en TB_UBIGEOS.csv
3. **Modelo recibe**: Código ubigeo para la predicción
4. **Respuesta incluye**: Tanto los datos de entrada como el ubigeo calculado

### Ventajas

- **Interfaz amigable**: No necesitas conocer los códigos ubigeo
- **Validación automática**: El sistema verifica que la combinación sea válida
- **Transparente**: La respuesta muestra el ubigeo usado en la predicción

### Endpoints de Ubicación

```bash
# Obtener provincias de un departamento
GET /metadata/provincias/{departamento}

# Obtener ubigeo de departamento + provincia
GET /metadata/ubigeo/{departamento}/{provincia}
```

**Ejemplo:**

```bash
# Obtener provincias de LIMA
curl http://localhost:8000/metadata/provincias/LIMA
# Respuesta: {"departamento": "LIMA", "provincias": ["BARRANCA", "CAJATAMBO", ...]}

# Obtener ubigeo de LIMA-LIMA
curl http://localhost:8000/metadata/ubigeo/LIMA/LIMA
# Respuesta: {"ubigeo": 140101, "location": {...}}
```

## 🤖 IA Explicable (Explainable AI - XAI)

El sistema incluye un módulo de **IA Explicable** que utiliza GPT para generar explicaciones claras y concisas sobre las predicciones del modelo.

### Características del XAI

- **Contexto Situacional**: Explica por qué la predicción tiene ese nivel de riesgo
- **Acciones Específicas**: Recomienda 3 acciones preventivas concretas adaptadas al contexto
- **Factores Clave**: Identifica los principales factores que influyen en la predicción
- **Explicaciones Concisas**: Diseñadas para encajar perfectamente en interfaces de usuario

### Cómo Funciona

1. **Usuario solicita predicción** con explicación mediante `/predict/explain`
2. **Modelo genera predicción** estándar con tasa de positividad
3. **Servicio XAI analiza** los parámetros y el resultado
4. **GPT genera explicación** contextual y accionable
5. **Respuesta integrada** incluye predicción + explicación

### Ventajas

- **Transparencia**: Los usuarios entienden por qué se obtuvo ese resultado
- **Accionable**: Proporciona recomendaciones específicas para cada caso
- **Adaptativo**: Las explicaciones se ajustan al contexto geográfico y demográfico
- **Formato UI-friendly**: Explicaciones concisas que no rompen el diseño de la interfaz

### Configuración

Para habilitar XAI, necesitas una API key de Perplexity:

```bash
# Configurar variable de entorno
export PERPLEXITY_API_KEY="pplx-..."  # En Linux/Mac
set PERPLEXITY_API_KEY=pplx-...       # En Windows CMD
```

### Uso Responsable

- Las explicaciones son generadas por IA y deben ser interpretadas como guías orientativas
- Recomendamos validar las recomendaciones con expertos en salud mental
- El sistema usa **sonar** (Llama 3.3 70B) por defecto para balance entre calidad y costo

## 📡 Endpoints de la API

### Predicción Individual

```bash
POST /predict
```

**Ejemplo de Request:**

```json
{
  "NroMes": 11,
  "Departamento": "LIMA",
  "Provincia": "LIMA",
  "Sexo": "M",
  "Etapa": "5 - 9",
  "DetalleTamizaje": "VIOLENCIA FAMILIAR/MALTRATO INFANTIL"
}
```

**Ejemplo de Response:**

```json
{
  "tasa_positividad_predicha": 33.54,
  "interpretacion": "Riesgo Muy Alto - Intervención urgente requerida",
  "input_data": {
    "NroMes": 11,
    "Departamento": "LIMA",
    "Provincia": "LIMA",
    "Sexo": "M",
    "Etapa": "5 - 9",
    "DetalleTamizaje": "VIOLENCIA FAMILIAR/MALTRATO INFANTIL",
    "ubigeo": 140101
  }
}
```

### Predicción con Explicación (XAI)

```bash
POST /predict/explain
```

**Descripción**: Realiza una predicción e incluye una explicación generada por IA sobre el contexto, acciones recomendadas y factores clave.

**Requisito**: Requiere configurar `PERPLEXITY_API_KEY` como variable de entorno.

**Ejemplo de Request:**

```json
{
  "NroMes": 1,
  "Departamento": "ANCASH",
  "Provincia": "AIJA",
  "Sexo": "F",
  "Etapa": "< 1",
  "DetalleTamizaje": "SINDROME Y/O TRASTORNO PSICOTICO"
}
```

**Ejemplo de Response:**

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

### Predicción en Lote

```bash
POST /predict/batch
```

**Ejemplo de Request:**

```json
{
  "predictions": [
    {
      "NroMes": 11,
      "Departamento": "LIMA",
      "Provincia": "LIMA",
      "Sexo": "M",
      "Etapa": "5 - 9",
      "DetalleTamizaje": "VIOLENCIA FAMILIAR/MALTRATO INFANTIL"
    },
    {
      "NroMes": 7,
      "Departamento": "CUSCO",
      "Provincia": "CUSCO",
      "Sexo": "F",
      "Etapa": "30 - 39",
      "DetalleTamizaje": "TRASTORNO DEPRESIVO"
    }
  ]
}
```

### Información del Modelo

```bash
GET /model/info
```

**Response:**

```json
{
  "model_type": "gradient_boosting",
  "n_features": 43,
  "metrics": {
    "optimized_test": {
      "R2": 0.6789,
      "MAE": 8.34,
      "MSE": 120.45,
      "RMSE": 10.97
    }
  }
}
```

### Feature Importance

```bash
GET /model/features?top_n=10
```

### Metadatos

```bash
GET /metadata/departamentos           # Lista de departamentos válidos
GET /metadata/provincias/{dept}       # Lista de provincias por departamento
GET /metadata/ubigeo/{dept}/{prov}   # Obtener ubigeo de dept+provincia
GET /metadata/tamizajes               # Lista de tipos de tamizaje
GET /metadata/etapas                  # Lista de grupos etarios
```

### Health Check

```bash
GET /health
```

## 🧪 Pruebas con cURL

```bash
# Predicción individual
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "NroMes": 11,
    "Departamento": "LIMA",
    "Provincia": "LIMA",
    "Sexo": "M",
    "Etapa": "5 - 9",
    "DetalleTamizaje": "VIOLENCIA FAMILIAR/MALTRATO INFANTIL"
  }'

# Obtener provincias de un departamento
curl http://localhost:8000/metadata/provincias/LIMA

# Obtener ubigeo de departamento + provincia
curl http://localhost:8000/metadata/ubigeo/LIMA/LIMA

# Health check
curl http://localhost:8000/health

# Feature importance
curl http://localhost:8000/model/features?top_n=5
```

## 📊 Valores Válidos para Predicciones

### Departamentos
```
ANCASH, APURIMAC, AREQUIPA, AYACUCHO, CAJAMARCA, CALLAO, CUSCO,
HUANCAVELICA, HUANUCO, ICA, JUNIN, LA LIBERTAD, LAMBAYEQUE, LIMA,
LORETO, MADRE DE DIOS, MOQUEGUA, PASCO, PIURA, PUNO, SAN MARTIN,
TACNA, UCAYALI
```

### Tipos de Tamizaje
```
- SINDROME Y/O TRASTORNO PSICOTICO
- TRASTORNO DE CONSUMO DE ALCOHOL Y OTROS DROGAS
- TRASTORNO DEPRESIVO
- VIOLENCIA FAMILIAR/MALTRATO INFANTIL
```

### Grupos Etarios
```
< 1, 1 - 4, 5 - 9, 10 - 11, 12 - 14, 15 - 17, 18 - 24,
25 - 29, 30 - 39, 40 - 59, 60 - 79, 80  +
```

### Sexo
```
F (Femenino), M (Masculino)
```

### Mes (NroMes)
```
1-12 (Enero a Diciembre)
```

## 🔍 Interpretación de Resultados

La API clasifica automáticamente las predicciones en niveles de riesgo:

- **Riesgo Muy Bajo** (< 2%): Bajo requerimiento de recursos
- **Riesgo Bajo** (2-5%): Requerimiento normal de recursos
- **Riesgo Moderado** (5-10%): Incrementar disponibilidad de personal
- **Riesgo Alto** (10-20%): Priorizar asignación de especialistas
- **Riesgo Muy Alto** (> 20%): Intervención urgente requerida

## 📈 Métricas del Modelo

El modelo se evalúa con las siguientes métricas:

- **R² Score**: Coeficiente de determinación (calidad del ajuste)
- **MAE**: Error Absoluto Medio (diferencia promedio de predicciones)
- **RMSE**: Raíz del Error Cuadrático Medio (penaliza errores grandes)

Típicamente, el modelo logra:
- R² > 0.65 en el conjunto de test
- MAE < 10% de error promedio

## 🛠️ Desarrollo y Testing

### Ejecutar Tests Unitarios

```bash
# Instalar pytest
pip install pytest

# Ejecutar tests
pytest tests/
```

### Agregar Nuevas Features

1. Modifica `src/data_preparation.py` para incluir nuevas transformaciones
2. Re-entrena el modelo con `python src/train_model.py`
3. Actualiza la validación en `api/main.py` si es necesario

### Cambiar el Modelo

El proyecto soporta dos tipos de modelos:

```bash
# Gradient Boosting (por defecto, más preciso)
python src/train_model.py --model gradient_boosting

# Random Forest (más rápido)
python src/train_model.py --model random_forest
```

## 📝 Notas Importantes

### Datos Sensibles

- **NO** incluir el archivo `tamizajes.csv` en el control de versiones
- Agregar `*.csv` al `.gitignore` (excepto ejemplos pequeños)
- Los modelos entrenados (`*.pkl`) pueden ser versionados o no según el tamaño

### Producción

Para desplegar en producción:

1. Usar variables de entorno para configuración
2. Implementar autenticación (JWT, API Keys)
3. Configurar CORS apropiadamente
4. Usar un servidor ASGI como Gunicorn + Uvicorn
5. Implementar logging adecuado
6. Monitorear métricas del modelo

```bash
# Ejemplo de deploy con Gunicorn
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🤝 Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos.

## 👥 Autores

- Desarrollado como proyecto final del curso de Machine Learning

## 📧 Contacto

Para preguntas o sugerencias sobre el proyecto, contactar a [tu-email]

---

**Última actualización**: 2024
