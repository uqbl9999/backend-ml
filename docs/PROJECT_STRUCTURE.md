# Documentación de la Estructura del Proyecto

## 📂 Organización de Directorios

```
backend-ml/
│
├── 📁 data/                          # Almacenamiento de datos
│   ├── dataset_limpio.csv            # Datos limpios
│   ├── df_clean_to_model.csv         # Características codificadas
│   ├── dataset_balanceado.csv        # Dataset balanceado
│   ├── tamizajes.csv                 # Datos originales
│   └── TB_UBIGEOS.csv               # Tabla de ubigeos de Perú
│
├── 📁 src/                           # Código fuente
│   ├── data_preparation.py           # Pipeline de procesamiento de datos
│   ├── train_model.py               # Script de entrenamiento
│   ├── example_prediction.py        # Ejemplo de uso
│   ├── 📁 models/                    # Módulos del modelo
│   │   ├── __init__.py
│   │   ├── training.py               # Lógica de entrenamiento
│   │   └── prediction.py             # Lógica de predicción
│   └── 📁 services/                  # Servicios adicionales
│       ├── __init__.py
│       ├── ubigeo_service.py         # Servicio de mapeo de ubigeos
│       ├── xai_service.py            # Servicio de IA Explicable
│       └── statistics_service.py     # Servicio de estadísticas
│
├── 📁 api/                           # API REST
│   └── main.py                       # Aplicación FastAPI
│
├── 📁 models/                        # Modelos entrenados
│   └── trained_model.pkl             # Modelo serializado
│
├── 📁 docs/                          # Documentación y gráficos
│   ├── evaluation_*.png              # Gráficos de evaluación
│   ├── PROJECT_STRUCTURE.md          # Este archivo
│   ├── XAI_GUIDE.md                 # Guía de IA Explicable
│   └── STATISTICS_API.md            # Documentación de API de estadísticas
│
├── 📁 tests/                         # Pruebas unitarias
│   └── test_prediction.py            # Pruebas de predicción
│
├── 📁 notebooks/                     # Jupyter notebooks
│   └── parcialfinal.ipynb           # Exploración original
│
├── 📄 requirements.txt               # Dependencias de Python
├── 📄 README.md                      # Documentación principal
├── 📄 QUICKSTART.md                  # Guía de inicio rápido
├── 📄 START_HERE.md                  # Orientación inicial
├── 📄 PROJECT_SUMMARY.md             # Resumen ejecutivo
└── 📄 .gitignore                     # Reglas de Git ignore
```

## 🔄 Flujo de Datos

```
┌─────────────────┐
│  tamizajes.csv  │  Datos Originales
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  data_preparation.py        │
│  ┌─────────────────────┐   │
│  │ 1. Cargar Datos     │   │
│  │ 2. Calcular Tasa    │   │
│  │ 3. Limpiar Datos    │   │
│  │ 4. Ing. Features    │   │
│  │ 5. Balancear Datos  │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  training.py                │
│  ┌─────────────────────┐   │
│  │ 1. Dividir Datos    │   │
│  │ 2. Entrenar Modelo  │   │
│  │ 3. Optimizar Params │   │
│  │ 4. Evaluar          │   │
│  │ 5. Guardar Modelo   │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  trained_model.pkl          │  Modelo Guardado
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  prediction.py              │
│  ┌─────────────────────┐   │
│  │ 1. Cargar Modelo    │   │
│  │ 2. Preparar Features│   │
│  │ 3. Hacer Predicción │   │
│  │ 4. Interpretar      │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  FastAPI (main.py)          │
│  ┌─────────────────────┐   │
│  │ Endpoints REST      │   │
│  │ - /predict          │   │
│  │ - /predict/explain  │   │
│  │ - /predict/batch    │   │
│  │ - /statistics/*     │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Cliente (Web/Mobile/API)   │
└─────────────────────────────┘
```

## 🏗️ Patrón de Arquitectura

Este proyecto sigue una **arquitectura de capas simplificada**:

### 1. Capa de Datos (`src/data_preparation.py`)
- **Responsabilidad**: Carga, limpieza y transformación de datos
- **Entrada**: Archivos CSV originales
- **Salida**: Datasets procesados y balanceados listos para ML

### 2. Capa de Modelo (`src/models/`)
- **training.py**
  - **Responsabilidad**: Entrenamiento, optimización y evaluación del modelo
  - **Entrada**: Datasets preparados
  - **Salida**: Modelo entrenado (archivo .pkl)

- **prediction.py**
  - **Responsabilidad**: Cargar modelo y realizar predicciones
  - **Entrada**: Diccionario de características
  - **Salida**: Predicción + interpretación

### 3. Capa de Servicios (`src/services/`)
- **ubigeo_service.py**
  - **Responsabilidad**: Mapeo de ubicaciones geográficas
  - **Funcionalidad**: Convertir Departamento + Provincia a Ubigeo

- **xai_service.py**
  - **Responsabilidad**: Generar explicaciones de predicciones
  - **Funcionalidad**: Usar Perplexity AI para explicar resultados

- **statistics_service.py**
  - **Responsabilidad**: Calcular estadísticas descriptivas
  - **Funcionalidad**: Heatmaps, distribuciones, resúmenes

### 4. Capa de API (`api/main.py`)
- **Responsabilidad**: Endpoints REST, validación, manejo de errores
- **Entrada**: Peticiones HTTP (JSON)
- **Salida**: Respuestas HTTP (JSON)
- **Framework**: FastAPI

### 5. Capa de Interfaz (Externa)
- **Responsabilidad**: Interacción con el usuario
- **Herramientas**: Swagger UI, curl, aplicaciones cliente

## 🔌 Arquitectura de Endpoints de la API

```
Aplicación FastAPI (main.py)
│
├── Middleware
│   └── CORS
│
├── Eventos de Inicio
│   ├── Cargar Modelo
│   ├── Cargar Servicio de Ubigeo
│   ├── Cargar Servicio XAI (opcional)
│   └── Cargar Servicio de Estadísticas
│
├── Endpoints de Salud e Info
│   ├── GET /
│   ├── GET /health
│   └── GET /model/info
│
├── Endpoints de Predicción
│   ├── POST /predict          → predict_single()
│   ├── POST /predict/explain  → predict_with_explanation()
│   └── POST /predict/batch    → predict_batch()
│
├── Endpoints de Info del Modelo
│   └── GET /model/features    → get_feature_importance()
│
├── Endpoints de Metadatos
│   ├── GET /metadata/departamentos
│   ├── GET /metadata/provincias/{dept}
│   ├── GET /metadata/ubigeo/{dept}/{prov}
│   ├── GET /metadata/tamizajes
│   └── GET /metadata/etapas
│
└── Endpoints de Estadísticas
    ├── GET /statistics/descriptive
    ├── GET /statistics/distribution
    ├── GET /statistics/heatmap/screening-type
    ├── GET /statistics/heatmap/department
    ├── GET /statistics/screening-types
    └── GET /statistics/departments
```

## 🧩 Diagrama de Clases

```
┌──────────────────────────┐
│   DataPreparation        │
├──────────────────────────┤
│ - data_path              │
│ - df                     │
│ - df_pivot               │
│ - df_clean               │
│ - df_encoded             │
├──────────────────────────┤
│ + load_data()            │
│ + calculate_positivity() │
│ + clean_data()           │
│ + feature_engineering()  │
│ + balance_data()         │
│ + prepare_full_pipeline()│
└──────────────────────────┘

┌──────────────────────────┐
│   ModelTrainer           │
├──────────────────────────┤
│ - model_type             │
│ - model                  │
│ - X_train, X_test        │
│ - y_train, y_test        │
│ - feature_names          │
│ - metrics                │
├──────────────────────────┤
│ + split_data()           │
│ + train_base_model()     │
│ + optimize_hyperparams() │
│ + save_model()           │
│ + load_model()           │
│ + predict()              │
│ + plot_results()         │
└──────────────────────────┘

┌──────────────────────────┐
│   Predictor              │
├──────────────────────────┤
│ - model_path             │
│ - model                  │
│ - feature_names          │
│ - model_type             │
│ - metrics                │
├──────────────────────────┤
│ + load_model()           │
│ + predict_single()       │
│ + predict_batch()        │
│ + validate_input()       │
│ + get_feature_import()   │
│ + get_model_info()       │
└──────────────────────────┘
```

## 🔐 Consideraciones de Seguridad

Implementación actual (Desarrollo):
- ✅ Validación de entrada (modelos Pydantic)
- ✅ CORS habilitado para todos los orígenes
- ❌ Sin autenticación
- ❌ Sin rate limiting
- ❌ Sin logging

Recomendado para Producción:
- 🔒 Agregar autenticación JWT
- 🔒 Implementar sistema de API keys
- 🔒 Agregar rate limiting
- 🔒 Restringir orígenes CORS
- 🔒 Agregar logging completo
- 🔒 Usar HTTPS
- 🔒 Agregar sanitización de entrada
- 🔒 Implementar monitoreo

## 📊 Pipeline del Modelo

```
Fase de Entrenamiento:
┌────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│   Datos    │ -> │  Limpiar &│ -> │ Balancear│ -> │  Entrenar  │
│  Originales│    │  Codificar│    │          │    │            │
└────────────┘    └───────────┘    └──────────┘    └────────────┘
                                                           │
                                                           ▼
                                                    ┌────────────┐
                                                    │  Guardar   │
                                                    │ Model.pkl  │
                                                    └────────────┘

Fase de Predicción:
┌────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│  Entrada   │ -> │  Preparar │ -> │ Predecir │ -> │ Interpretar│
│   JSON     │    │  Features │    │          │    │            │
└────────────┘    └───────────┘    └──────────┘    └────────────┘
```

## 🎯 Decisiones de Diseño

### ¿Por Qué Esta Estructura?

1. **Separación de Responsabilidades**
   - La preparación de datos es independiente del entrenamiento
   - La lógica de predicción está separada de la lógica de la API
   - Fácil modificar un componente sin afectar otros

2. **Simplicidad Primero**
   - Se eligió estructura simple sobre DDD complejo
   - Apropiado para proyecto académico/pequeña escala
   - Fácil de entender y mantener

3. **Camino de Escalabilidad**
   - Estructura clara permite migración fácil a DDD si se necesita
   - Se pueden agregar capas (caché, colas) sin refactorización mayor
   - Diseño de API soporta múltiples clientes

4. **Testeabilidad**
   - Cada módulo puede ser testeado independientemente
   - Fácil crear mocks de datos/modelos
   - Pruebas unitarias para funciones críticas

### ¿Por Qué FastAPI?

- ✅ Documentación automática de API (Swagger/ReDoc)
- ✅ Validación de tipos con Pydantic
- ✅ Soporte asíncrono (escalabilidad futura)
- ✅ Python moderno (3.8+)
- ✅ Alto rendimiento
- ✅ Fácil de aprender

### ¿Por Qué Pickle para el Modelo?

- ✅ Serialización estándar de scikit-learn
- ✅ Preserva todo el estado del modelo
- ✅ Fácil de cargar y usar
- ⚠️  No seguro para fuentes no confiables
- ⚠️  Dependiente de la versión de Python

Alternativa: ONNX (para producción/multiplataforma)

## 📈 Mejoras Futuras

Mejoras potenciales:

1. **Agregar Capa de Caché** (Redis)
   - Cachear predicciones frecuentes
   - Almacenar modelo en memoria

2. **Agregar Base de Datos** (PostgreSQL)
   - Almacenar historial de predicciones
   - Gestión de usuarios
   - Analíticas

3. **Agregar Cola de Mensajes** (RabbitMQ/Celery)
   - Predicciones en lote asíncronas
   - Trabajos de reentrenamiento del modelo

4. **Agregar Monitoreo** (Prometheus/Grafana)
   - Métricas de la API
   - Drift del rendimiento del modelo
   - Seguimiento de errores

5. **Containerización** (Docker)
   - Despliegue fácil
   - Consistencia de entorno

6. **Pipeline CI/CD** (GitHub Actions)
   - Pruebas automatizadas
   - Despliegue automatizado

## 📚 Documentación Relacionada

- [README.md](../README.md) - Documentación principal
- [QUICKSTART.md](../QUICKSTART.md) - Guía de inicio rápido
- [XAI_GUIDE.md](XAI_GUIDE.md) - Guía de IA Explicable
- [STATISTICS_API.md](STATISTICS_API.md) - Documentación de API de estadísticas
- [Documentación de API](http://localhost:8000/docs) - Docs interactivas de la API (cuando esté corriendo)
