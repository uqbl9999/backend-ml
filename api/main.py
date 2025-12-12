"""
FastAPI Application for Mental Health Screening Predictions
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, HttpUrl
from typing import List, Optional, Dict
import sys
import os
from dotenv import load_dotenv

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.prediction import Predictor
from src.services.ubigeo_service import get_ubigeo_service
from src.services.xai_service import get_xai_service
from src.services.statistics_service import get_statistics_service
from src.services.image_service import get_image_service

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Screening Prediction API",
    description="API para predecir la tasa de positividad de tamizajes de salud mental en Perú",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pkl')
predictor = None
ubigeo_service = None
xai_service = None
statistics_service = None
image_service = None


@app.on_event("startup")
async def startup_event():
    """Cargar modelo y servicios al iniciar"""
    global predictor, ubigeo_service, xai_service, statistics_service, image_service
    try:
        predictor = Predictor(MODEL_PATH)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo cargar el modelo: {e}")
        print("    La API iniciará pero las predicciones no estarán disponibles")

    try:
        ubigeo_service = get_ubigeo_service()
        print("✅ Servicio de ubigeo cargado correctamente")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo cargar el servicio de ubigeo: {e}")
        print("    El mapeo de ubicación no estará disponible")

    try:
        xai_service = get_xai_service()
        if xai_service:
            print("✅ Servicio XAI cargado correctamente")
        else:
            print("⚠️  Advertencia: Servicio XAI no disponible (PERPLEXITY_API_KEY no configurada)")
            print("    Las funciones de IA explicable no estarán disponibles")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo cargar el servicio XAI: {e}")
        print("    Las funciones de IA explicable no estarán disponibles")

    try:
        statistics_service = get_statistics_service()
        if statistics_service:
            print("✅ Servicio de estadísticas cargado correctamente")
        else:
            print("⚠️  Advertencia: Servicio de estadísticas no disponible")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo cargar el servicio de estadísticas: {e}")
        print("    Las funciones de estadísticas no estarán disponibles")

    # Cargar servicio de imágenes
    try:
        image_service = get_image_service()
        if image_service:
            print("✅ Servicio de reconocimiento de imágenes cargado correctamente")
        else:
            print("⚠️  Advertencia: Servicio de imágenes no disponible (modelo no encontrado)")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo cargar el servicio de imágenes: {e}")
        print("    Los endpoints de imágenes no estarán disponibles")


# Request models
class PredictionInput(BaseModel):
    """Datos de entrada para predicción"""
    NroMes: int = Field(..., ge=1, le=12, description="Mes del año (1-12)")
    Departamento: str = Field(..., description="Departamento del Perú")
    Provincia: str = Field(..., description="Provincia del Perú")
    Sexo: str = Field(..., description="Sexo (F o M)")
    Etapa: str = Field(..., description="Grupo etario")
    DetalleTamizaje: str = Field(..., description="Tipo de tamizaje")
    ubigeo: Optional[int] = Field(None, description="Código de ubigeo (se calcula automáticamente si no se provee)")

    @field_validator('Sexo')
    @classmethod
    def validate_sexo(cls, v):
        if v not in ['F', 'M']:
            raise ValueError('Sexo debe ser F o M')
        return v

    @field_validator('Departamento')
    @classmethod
    def validate_departamento(cls, v):
        departamentos_validos = [
            'ANCASH', 'APURIMAC', 'AREQUIPA', 'AYACUCHO', 'CAJAMARCA',
            'CALLAO', 'CUSCO', 'HUANCAVELICA', 'HUANUCO', 'ICA',
            'JUNIN', 'LA LIBERTAD', 'LAMBAYEQUE', 'LIMA', 'LORETO',
            'MADRE DE DIOS', 'MOQUEGUA', 'PASCO', 'PIURA', 'PUNO',
            'SAN MARTIN', 'TACNA', 'UCAYALI'
        ]
        if v.upper() not in departamentos_validos:
            raise ValueError(f'Departamento no válido. Debe ser uno de: {", ".join(departamentos_validos)}')
        return v.upper()

    @field_validator('DetalleTamizaje')
    @classmethod
    def validate_detalle_tamizaje(cls, v):
        tipos_validos = [
            'SINDROME Y/O TRASTORNO PSICOTICO',
            'TRASTORNO DE CONSUMO DE ALCOHOL Y OTROS DROGAS',
            'TRASTORNO DEPRESIVO',
            'VIOLENCIA FAMILIAR/MALTRATO INFANTIL'
        ]
        if v not in tipos_validos:
            raise ValueError(f'DetalleTamizaje no válido. Debe ser uno de: {", ".join(tipos_validos)}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "NroMes": 11,
                "Departamento": "LIMA",
                "Provincia": "LIMA",
                "Sexo": "MASCULINO",
                "Etapa": "NIÑO",
                "DetalleTamizaje": "VIOLENCIA FAMILIAR"
            }
        }


class PredictionOutput(BaseModel):
    """Datos de salida de la predicción"""
    tasa_positividad_predicha: float = Field(..., description="Tasa de positividad predicha (%)")
    interpretacion: str = Field(..., description="Interpretación del nivel de riesgo")
    input_data: dict = Field(..., description="Datos de entrada utilizados")


class BatchPredictionInput(BaseModel):
    """Entrada para predicciones por lote"""
    predictions: List[PredictionInput]


class ScreeningModelInfoOutput(BaseModel):
    """Salida de información del modelo de screening"""
    model_type: str
    n_features: int
    metrics: dict


class XAIExplanationOutput(BaseModel):
    """Salida para explicación XAI"""
    contexto_situacional: str = Field(..., description="Explicación contextual del riesgo")
    acciones: List[str] = Field(..., description="Lista de acciones preventivas recomendadas")
    factores_clave: List[str] = Field(..., description="Factores clave que influyen en la predicción")


class PredictionWithXAIOutput(BaseModel):
    """Salida de predicción con explicación XAI"""
    tasa_positividad_predicha: float = Field(..., description="Tasa de positividad predicha (%)")
    interpretacion: str = Field(..., description="Interpretación del nivel de riesgo")
    input_data: dict = Field(..., description="Datos de entrada utilizados")
    explicacion: Optional[XAIExplanationOutput] = Field(None, description="Explicación de IA explicable")


# ============================================================================
# IMAGE RECOGNITION MODELS
# ============================================================================

class ImagePredictionOutput(BaseModel):
    """Respuesta de predicción de imagen"""
    predicted_class: str = Field(..., description="Clase predicha")
    confidence: float = Field(..., ge=0, le=1, description="Confianza (0-1)")
    interpretation: str = Field(..., description="Interpretación del resultado")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilidades de todas las clases")
    metadata: Dict = Field(..., description="Metadata del procesamiento")


class ImageURLInput(BaseModel):
    """Input para predicción desde URL"""
    image_url: HttpUrl = Field(..., description="URL de la imagen de rayos X")


class ImageXAIOutput(BaseModel):
    """Salida de predicción con explicación XAI"""
    predicted_class: str
    confidence: float
    interpretation: str
    all_probabilities: Dict[str, float]
    metadata: Dict
    explicacion: Optional[Dict] = Field(None, description="Explicación médica XAI")


class ImageModelInfoOutput(BaseModel):
    """Información del modelo CNN"""
    model_type: str
    framework: str
    input_shape: List[int]
    num_classes: int
    classes: List[str]
    architecture: Dict
    training_info: Dict


class ClassInfoOutput(BaseModel):
    """Información de una clase"""
    class_name: str
    description: str


class ModelStatisticsOutput(BaseModel):
    """Estadísticas del modelo"""
    test_accuracy: float
    test_loss: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]


# API Endpoints

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Mental Health Screening Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "screening": {
                "predict": "/predict",
                "predict_batch": "/predict/batch",
                "predict_with_explanation": "/predict/explain",
                "model_info": "/model/info",
                "feature_importance": "/model/features"
            },
            "image_recognition": {
                "predict_upload": "/image/predict",
                "predict_url": "/image/predict-url",
                "predict_with_explanation": "/image/predict/explain",
                "model_info": "/image/model/info",
                "classes": "/image/model/classes",
                "statistics": "/image/model/statistics"
            },
            "health": "/health",
            "statistics": {
                "descriptive_stats": "/statistics/descriptive",
                "distribution_by_groups": "/statistics/distribution",
                "heatmap_by_screening_type": "/statistics/heatmap/screening-type",
                "heatmap_by_department": "/statistics/heatmap/department",
                "screening_types_summary": "/statistics/screening-types",
                "department_summary": "/statistics/departments"
            }
        }
    }


@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    model_loaded = predictor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Realizar una predicción individual

    Predice la tasa de positividad de tamizajes de salud mental
    basándose en características demográficas, geográficas y temporales.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    if ubigeo_service is None:
        raise HTTPException(status_code=503, detail="Servicio de ubigeo no cargado")

    try:
        # Convert Pydantic model to dict
        input_dict = input_data.dict()

        # If ubigeo is not provided, calculate it from Departamento + Provincia
        if input_dict.get('ubigeo') is None:
            departamento = input_dict.get('Departamento')
            provincia = input_dict.get('Provincia')

            if not departamento or not provincia:
                raise HTTPException(
                    status_code=400,
                    detail="Debe proporcionar Departamento y Provincia, o un ubigeo válido"
                )

            # Get ubigeo from department and province
            ubigeo = ubigeo_service.get_ubigeo_by_dept_prov(departamento, provincia)

            if ubigeo is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"No se encontró ubigeo para {departamento} - {provincia}"
                )

            input_dict['ubigeo'] = ubigeo

        # Validate input
        validation_result = predictor.validate_input(input_dict)
        if not validation_result['is_valid']:
            raise HTTPException(status_code=400, detail=validation_result['errors'])

        # Make prediction
        result = predictor.predict_single(input_dict)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de predicción: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Realizar predicciones en lote

    Permite predecir múltiples casos de manera eficiente.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        # Convert Pydantic models to dicts
        input_list = [item.dict() for item in batch_input.predictions]

        # Make predictions
        results = predictor.predict_batch(input_list)

        return {
            "predictions": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción por lote: {str(e)}")


@app.post("/predict/explain", response_model=PredictionWithXAIOutput)
async def predict_with_explanation(input_data: PredictionInput):
    """
    Realizar predicción con explicación de IA explicable (XAI)

    Predice la tasa de positividad e incluye una explicación generada por IA
    que detalla el contexto situacional, acciones recomendadas y factores clave.

    Requiere configurar PERPLEXITY_API_KEY en las variables de entorno.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    if ubigeo_service is None:
        raise HTTPException(status_code=503, detail="Servicio de ubigeo no cargado")

    try:
        # Convert Pydantic model to dict
        input_dict = input_data.dict()

        # If ubigeo is not provided, calculate it from Departamento + Provincia
        if input_dict.get('ubigeo') is None:
            departamento = input_dict.get('Departamento')
            provincia = input_dict.get('Provincia')

            if not departamento or not provincia:
                raise HTTPException(
                    status_code=400,
                    detail="Debe proporcionar Departamento y Provincia, o un ubigeo válido"
                )

            # Get ubigeo from department and province
            ubigeo = ubigeo_service.get_ubigeo_by_dept_prov(departamento, provincia)

            if ubigeo is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"No se encontró ubigeo para {departamento} - {provincia}"
                )

            input_dict['ubigeo'] = ubigeo

        # Validate input
        validation_result = predictor.validate_input(input_dict)
        if not validation_result['is_valid']:
            raise HTTPException(status_code=400, detail=validation_result['errors'])

        # Make prediction
        result = predictor.predict_single(input_dict)

        # Generate XAI explanation if service is available
        if xai_service:
            xai_result = xai_service.generate_explanation(
                params=result['input_data'],
                prediction=result['tasa_positividad_predicha'],
                interpretation=result['interpretacion']
            )

            if xai_result['success']:
                result['explicacion'] = xai_result['explanation']
            else:
                # Log the error for debugging
                print(f"⚠️  Error de XAI: {xai_result.get('error', 'Error desconocido')}")
                # Use fallback explanation if XAI fails
                result['explicacion'] = xai_result['explanation']
        else:
            # Servicio XAI no disponible
            raise HTTPException(
                status_code=503,
                detail="Servicio XAI no disponible. Configure la variable de entorno PERPLEXITY_API_KEY."
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de predicción: {str(e)}")


@app.get("/model/info", response_model=ScreeningModelInfoOutput)
async def get_model_info():
    """
    Obtener información del modelo

    Retorna información sobre el tipo de modelo, número de features y métricas de evaluación.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        info = predictor.get_model_info()
        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener info del modelo: {str(e)}")


@app.get("/model/features")
async def get_feature_importance(top_n: int = 10):
    """
    Obtener las características más importantes del modelo

    Parámetros:
    - top_n: Número de características a retornar (default: 10)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        if top_n < 1 or top_n > 50:
            raise HTTPException(status_code=400, detail="top_n debe estar entre 1 y 50")

        features = predictor.get_feature_importance(top_n=top_n)

        return {
            "top_features": features,
            "count": len(features)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener importancia de características: {str(e)}")


@app.get("/metadata/departamentos")
async def get_departamentos():
    """Obtener lista de departamentos válidos"""
    return {
        "departamentos": [
            'ANCASH', 'APURIMAC', 'AREQUIPA', 'AYACUCHO', 'CAJAMARCA',
            'CALLAO', 'CUSCO', 'HUANCAVELICA', 'HUANUCO', 'ICA',
            'JUNIN', 'LA LIBERTAD', 'LAMBAYEQUE', 'LIMA', 'LORETO',
            'MADRE DE DIOS', 'MOQUEGUA', 'PASCO', 'PIURA', 'PUNO',
            'SAN MARTIN', 'TACNA', 'UCAYALI'
        ]
    }


@app.get("/metadata/tamizajes")
async def get_tipos_tamizaje():
    """Obtener lista de tipos de tamizaje válidos"""
    return {
        "tipos_tamizaje": [
            'SINDROME Y/O TRASTORNO PSICOTICO',
            'TRASTORNO DE CONSUMO DE ALCOHOL Y OTROS DROGAS',
            'TRASTORNO DEPRESIVO',
            'VIOLENCIA FAMILIAR/MALTRATO INFANTIL'
        ]
    }


@app.get("/metadata/etapas")
async def get_etapas():
    """Obtener lista de grupos etarios válidos"""
    return {
        "etapas": [
            '< 1', '1 - 4', '5 - 9', '10 - 11', '12 - 14',
            '15 - 17', '18 - 24', '25 - 29', '30 - 39',
            '40 - 59', '60 - 79', '80  +'
        ]
    }


@app.get("/metadata/provincias/{departamento}")
async def get_provincias(departamento: str):
    """
    Obtener lista de provincias para un departamento específico

    Parameters:
    - departamento: Nombre del departamento
    """
    if ubigeo_service is None:
        raise HTTPException(status_code=503, detail="Servicio de ubigeo no cargado")

    try:
        provincias = ubigeo_service.get_provincias_by_departamento(departamento)

        if not provincias:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron provincias para el departamento: {departamento}"
            )

        return {
            "departamento": departamento.upper(),
            "provincias": provincias
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener provincias: {str(e)}")


@app.get("/metadata/ubigeo/{departamento}/{provincia}")
async def get_ubigeo(departamento: str, provincia: str):
    """
    Obtener código de ubigeo para un departamento y provincia

    Parameters:
    - departamento: Nombre del departamento
    - provincia: Nombre de la provincia
    """
    if ubigeo_service is None:
        raise HTTPException(status_code=503, detail="Servicio de ubigeo no cargado")

    try:
        ubigeo = ubigeo_service.get_ubigeo_by_dept_prov(departamento, provincia)

        if ubigeo is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ubigeo para {departamento} - {provincia}"
            )

        location_info = ubigeo_service.get_location_info(ubigeo)

        return {
            "ubigeo": ubigeo,
            "location": location_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener ubigeo: {str(e)}")


@app.get("/statistics/descriptive")
async def get_descriptive_statistics():
    """
    Obtener estadísticas descriptivas sobre la tasa de positividad

    Retorna media, mediana, desviación estándar y máximo de la tasa de positividad.
    """
    if statistics_service is None:
        raise HTTPException(status_code=503, detail="Servicio de estadísticas no disponible")

    try:
        stats = statistics_service.get_descriptive_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas descriptivas: {str(e)}")


@app.get("/statistics/distribution")
async def get_distribution_by_groups():
    """
    Obtener distribución por grupos de tamizaje

    Retorna la distribución de registros y suma de casos por grupo:
    - Total de Tamizajes
    - Solo Tamizajes Positivos
    - Tamizajes con Condición Adicional Violencia Política
    """
    if statistics_service is None:
        raise HTTPException(status_code=503, detail="Servicio de estadísticas no disponible")

    try:
        distribution = statistics_service.get_distribution_by_groups()
        return distribution
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener distribución: {str(e)}")


@app.get("/statistics/heatmap/screening-type")
async def get_heatmap_by_screening_type(grupo: Optional[str] = None):
    """
    Obtener heatmap de casos agregados por grupo de tamizaje y tipo específico

    Parámetros:
    - grupo: Filtro opcional por grupo ('TOTAL', 'POSITIVOS', o 'VIOLENCIA')

    Retorna casos agregados por:
    - Violencia Familiar
    - Maltrato Infantil
    - Trastorno Depresivo
    - Consumo de Alcohol y Drogas
    - Trastorno Psicótico
    - Violencia Política (si aplica)
    """
    if statistics_service is None:
        raise HTTPException(status_code=503, detail="Servicio de estadísticas no disponible")

    try:
        heatmap = statistics_service.get_heatmap_by_screening_type(grupo=grupo)
        return {
            "grupo_filtro": grupo if grupo else "todos",
            "data": heatmap
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener heatmap por tipo: {str(e)}")


@app.get("/statistics/heatmap/department")
async def get_heatmap_by_department(
    grupo: Optional[str] = None,
    top_n: Optional[int] = None
):
    """
    Obtener heatmap de casos agregados por departamento y grupo

    Parámetros:
    - grupo: Filtro opcional por grupo ('TOTAL', 'POSITIVOS', o 'VIOLENCIA')
    - top_n: Limitar a los top N departamentos con más casos

    Retorna volumen de casos por departamento y grupo de tamizaje.
    Los departamentos se ordenan por total de casos (descendente).
    """
    if statistics_service is None:
        raise HTTPException(status_code=503, detail="Servicio de estadísticas no disponible")

    try:
        if top_n is not None and (top_n < 1 or top_n > 50):
            raise HTTPException(status_code=400, detail="top_n debe estar entre 1 y 50")

        heatmap = statistics_service.get_heatmap_by_department(grupo=grupo, top_n=top_n)
        return {
            "grupo_filtro": grupo if grupo else "todos",
            "top_n": top_n if top_n else "todos",
            "data": heatmap
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener heatmap por departamento: {str(e)}")


@app.get("/statistics/screening-types")
async def get_screening_types_summary():
    """
    Obtener resumen de tipos de tamizaje con estadísticas

    Retorna para cada tipo de tamizaje:
    - Total de registros
    - Suma total de casos
    - Suma de positivos
    - Tasa de positividad promedio, mediana y máxima

    Los resultados se ordenan por suma total de casos (descendente).
    """
    if statistics_service is None:
        raise HTTPException(status_code=503, detail="Servicio de estadísticas no disponible")

    try:
        summary = statistics_service.get_screening_types_summary()
        return {
            "count": len(summary),
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener resumen de tipos: {str(e)}")


@app.get("/statistics/departments")
async def get_department_summary(top_n: Optional[int] = None):
    """
    Obtener resumen de departamentos con estadísticas

    Parámetros:
    - top_n: Limitar a los top N departamentos con más casos

    Retorna para cada departamento:
    - Total de registros
    - Suma total de casos
    - Suma de positivos
    - Tasa de positividad promedio, mediana y máxima

    Los resultados se ordenan por suma total de casos (descendente).
    """
    if statistics_service is None:
        raise HTTPException(status_code=503, detail="Servicio de estadísticas no disponible")

    try:
        if top_n is not None and (top_n < 1 or top_n > 50):
            raise HTTPException(status_code=400, detail="top_n debe estar entre 1 y 50")

        summary = statistics_service.get_department_summary(top_n=top_n)
        return {
            "count": len(summary),
            "top_n": top_n if top_n else "todos",
            "data": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener resumen de departamentos: {str(e)}")


# ============================================================================
# ENDPOINTS DE RECONOCIMIENTO DE IMÁGENES MÉDICAS
# ============================================================================

@app.post("/image/predict", response_model=ImagePredictionOutput)
async def predict_image(file: UploadFile = File(...)):
    """
    Predicción de rayos X desde archivo upload

    Acepta imágenes en formato JPG, JPEG, PNG (máx 10 MB)
    """
    if image_service is None:
        raise HTTPException(status_code=503, detail="Servicio de imágenes no disponible")

    # Validar content type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        # Leer bytes
        file_bytes = await file.read()

        # Validar tamaño (10 MB máx)
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen demasiado grande (máx 10 MB)")

        # Predicción
        result = image_service.predict_from_upload(file_bytes, file.filename)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.post("/image/predict-url", response_model=ImagePredictionOutput)
async def predict_image_from_url(input_data: ImageURLInput):
    """
    Predicción de rayos X desde URL

    Descarga la imagen desde la URL proporcionada y realiza la predicción
    """
    if image_service is None:
        raise HTTPException(status_code=503, detail="Servicio de imágenes no disponible")

    try:
        result = image_service.predict_from_url(str(input_data.image_url))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen desde URL: {str(e)}")


@app.post("/image/predict/explain", response_model=ImageXAIOutput)
async def predict_image_with_explanation(file: UploadFile = File(...)):
    """
    Predicción de rayos X con explicación médica XAI

    Genera una predicción y una explicación detallada usando IA explicable.
    Requiere PERPLEXITY_API_KEY configurada.
    """
    if image_service is None:
        raise HTTPException(status_code=503, detail="Servicio de imágenes no disponible")

    if xai_service is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio XAI no disponible. Configure PERPLEXITY_API_KEY."
        )

    try:
        # Predicción base
        file_bytes = await file.read()

        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen demasiado grande (máx 10 MB)")

        result = image_service.predict_from_upload(file_bytes, file.filename)

        # Generar explicación XAI
        xai_result = image_service.generate_explanation_with_xai(result, xai_service)

        if xai_result['success']:
            result['explicacion'] = xai_result['explanation']
        else:
            result['explicacion'] = xai_result.get('explanation', None)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/image/model/info", response_model=ImageModelInfoOutput)
async def get_image_model_info():
    """
    Información del modelo CNN desde HF Space

    Retorna arquitectura, parámetros de entrenamiento y detalles técnicos.
    """
    if image_service is None:
        raise HTTPException(status_code=503, detail="Servicio de imágenes no disponible")

    try:
        info = image_service.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/image/model/classes", response_model=List[ClassInfoOutput])
async def get_image_classes():
    """
    Lista de clases con descripciones médicas

    Retorna información sobre las 4 clases: COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS
    """
    if image_service is None:
        raise HTTPException(status_code=503, detail="Servicio de imágenes no disponible")

    try:
        classes = image_service.get_class_info()
        return classes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/image/model/statistics", response_model=ModelStatisticsOutput)
async def get_image_model_statistics():
    """
    Estadísticas detalladas del modelo

    Retorna accuracy, métricas por clase (precision, recall, F1) y matriz de confusión.
    """
    if image_service is None:
        raise HTTPException(status_code=503, detail="Servicio de imágenes no disponible")

    try:
        stats = image_service.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
