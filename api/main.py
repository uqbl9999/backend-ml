"""
FastAPI Application for Mental Health Screening Predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.prediction import Predictor
from src.services.ubigeo_service import get_ubigeo_service
from src.services.xai_service import get_xai_service
from src.services.statistics_service import get_statistics_service

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


@app.on_event("startup")
async def startup_event():
    """Cargar modelo y servicios al iniciar"""
    global predictor, ubigeo_service, xai_service, statistics_service
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


class ModelInfoOutput(BaseModel):
    """Salida de información del modelo"""
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


# API Endpoints

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Mental Health Screening Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "predict_with_explanation": "/predict/explain",
            "model_info": "/model/info",
            "feature_importance": "/model/features",
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


@app.get("/model/info", response_model=ModelInfoOutput)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
