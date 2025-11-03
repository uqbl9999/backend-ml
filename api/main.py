"""
FastAPI Application for Mental Health Screening Predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.prediction import Predictor

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


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    try:
        predictor = Predictor(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("    API will start but predictions will not be available")


# Request models
class PredictionInput(BaseModel):
    """Input data for prediction"""
    NroMes: int = Field(..., ge=1, le=12, description="Mes del año (1-12)")
    ubigeo: Optional[int] = Field(None, description="Código de ubigeo")
    Departamento: str = Field(..., description="Departamento del Perú")
    Sexo: str = Field(..., description="Sexo (F o M)")
    Etapa: str = Field(..., description="Grupo etario")
    DetalleTamizaje: str = Field(..., description="Tipo de tamizaje")

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
                "NroMes": 5,
                "ubigeo": 150101,
                "Departamento": "LIMA",
                "Sexo": "F",
                "Etapa": "30 - 39",
                "DetalleTamizaje": "TRASTORNO DEPRESIVO"
            }
        }


class PredictionOutput(BaseModel):
    """Output data for prediction"""
    tasa_positividad_predicha: float = Field(..., description="Tasa de positividad predicha (%)")
    interpretacion: str = Field(..., description="Interpretación del nivel de riesgo")
    input_data: dict = Field(..., description="Datos de entrada utilizados")


class BatchPredictionInput(BaseModel):
    """Input for batch predictions"""
    predictions: List[PredictionInput]


class ModelInfoOutput(BaseModel):
    """Model information output"""
    model_type: str
    n_features: int
    metrics: dict


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Screening Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "feature_importance": "/model/features",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict
        input_dict = input_data.dict()

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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Realizar predicciones en lote

    Permite predecir múltiples casos de manera eficiente.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", response_model=ModelInfoOutput)
async def get_model_info():
    """
    Obtener información del modelo

    Retorna información sobre el tipo de modelo, número de features y métricas de evaluación.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        info = predictor.get_model_info()
        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.get("/model/features")
async def get_feature_importance(top_n: int = 10):
    """
    Obtener las características más importantes del modelo

    Parámetros:
    - top_n: Número de características a retornar (default: 10)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if top_n < 1 or top_n > 50:
            raise HTTPException(status_code=400, detail="top_n must be between 1 and 50")

        features = predictor.get_feature_importance(top_n=top_n)

        return {
            "top_features": features,
            "count": len(features)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
