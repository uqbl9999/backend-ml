# Módulo de Reconocimiento de Imágenes Médicas

Este documento explica cómo usar el módulo de reconocimiento de imágenes de rayos X que clasifica imágenes en 4 categorías: COVID19, NORMAL, PNEUMONIA y TUBERCULOSIS.

## 📋 Tabla de Contenidos

1. [Configuración Inicial](#configuración-inicial)
2. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
3. [Uso de la API](#uso-de-la-api)
4. [Deploy a Producción](#deploy-a-producción)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Configuración Inicial

### 1. Preparar el Dataset

El dataset debe estar organizado de la siguiente manera:

```
data/images/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TUBERCULOSIS/
├── val/
│   ├── COVID19/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TUBERCULOSIS/
└── test/
    ├── COVID19/
    ├── NORMAL/
    ├── PNEUMONIA/
    └── TUBERCULOSIS/
```

**IMPORTANTE**: Si tu dataset tiene la carpeta `TURBERCULOSIS` (con typo), renómbrala a `TUBERCULOSIS`.

### 2. Instalar Dependencias

**Para desarrollo local (con GPU):**
```bash
pip install tensorflow==2.20.0 Pillow opencv-python-headless
```

**Para producción (solo CPU):**
Las dependencias en `requirements.txt` ya incluyen `tensorflow-cpu==2.15.0`.

---

## 🎓 Entrenamiento del Modelo

### Entrenar en tu PC con GPU

1. Navega al directorio del proyecto:
```bash
cd backend-ml
```

2. Ejecuta el script de entrenamiento:
```bash
python scripts/train_image_model.py --epochs 50 --batch-size 32
```

**Parámetros disponibles:**
- `--data-dir`: Directorio con datos (default: `data/images`)
- `--epochs`: Número de épocas (default: 50)
- `--batch-size`: Tamaño del batch (default: 32)
- `--learning-rate`: Tasa de aprendizaje (default: 0.0005)
- `--output`: Ruta del modelo (default: `models/image_models/best_model.keras`)

3. El script generará:
   - `models/image_models/best_model.keras` - Modelo entrenado
   - `models/image_models/model_metadata.json` - Métricas y metadata
   - `models/image_models/training_history.png` - Gráfico de entrenamiento
   - `models/image_models/confusion_matrix.png` - Matriz de confusión

### Verificar GPU

```python
import tensorflow as tf
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
```

---

## 🌐 Uso de la API

### Iniciar la API localmente

```bash
# Desarrollo
python api/main.py

# O con uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints Disponibles

#### 1. Predicción desde Archivo Upload

```bash
curl -X POST "http://localhost:8000/image/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

**Respuesta:**
```json
{
  "predicted_class": "COVID19",
  "confidence": 0.95,
  "interpretation": "Alta confianza - COVID19",
  "all_probabilities": {
    "COVID19": 0.95,
    "NORMAL": 0.02,
    "PNEUMONIA": 0.02,
    "TUBERCULOSIS": 0.01
  },
  "metadata": {
    "image_size": [224, 224],
    "processing_time_ms": 234,
    "filename": "chest_xray.jpg"
  }
}
```

#### 2. Predicción desde URL

```bash
curl -X POST "http://localhost:8000/image/predict-url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/xray.jpg"}'
```

#### 3. Predicción con Explicación XAI

```bash
curl -X POST "http://localhost:8000/image/predict/explain" \
  -H "accept: application/json" \
  -F "file=@chest_xray.jpg"
```

**Requiere:** Variable de entorno `PERPLEXITY_API_KEY`

**Respuesta incluye:**
```json
{
  ...
  "explicacion": {
    "contexto_clinico": "Patrón compatible con neumonía viral por COVID-19...",
    "recomendaciones": [
      "Solicitar prueba PCR para SARS-CoV-2",
      "Evaluar saturación de oxígeno",
      "Considerar aislamiento preventivo"
    ],
    "consideraciones": [
      "Confianza alta en diagnóstico",
      "Correlacionar con antecedentes"
    ]
  }
}
```

#### 4. Información del Modelo

```bash
curl -X GET "http://localhost:8000/image/model/info"
```

#### 5. Información de Clases

```bash
curl -X GET "http://localhost:8000/image/model/classes"
```

#### 6. Estadísticas del Modelo

```bash
curl -X GET "http://localhost:8000/image/model/statistics"
```

---

## 🚢 Deploy a Producción

### Flujo de Trabajo

#### 1. Entrenar Localmente (PC con GPU)

```bash
# Entrenar el modelo
python scripts/train_image_model.py --epochs 50
```

#### 2. Copiar Modelo al Proyecto

```bash
# El modelo ya está en models/image_models/best_model.keras
# Verificar que model_metadata.json también existe
ls models/image_models/
```

#### 3. Preparar para Deploy

Asegúrate de que `requirements.txt` tenga:
```txt
tensorflow-cpu==2.15.0
Pillow==10.1.0
opencv-python-headless==4.8.1.78
```

#### 4. Build con Docker

```bash
docker-compose build
docker-compose up
```

#### 5. Deploy a Render/Producción

1. Commit y push al repositorio:
```bash
git add .
git commit -m "Add image recognition module with trained model"
git push
```

2. En Render:
   - Configurar variable de entorno `PERPLEXITY_API_KEY` (opcional, para XAI)
   - El build usará `tensorflow-cpu` automáticamente
   - El modelo se cargará desde `models/image_models/best_model.keras`

#### 6. Verificar Deploy

```bash
# Health check
curl https://tu-app.onrender.com/health

# Probar endpoint de imágenes
curl https://tu-app.onrender.com/image/model/info
```

---

## 🔧 Troubleshooting

### Problema: "Servicio de imágenes no disponible"

**Causa:** El modelo no se encuentra en la ruta esperada.

**Solución:**
1. Verificar que existe `models/image_models/best_model.keras`
2. Verificar logs de startup:
```
⚠️  Advertencia: Servicio de imágenes no disponible (modelo no encontrado)
```

### Problema: Error al cargar TensorFlow

**En desarrollo local:**
```bash
pip install --upgrade tensorflow==2.20.0
```

**En producción (Render):**
Asegurar que `requirements.txt` tiene `tensorflow-cpu==2.15.0`

### Problema: "Imagen demasiado grande"

**Solución:** Las imágenes deben ser < 10 MB. Redimensionar si es necesario:
```python
from PIL import Image
img = Image.open('large_image.jpg')
img.thumbnail((2000, 2000))
img.save('resized_image.jpg')
```

### Problema: Predicción muy lenta en CPU

**Esperado:** 300-500 ms por imagen en CPU.

**Si es más lento:**
- Verificar que se usa `tensorflow-cpu` (no `tensorflow` completo)
- Verificar que el modelo se carga solo una vez (en startup)
- Considerar reducir tamaño de imagen de entrada

### Problema: Typo en nombre de carpeta (TURBERCULOSIS)

**Solución:**
```bash
cd data/images/train
mv TURBERCULOSIS TUBERCULOSIS

cd ../val
mv TURBERCULOSIS TUBERCULOSIS

cd ../test
mv TURBERCULOSIS TUBERCULOSIS
```

---

## 📊 Métricas Esperadas

Con el dataset completo y entrenamiento de 50 epochs:

- **Test Accuracy:** ~90-95%
- **Precisión por clase:**
  - COVID19: ~91%
  - NORMAL: ~95%
  - PNEUMONIA: ~89%
  - TUBERCULOSIS: ~93%

---

## 🔐 Seguridad

- Imágenes subidas se validan:
  - Formato: PNG, JPEG, JPG
  - Tamaño máximo: 10 MB
  - Content-type verificado

- Descargas desde URL:
  - Timeout de 10 segundos
  - User-Agent header incluido
  - Content-type verificado

---

## 📝 Notas Importantes

1. **Entrenamiento:**
   - Requiere GPU para ser eficiente (~2-3 horas con GPU)
   - En CPU puede tardar 10-20x más tiempo
   - Early stopping incluido (paciencia de 10 epochs)

2. **Inferencia:**
   - CPU es suficiente para producción
   - Modelo se carga una vez en memoria (caché)
   - No se recomienda batch prediction en producción (memoria limitada)

3. **XAI (Explicaciones):**
   - Opcional, requiere API key de Perplexity
   - Genera explicaciones médicas contextuales
   - Tiene fallback si API no disponible

4. **Dataset:**
   - NO se incluye en Docker build (2.6 GB)
   - Solo necesario para entrenamiento local
   - Modelo ya entrenado se incluye en deploy

---

## 🆘 Soporte

Si encuentras problemas:

1. Verificar logs de la aplicación
2. Revisar que el modelo existe en `models/image_models/`
3. Verificar que las dependencias están instaladas correctamente
4. Revisar la documentación de TensorFlow para problemas específicos de la librería
