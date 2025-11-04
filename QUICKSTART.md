# 🚀 Quick Start Guide

Esta guía te ayudará a poner en marcha el proyecto en 5 minutos.

## ⚡ Instalación Rápida

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (requiere tamizajes.csv)
python src/train_model.py

# 4. Iniciar la API
uvicorn api.main:app --reload
```

## 🧪 Probar la API

### Opción 1: Swagger UI (Recomendado)

1. Abre tu navegador en: `http://localhost:8000/docs`
2. Expande el endpoint `/predict`
3. Click en "Try it out"
4. Usa el ejemplo pre-cargado o modifica los valores
5. Click en "Execute"

### Opción 2: cURL

```bash
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
```

### Opción 3: Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "NroMes": 11,
    "Departamento": "LIMA",
    "Provincia": "LIMA",
    "Sexo": "M",
    "Etapa": "5 - 9",
    "DetalleTamizaje": "VIOLENCIA FAMILIAR/MALTRATO INFANTIL"
}

response = requests.post(url, json=data)
print(response.json())
```

### Opción 4: Ejemplo Script

```bash
python src/example_prediction.py
```

## 📁 Archivos Necesarios

Antes de entrenar el modelo, asegúrate de tener:

- `tamizajes.csv` en el directorio raíz

Después del entrenamiento, se generarán:

- `models/trained_model.pkl` - Modelo entrenado
- `data/dataset_*.csv` - Datos procesados
- `docs/evaluation_*.png` - Gráficos de evaluación

## 🔧 Comandos Útiles

```bash
# Ver información del modelo
curl http://localhost:8000/model/info

# Ver features más importantes
curl http://localhost:8000/model/features?top_n=5

# Health check
curl http://localhost:8000/health

# Ver departamentos válidos
curl http://localhost:8000/metadata/departamentos

# Ver tipos de tamizaje válidos
curl http://localhost:8000/metadata/tamizajes

# Ver provincias de un departamento
curl http://localhost:8000/metadata/provincias/LIMA

# Obtener ubigeo de departamento y provincia
curl http://localhost:8000/metadata/ubigeo/LIMA/LIMA
```

## ⚠️ Solución de Problemas

### Error: "Model not loaded"

**Solución:** Entrena el modelo primero

```bash
python src/train_model.py
```

### Error: "tamizajes.csv not found"

**Solución:** Asegúrate de tener el archivo de datos en el directorio raíz

### Error: Módulo no encontrado

**Solución:** Verifica que el entorno virtual esté activado e instala dependencias

```bash
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

### La API no inicia

**Solución:** Verifica que el puerto 8000 no esté en uso

```bash
# Cambiar puerto si es necesario
uvicorn api.main:app --reload --port 8001
```

## 📚 Próximos Pasos

1. **Explorar la API**: Visita `http://localhost:8000/docs`
2. **Ver ejemplos**: Ejecuta `python src/example_prediction.py`
3. **Leer documentación completa**: Consulta [README.md](README.md)
4. **Personalizar el modelo**: Modifica parámetros en `src/train_model.py`

## 🎯 Endpoints Principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información general de la API |
| `/predict` | POST | Predicción individual |
| `/predict/batch` | POST | Predicciones en lote |
| `/model/info` | GET | Información del modelo |
| `/model/features` | GET | Features más importantes |
| `/metadata/departamentos` | GET | Lista de departamentos válidos |
| `/metadata/provincias/{dept}` | GET | Lista de provincias por departamento |
| `/metadata/ubigeo/{dept}/{prov}` | GET | Obtener ubigeo de dept+provincia |
| `/metadata/tamizajes` | GET | Tipos de tamizaje válidos |
| `/metadata/etapas` | GET | Grupos etarios válidos |
| `/health` | GET | Estado de la API |

## 💡 Ejemplo de Respuesta

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

## 🆘 Ayuda

Si encuentras problemas:

1. Revisa la [documentación completa](README.md)
2. Verifica que todos los requisitos estén instalados
3. Asegúrate de que el modelo esté entrenado
4. Revisa los logs de la API en la terminal

---

¡Listo! Ya tienes todo configurado para usar el sistema de predicción de salud mental. 🎉
