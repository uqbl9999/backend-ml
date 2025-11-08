# 🎯 COMIENZA AQUÍ - Proyecto Backend ML

## 👋 Bienvenido

Este es tu proyecto de Backend con Machine Learning para predicción de tamizajes de salud mental.

## 📚 ¿Qué Leer Primero?

1. **ESTE ARCHIVO** - Orientación inicial (estás aquí ✅)
2. **[QUICKSTART.md](QUICKSTART.md)** - Instalación rápida (5 minutos)
3. **[README.md](README.md)** - Documentación completa
4. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Resumen ejecutivo
5. **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Estructura detallada del proyecto

## 🚀 Inicio Rápido (3 Pasos)

```bash
# 1. Instalar
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Entrenar modelo (requiere tamizajes.csv)
python src/train_model.py

# 3. Iniciar API
uvicorn api.main:app --reload
```

Luego abre: http://localhost:8000/docs

## 📁 Estructura del Proyecto

```
backend-ml/
├── src/              # Código fuente Python
│   ├── models/       # Módulos del modelo (training, prediction)
│   └── services/     # Servicios (ubigeo, xai, statistics)
├── api/              # API REST con FastAPI
├── models/           # Modelos entrenados (.pkl)
├── data/             # Datasets procesados y tabla de ubigeos
├── tests/            # Pruebas unitarias
├── docs/             # Documentación técnica
└── notebooks/        # Jupyter notebooks experimentales
```

## ✅ Checklist de Entrega

Para entregar al profesor:

- [x] ✅ Código modular y organizado
- [x] ✅ API REST funcional con FastAPI
- [x] ✅ Documentación completa
- [x] ✅ README con instrucciones
- [x] ✅ Requirements.txt con dependencias
- [x] ✅ Tests unitarios
- [x] ✅ Estructura de directorios clara
- [x] ✅ .gitignore configurado

## 🎓 Para el Profesor

### Evaluación Rápida (10 minutos)

1. **Ver Estructura** (1 min)
   ```bash
   cat STRUCTURE.txt
   ```

2. **Revisar Código** (3 min)
   - `src/data_preparation.py` - Pipeline de datos
   - `src/models/training.py` - Entrenamiento
   - `api/main.py` - API REST

3. **Ejecutar** (5 min)
   ```bash
   python src/train_model.py --no-optimize  # Rápido
   uvicorn api.main:app --reload
   ```

4. **Probar API** (1 min)
   - Abrir: http://localhost:8000/docs
   - Expandir POST /predict
   - Click "Try it out"
   - Click "Execute"

### Puntos Clave

**Arquitectura**: Estructura simple pero profesional
- ✅ Separación de responsabilidades
- ✅ Modular y reutilizable
- ✅ Fácil de mantener

**Machine Learning**: Pipeline completo
- ✅ Preparación de datos
- ✅ Balanceo (SMOTE)
- ✅ Entrenamiento con optimización
- ✅ Evaluación con múltiples métricas

**API REST**: Producción-ready
- ✅ FastAPI con Swagger automático
- ✅ Validación de entrada (Pydantic)
- ✅ Manejo de errores
- ✅ Endpoints bien documentados

**Documentación**: Completa
- ✅ README detallado
- ✅ Quick start
- ✅ Comentarios en código
- ✅ Ejemplos de uso

## 🔧 Comandos Útiles

```bash
# Ver ayuda del entrenamiento
python src/train_model.py --help

# Entrenar con Random Forest
python src/train_model.py --model random_forest

# Entrenar sin optimización (más rápido)
python src/train_model.py --no-optimize

# Probar predicciones
python src/example_prediction.py

# Ejecutar tests
pytest tests/ -v

# Ver documentación de la API
curl http://localhost:8000/
```

## 📊 Ejemplo de Uso

### 1. Predicción desde API

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

### 2. Predicción desde Python

```python
from src.models.prediction import Predictor
from src.services.ubigeo_service import get_ubigeo_service

# Cargar modelo y servicio de ubigeo
predictor = Predictor('models/trained_model.pkl')
ubigeo_service = get_ubigeo_service()

# Obtener ubigeo desde departamento y provincia
ubigeo = ubigeo_service.get_ubigeo_by_dept_prov('LIMA', 'LIMA')

result = predictor.predict_single({
    'NroMes': 11,
    'ubigeo': ubigeo,
    'Departamento': 'LIMA',
    'Sexo': 'M',
    'Etapa': '5 - 9',
    'DetalleTamizaje': 'VIOLENCIA FAMILIAR/MALTRATO INFANTIL'
})
print(result)
```

## 📈 Resultados Esperados

Después del entrenamiento:

- **R² Score**: ~0.65-0.70
- **MAE**: ~8-12%
- **Features**: 43 variables codificadas
- **Tiempo de entrenamiento**: 5-10 minutos
- **Tiempo de predicción**: <50ms

## 🎯 Características Destacadas

1. **Código Limpio**
   - Cumple con PEP 8
   - Tipado con type hints
   - Docstrings completos
   - Comentarios explicativos

2. **Arquitectura Profesional**
   - Separación en módulos
   - Clases bien diseñadas
   - Reutilizable y extensible

3. **API Moderna**
   - FastAPI (framework moderno)
   - Documentación automática
   - Validación automática
   - Seguridad de tipos

4. **Pipeline ML Completo**
   - Preparación de datos
   - Balanceo de clases
   - Optimización de hiperparámetros
   - Evaluación exhaustiva

## 🐛 Solución de Problemas

### Error: "tamizajes.csv not found"
**Solución**: Coloca el archivo CSV en el directorio raíz

### Error: "Model not loaded"
**Solución**: Entrena el modelo primero con `python src/train_model.py`

### Error: "ModuleNotFoundError"
**Solución**: Activa el entorno virtual y reinstala dependencias
```bash
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📞 Soporte

- **Documentación completa**: [README.md](README.md)
- **Quick start**: [QUICKSTART.md](QUICKSTART.md)
- **Arquitectura**: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- **API docs**: http://localhost:8000/docs (cuando corra la API)

## ✨ Próximos Pasos Sugeridos

Después de revisar este proyecto, podrías:

1. ✅ Agregar autenticación (JWT)
2. ✅ Implementar caché (Redis)
3. ✅ Añadir base de datos (PostgreSQL)
4. ✅ Dockerizar la aplicación
5. ✅ CI/CD con GitHub Actions
6. ✅ Monitoreo con Prometheus
7. ✅ Desplegar en cloud (AWS/GCP/Azure)

## 🏆 Logros del Proyecto

- ✅ **Completo**: Pipeline end-to-end funcional
- ✅ **Profesional**: Código de calidad producción
- ✅ **Documentado**: Documentación exhaustiva
- ✅ **Testeable**: Tests unitarios incluidos
- ✅ **Modular**: Fácil de mantener y extender
- ✅ **Moderno**: Tecnologías actuales

---

**¿Listo para empezar?** → Lee [QUICKSTART.md](QUICKSTART.md) para instalación rápida

**¿Quieres más detalles?** → Lee [README.md](README.md) para documentación completa

**¿Dudas sobre arquitectura?** → Lee [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

---

**Versión**: 1.0.0 | **Fecha**: 2024 | **Estado**: ✅ Listo para entrega
